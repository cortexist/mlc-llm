"""Official Gemma4 Unified MTP assistant and explicit-state target exports.

The first MTP driver uses these compiled programs directly. It is restricted to
    one sequence and greedy generation. Local attention masks the shared history.
Assistant KV inputs are frozen throughout a draft round and position is fixed.
"""

import dataclasses

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op
from tvm_tirx import tirx

from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.model.model_utils import index_last_token
from mlc_llm.nn import PagedKVCache
from mlc_llm.op import xavier
from mlc_llm.support.config import ConfigBase

from .gemma4_audio import Gemma4RMSNorm
from .gemma4_config import Gemma4TextConfig, _as_config_dict
from .gemma4_model import Gemma4TextMLP, Gemma4TextRotaryEmbedding
from .gemma4_unified import Gemma4UnifiedForConditionalGeneration


@dataclasses.dataclass
class AssistantTextConfig(Gemma4TextConfig):
    def __post_init__(self):
        if self.num_kv_shared_layers != self.num_hidden_layers or self.hidden_size_per_layer_input:
            raise ValueError("MTP assistant must use shared KV in every layer and no PLE")
        if self.enable_moe_block or self.attention_bias or self.attention_dropout:
            raise ValueError("Only dense, bias-free causal-text MTP is supported")
        if (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_global_key_value_heads,
        ) not in ((16, 8, 1), (4, 2, 2)):
            raise ValueError("Expected Unified 12B or E4B assistant attention heads")
        if (self.head_dim, self.global_head_dim) != (256, 512):
            raise ValueError("Expected 256/512 local/global head dimensions")
        if len(self.layer_types or []) != self.num_hidden_layers or self.rope_parameters is None:
            raise ValueError("Explicit layer types and RoPE settings are required")
        if self.use_double_wide_mlp or self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError("Expected the Unified assistant's ordinary GeLU MLP")


@dataclasses.dataclass
class Gemma4UnifiedAssistantConfig(ConfigBase):
    text_config: dict | AssistantTextConfig = dataclasses.field(default_factory=dict)
    backbone_hidden_size: int = 3840
    use_ordered_embeddings: bool = False
    context_window_size: int = 4096
    prefill_chunk_size: int = 512
    max_batch_size: int = 1
    tensor_parallel_shards: int = 1
    sliding_window_size: int = -1
    kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.text_config = AssistantTextConfig.from_dict(_as_config_dict(self.text_config, {}))
        if self.use_ordered_embeddings:
            raise ValueError("Ordered/centroid heads are not implemented by this Unified port")
        if self.tensor_parallel_shards != 1 or self.max_batch_size != 1:
            raise ValueError("The MTP prototype supports one device and one sequence")
        if not 0 < self.context_window_size <= self.text_config.max_position_embeddings:
            raise ValueError("Invalid MTP context window")

    @property
    def vocab_size(self):
        return self.text_config.vocab_size


class SharedAttention(nn.Module):
    def __init__(self, config, layer):
        self.window = (
            config.sliding_window if config.layer_types[layer] == "sliding_attention" else 0
        )
        self.head_dim = config.head_dim_for_layer(layer)
        self.heads = config.num_attention_heads
        self.kv_heads = (
            config.num_global_key_value_heads
            if config.layer_types[layer] == "full_attention"
            else config.num_key_value_heads
        )
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config, layer)

    def forward(self, hidden, key, value, position, context_length):
        query = self.q_norm(op.reshape(self.q_proj(hidden), (1, 1, self.heads, self.head_dim)))
        query = self.rotary_emb.apply_query(query, position)
        query = op.reshape(
            op.permute_dims(query, (0, 2, 1, 3)),
            (1, self.kv_heads, self.heads // self.kv_heads, self.head_dim),
        )
        key = op.permute_dims(key, (0, 2, 3, 1))

        # Keep strided, dynamic-context attention out of the cuBLAS-Lt dense
        # projection dispatcher (CUDA 11.4 misidentifies its batch dimensions).
        def score_product(q, k):
            r = te.reduce_axis((0, self.head_dim), name="head_dim")
            return te.compute(
                (1, self.kv_heads, self.heads // self.kv_heads, k.shape[-1]),
                lambda b, h, g, s: te.sum(
                    q[b, h, g, r].astype("float32") * k[b, h, r, s].astype("float32"),
                    axis=r,
                ),
                name="mtp_attention_scores",
            )

        scores = op.tensor_expr_op(
            score_product, "mtp_attention_scores", [query, key], attrs={"op_pattern": 8}
        )
        scores = scores.astype(hidden.dtype)

        def mask_scores(scores, length):
            return te.compute(
                scores.shape,
                lambda b, h, g, s: tirx.if_then_else(
                    (s < length[0]) & (s >= length[0] - self.window - 1 if self.window else s >= 0),
                    scores[b, h, g, s],
                    tirx.const(float("-inf"), scores.dtype),
                ),
                name="mtp_context_mask",
            )

        scores = op.tensor_expr_op(mask_scores, "mtp_context_mask", [scores, context_length])
        weights = op.softmax(scores.astype("float32"), axis=-1).astype(hidden.dtype)
        value = op.permute_dims(value, (0, 2, 1, 3))

        def value_product(w, v):
            r = te.reduce_axis((0, v.shape[-2]), name="context")
            return te.compute(
                (1, self.kv_heads, self.heads // self.kv_heads, self.head_dim),
                lambda b, h, g, d: te.sum(
                    w[b, h, g, r].astype("float32") * v[b, h, r, d].astype("float32"),
                    axis=r,
                ),
                name="mtp_attention_values",
            )

        result = op.tensor_expr_op(
            value_product, "mtp_attention_values", [weights, value], attrs={"op_pattern": 8}
        )
        result = result.astype(hidden.dtype)
        result = op.reshape(result, (1, 1, self.heads * self.head_dim))
        return self.o_proj(result)


class AssistantLayer(nn.Module):
    def __init__(self, config, layer):
        self.self_attn = SharedAttention(config, layer)
        self.mlp = Gemma4TextMLP(config, layer)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.layer_scalar = nn.Parameter((1,))

    def forward(self, hidden, key, value, position, context_length):
        hidden = hidden + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(hidden), key, value, position, context_length)
        )
        hidden = hidden + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden))
        )
        return hidden * self.layer_scalar


class AssistantBody(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList(
            [AssistantLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)


class Gemma4UnifiedAssistantForCausalLM(nn.Module):
    def __init__(self, config):
        self.config = config
        text = config.text_config
        self.model = AssistantBody(text)
        self.pre_projection = nn.Linear(
            2 * config.backbone_hidden_size, text.hidden_size, bias=False
        )
        self.post_projection = nn.Linear(text.hidden_size, config.backbone_hidden_size, bias=False)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.dtype = "float32"

    def to(self, dtype=None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def mtp_step(
        self,
        embedding,
        previous_hidden,
        local_key,
        local_value,
        global_key,
        global_value,
        position,
        context_length,
    ):
        hidden = self.pre_projection(op.concat([embedding, previous_hidden], dim=-1))
        for i, layer in enumerate(self.model.layers):
            key, value = (
                (global_key, global_value)
                if self.config.text_config.layer_types[i] == "full_attention"
                else (local_key, local_value)
            )
            hidden = layer(hidden, key, value, position, context_length)
        hidden = self.model.norm(hidden)
        return self.lm_head(hidden).astype("float32"), self.post_projection(hidden)

    def get_default_spec(self):
        hidden = self.config.backbone_hidden_size
        context = self.config.context_window_size
        return nn.spec.ModuleSpec.from_raw(
            {
                "mtp_step": {
                    "embedding": nn.spec.Tensor([1, 1, hidden], self.dtype),
                    "previous_hidden": nn.spec.Tensor([1, 1, hidden], self.dtype),
                    "local_key": nn.spec.Tensor([1, context, 8, 256], self.dtype),
                    "local_value": nn.spec.Tensor([1, context, 8, 256], self.dtype),
                    "global_key": nn.spec.Tensor([1, context, 1, 512], self.dtype),
                    "global_value": nn.spec.Tensor([1, context, 1, 512], self.dtype),
                    "position": nn.spec.Tensor([1], "int32"),
                    "context_length": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


def assistant_huggingface(config, quantization):
    mapping = make_standard_hf_loader(
        model_cls=Gemma4UnifiedAssistantForCausalLM,
        layer_prefix="model.layers",
        include_qkv=False,
        gate_up_target_name="gate_up_proj",
        num_layers_getter=lambda cfg: cfg.text_config.num_hidden_layers,
    )(config, quantization)
    dtype = quantization.model_dtype
    mapping.add_mapping(
        "lm_head.weight", ["model.embed_tokens.weight"], lambda weight: weight.astype(dtype)
    )
    return mapping


class Gemma4UnifiedMTPTarget(Gemma4UnifiedForConditionalGeneration):
    """Target-only exports for the explicit-state experimental MTP driver."""

    def _mtp_forward(self, tokens, cache):
        hidden = self.language_model.embed(tokens)
        positions = cache.get_query_positions(tokens.shape[0] * tokens.shape[1])
        state = {}
        for i, layer in enumerate(self.language_model.layers):
            hidden, kv = layer(hidden, None, cache, positions, None)
            state[self.config.text_config.layer_types[i]] = kv
        hidden = self.language_model.norm(hidden)
        return hidden, *state["sliding_attention"], *state["full_attention"]

    def mtp_prefill(self, tokens, cache):
        hidden, *states = self._mtp_forward(tokens, cache)
        return self.get_logits(index_last_token(hidden)), hidden, *states, cache

    @xavier.small_batch()
    def mtp_verify(self, tokens, cache):
        hidden, *states = self._mtp_forward(tokens, cache)
        return self.get_logits(hidden), hidden, *states, cache

    def get_default_spec(self):
        packed = {"param_mode": "packed", "effect_mode": "none"}
        cache = nn.spec.Object(object_type=PagedKVCache)
        return nn.spec.ModuleSpec.from_raw(
            {
                "embed": {"input_ids": nn.spec.Tensor(["seq_len"], "int32"), "$": packed},
                "mtp_prefill": {
                    "tokens": nn.spec.Tensor([1, "seq_len"], "int32"),
                    "cache": cache,
                    "$": packed,
                },
                "mtp_verify": {
                    "tokens": nn.spec.Tensor([1, "verify_len"], "int32"),
                    "cache": cache,
                    "$": packed,
                },
                "create_paged_kv_cache": {
                    "max_batch_size": int,
                    "max_total_seq_len": int,
                    "prefill_chunk_size": int,
                    "page_size": int,
                    "support_sliding_window": int,
                    "$": {"param_mode": "none", "effect_mode": "none"},
                },
            },
            self,
        )
