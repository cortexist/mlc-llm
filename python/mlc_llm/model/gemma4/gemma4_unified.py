"""Text-only Gemma4 Unified 12B, using the existing token-generation ABI.

The uniform MLC cache stores eight KV heads per layer. Global layers broadcast
their single KV head into those slots; this preserves attention semantics but
uses more cache memory than a cache supporting different head counts per layer.
"""

import dataclasses
import math

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm.loader.standard_loader import make_standard_hf_loader

from ..gemma.gemma_model import GemmaEmbedding
from .gemma4_audio import Gemma4RMSNorm
from .gemma4_config import Gemma4Config, Gemma4TextConfig, _as_config_dict
from .gemma4_model import (
    _TEXT_ONLY,
    Gemma4ForConditionalGeneration,
    Gemma4TextAttention,
    Gemma4TextDecoderLayer,
    Gemma4TextMLP,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    _pad_head_dim,
    _slice_head_dim,
)


@dataclasses.dataclass
class Gemma4UnifiedTextConfig(Gemma4TextConfig):
    """The dense 12B variant; E-series configurations retain their own model."""

    def __post_init__(self):
        if not _TEXT_ONLY:
            raise ValueError("Gemma4 Unified requires MLC_GEMMA4_TEXT_ONLY=1")
        if self.enable_moe_block or self.num_kv_shared_layers or self.hidden_size_per_layer_input:
            raise ValueError("Unified support requires dense layers without KV sharing or PLE")
        if (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_global_key_value_heads,
        ) != (16, 8, 1):
            raise ValueError("Unified support requires 16 query, 8 local KV and 1 global KV heads")
        if (self.head_dim, self.global_head_dim) != (256, 512) or not self.attention_k_eq_v:
            raise ValueError("Unified support requires 256/512 heads and global K=V projections")
        if not self.tie_word_embeddings or self.attention_bias or self.attention_dropout:
            raise ValueError("Unsupported Unified embedding or attention configuration")
        if self.use_bidirectional_attention not in (None, "vision"):
            raise ValueError("Only causal text attention is supported")
        if self.hidden_activation not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError("Only GeLU is supported")
        if self.layer_types is None or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("Expected explicit Unified layer types")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported Unified attention type")
        if self.rope_parameters is None:
            raise ValueError("Expected explicit Unified RoPE configuration")


@dataclasses.dataclass
class Gemma4UnifiedConfig(Gemma4Config):
    def __post_init__(self):
        self.text_config = Gemma4UnifiedTextConfig.from_dict(
            _as_config_dict(self.text_config, self.kwargs)
        )
        if self.tensor_parallel_shards != 1:
            raise ValueError("Unified supports one tensor-parallel shard")
        if self.context_window_size <= 0:
            self.context_window_size = self.text_config.max_position_embeddings
        self.sliding_window_size = -1
        self.prefill_chunk_size = min(
            self.prefill_chunk_size if self.prefill_chunk_size > 0 else 512,
            self.text_config.sliding_window,
        )


class UnifiedAttention(Gemma4TextAttention):
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim_for_layer(layer_idx)
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = (
            config.num_global_key_value_heads
            if self.layer_type == "full_attention"
            else config.num_key_value_heads
        )
        self.cache_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = (
            None
            if self.layer_type == "full_attention"
            else nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config, layer_idx)

    def forward(self, hidden_states, paged_kv_cache, positions, shared_kv):
        batch, seq_len, _ = hidden_states.shape
        query = self.q_norm(
            op.reshape(
                self.q_proj(hidden_states), (batch, seq_len, self.num_q_heads, self.head_dim)
            )
        )
        raw_key = self.k_proj(hidden_states)
        raw_value = self.v_proj(hidden_states) if self.v_proj is not None else raw_key
        key = self.k_norm(op.reshape(raw_key, (batch, seq_len, self.num_kv_heads, self.head_dim)))
        value = self.v_norm(
            op.reshape(raw_value, (batch, seq_len, self.num_kv_heads, self.head_dim))
        )
        query, key = self.rotary_emb(query, key, positions)
        shared_kv = (key, value)
        if self.num_kv_heads != self.cache_heads:
            key = op.broadcast_to(key, (batch, seq_len, self.cache_heads, self.head_dim))
            value = op.broadcast_to(value, (batch, seq_len, self.cache_heads, self.head_dim))
        qkv = op.concat(
            [
                _pad_head_dim(query, self.head_dim),
                _pad_head_dim(key, self.head_dim),
                _pad_head_dim(value, self.head_dim),
            ],
            dim=2,
        )
        output = paged_kv_cache.attention_with_fused_qkv(
            self.layer_idx, qkv, self.num_q_heads, sm_scale=1.0
        )
        output = op.reshape(
            _slice_head_dim(output, self.head_dim),
            (batch, seq_len, self.num_q_heads * self.head_dim),
        )
        return self.o_proj(output), shared_kv


class UnifiedDecoderLayer(Gemma4TextDecoderLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = UnifiedAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps))
        self.layer_scalar = nn.Parameter((1,))

    def forward(self, hidden_states, per_layer_input, paged_kv_cache, positions, shared_kv):
        attention, current_kv = self.self_attn(
            self.input_layernorm(hidden_states), paged_kv_cache, positions, None
        )
        hidden_states = hidden_states + self.post_attention_layernorm(attention)
        hidden_states = hidden_states + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden_states))
        )
        return hidden_states * self.layer_scalar, current_kv


class UnifiedTextModel(Gemma4TextModel):
    def __init__(self, config):
        self.config = config
        self.embed_tokens = GemmaEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [UnifiedDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.shared_kv_source_layers = set()
        self.norm = Gemma4RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed(self, input_ids):
        # The reference embedding multiplier is cast to the embedding dtype.
        return self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

    def _per_layer_inputs(self, input_embeds, token_ids, modality_ids):
        return [None] * self.config.num_hidden_layers


class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    def __init__(self, config):
        self.config = config
        self.language_model = UnifiedTextModel(config.text_config)
        self.dflash_target_layer_ids = tuple(config.kwargs.get("dflash_target_layer_ids", ()))
        if self.dflash_target_layer_ids and (
            list(self.dflash_target_layer_ids) != sorted(set(self.dflash_target_layer_ids))
            or min(self.dflash_target_layer_ids) < 0
            or max(self.dflash_target_layer_ids) >= config.text_config.num_hidden_layers
        ):
            raise ValueError("DFlash target layers must be unique, ascending, valid layer indices")
        self.dtype = "float32"


def huggingface(config, quantization):
    loader = make_standard_hf_loader(
        model_cls=Gemma4UnifiedForConditionalGeneration,
        layer_prefix="language_model.layers",
        include_qkv=False,
        include_gate_up=True,
        gate_up_target_name="gate_up_proj",
        name_transform=lambda name: f"model.{name}",
        num_layers_getter=lambda cfg: cfg.text_config.num_hidden_layers,
    )
    mapping = loader(config, quantization)
    for name in (
        "embed_audio.embedding_projection.weight",
        "embed_vision.embedding_projection.weight",
        "vision_embedder.patch_dense.bias",
        "vision_embedder.patch_dense.weight",
        "vision_embedder.patch_ln1.bias",
        "vision_embedder.patch_ln1.weight",
        "vision_embedder.patch_ln2.bias",
        "vision_embedder.patch_ln2.weight",
        "vision_embedder.pos_embedding",
        "vision_embedder.pos_norm.bias",
        "vision_embedder.pos_norm.weight",
    ):
        mapping.add_unused(f"model.{name}")
    return mapping
