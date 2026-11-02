"""FP32, text-only EmbeddingGemma 2 encoder for bounded resident-model trials."""

import dataclasses
import math

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op
from tvm_tirx import tirx

from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.support.config import ConfigBase

from .gemma4_audio import Gemma4RMSNorm
from .gemma4_model import Gemma4TextRotaryEmbedding


@dataclasses.dataclass
class EmbeddingConfig(ConfigBase):
    text_config: dict = dataclasses.field(default_factory=dict)
    context_window_size: int = 128
    prefill_chunk_size: int = 128
    max_batch_size: int = 1
    tensor_parallel_shards: int = 1
    sliding_window_size: int = -1
    kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.max_batch_size != 1 or self.tensor_parallel_shards != 1:
            raise ValueError("Embedding PoC supports one sequence on one device")
        if self.prefill_chunk_size != self.context_window_size:
            raise ValueError("Bidirectional encoder uses one fixed padded input bucket")
        self.vocab_size = self.text_config["vocab_size"]


class MLP(nn.Module):
    def __init__(self, c):
        h, n = c["hidden_size"], c["intermediate_size"]
        self.gate_proj = nn.Linear(h, n, bias=False)
        self.up_proj = nn.Linear(h, n, bias=False)
        self.down_proj = nn.Linear(n, h, bias=False)

    def forward(self, x):
        return self.down_proj(op.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class PLE(nn.Module):
    def __init__(self, c):
        self.layers, self.width = c["num_hidden_layers"], c["hidden_size_per_layer_input"]
        self.scale = c["hidden_size"] ** -0.5
        self.per_layer_model_projection = nn.Linear(
            c["hidden_size"], self.layers * self.width, bias=False
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(self.width, c["rms_norm_eps"])

    def forward(self, x):
        b, s, _ = x.shape
        return self.per_layer_projection_norm(
            op.reshape(
                self.per_layer_model_projection(x) * self.scale, (b, s, self.layers, self.width)
            )
        )


class PLEBlock(nn.Module):
    def __init__(self, c):
        h, p = c["hidden_size"], c["hidden_size_per_layer_input"]
        self.per_layer_input_gate = nn.Linear(h, p, bias=False)
        self.per_layer_projection = nn.Linear(p, h, bias=False)
        self.post_per_layer_input_norm = Gemma4RMSNorm(h, c["rms_norm_eps"])

    def forward(self, x, ple):
        gated = op.gelu(self.per_layer_input_gate(x), approximate="tanh") * ple
        return x + self.post_per_layer_input_norm(self.per_layer_projection(gated))


class Attention(nn.Module):
    def __init__(self, c, layer):
        override = c.get("per_layer_config", {}).get(
            f"{layer:02d}", c.get("per_layer_config", {}).get(str(layer), {})
        )
        self.dim = override.get("head_dim", c["head_dim"])
        self.kv = override.get("num_key_value_heads", c["num_key_value_heads"])
        self.heads = c["num_attention_heads"]
        self.window = c["sliding_window"] if c["layer_types"][layer] == "sliding_attention" else 0
        h, d = c["hidden_size"], self.dim
        self.q_proj = nn.Linear(h, self.heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.kv * d, bias=False)
        self.v_proj = nn.Linear(h, self.kv * d, bias=False)
        self.o_proj = nn.Linear(self.heads * d, h, bias=False)
        self.q_norm = Gemma4RMSNorm(d, c["rms_norm_eps"])
        self.k_norm = Gemma4RMSNorm(d, c["rms_norm_eps"])
        self.v_norm = Gemma4RMSNorm(d, c["rms_norm_eps"], with_scale=False)

        # The shared RoPE implementation supports a full rotary dimension for
        # both layer types when partial_rotary_factor is omitted.
        class RopeConfig:
            layer_types = c["layer_types"]
            rope_parameters = c["rope_parameters"]

            def head_dim_for_layer(_, index):
                return d

        self.rope = Gemma4TextRotaryEmbedding(RopeConfig(), layer)

    def forward(self, x, mask, positions):
        _, s, _ = x.shape
        q = self.q_norm(op.reshape(self.q_proj(x), (1, s, self.heads, self.dim)))
        k = self.k_norm(op.reshape(self.k_proj(x), (1, s, self.kv, self.dim)))
        v = self.v_norm(op.reshape(self.v_proj(x), (1, s, self.kv, self.dim)))
        q, k = self.rope(q, k, positions)
        q = op.reshape(q, (1, s, self.kv, self.heads // self.kv, self.dim))

        def scores(q, k, mask):
            r = te.reduce_axis((0, self.dim), name="d")
            dot = te.compute(
                (1, self.kv, self.heads // self.kv, s, s),
                lambda b, h, g, i, j: te.sum(q[b, i, h, g, r] * k[b, j, h, r], axis=r),
                name="eg2_dot",
            )
            return te.compute(
                dot.shape,
                lambda b, h, g, i, j: tirx.if_then_else(
                    (mask[b, j] != 0)
                    & ((tirx.abs(i - j) <= self.window) if self.window else (j >= 0)),
                    dot[b, h, g, i, j],
                    tirx.const(-3.4028234663852886e38, "float32"),
                ),
                name="eg2_mask",
            )

        weights = op.softmax(
            op.tensor_expr_op(scores, "eg2_scores", [q, k, mask], attrs={"op_pattern": 8}), axis=-1
        )

        def values(w, v):
            r = te.reduce_axis((0, s), name="s")
            return te.compute(
                (1, s, self.kv, self.heads // self.kv, self.dim),
                lambda b, i, h, g, d: te.sum(w[b, h, g, i, r] * v[b, r, h, d], axis=r),
                name="eg2_values",
            )

        output = op.tensor_expr_op(values, "eg2_values", [weights, v], attrs={"op_pattern": 8})
        return self.o_proj(op.reshape(output, (1, s, self.heads * self.dim)))


class Layer(nn.Module):
    def __init__(self, c, i):
        self.self_attn = Attention(c, i)
        self.mlp = MLP(c)
        self.ple_block = PLEBlock(c)
        self.layer_scalar = nn.Parameter((1,))
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ]:
            setattr(self, name, Gemma4RMSNorm(c["hidden_size"], c["rms_norm_eps"]))

    def forward(self, x, ple, mask, positions):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask, positions)
        )
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return self.ple_block(x, ple) * self.layer_scalar


class TextBody(nn.Module):
    def __init__(self, c):
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Layer(c, i) for i in range(c["num_hidden_layers"])])
        self.ple = PLE(c)
        self.norm = Gemma4RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.embedding_projection = nn.Linear(c["hidden_size"], c["embedding_dim"], bias=False)


class EmbeddingGemma2(nn.Module):
    def __init__(self, config):
        self.config = config
        self.language_model = TextBody(config.text_config)
        self.dtype = "float32"

    def to(self, dtype=None):
        if dtype not in (None, "float32"):
            raise ValueError("EmbeddingGemma 2 PoC requires FP32 on Xavier")
        super().to(dtype=dtype)

    def prefill(self, input_ids, attention_mask):
        m = self.language_model
        hidden = m.embed_tokens(input_ids) * math.sqrt(self.config.text_config["hidden_size"])
        ple = m.ple(hidden)
        positions = op.arange(0, input_ids.shape[1], dtype="int32")
        slices = op.split(ple, self.config.text_config["num_hidden_layers"], axis=2)
        for layer, p in zip(m.layers, slices):
            hidden = layer(hidden, op.squeeze(p, axis=2), attention_mask, positions)
        return m.embedding_projection(m.norm(hidden))

    def get_default_spec(self):
        n = self.config.context_window_size
        return nn.spec.ModuleSpec.from_raw(
            {
                "prefill": {
                    "input_ids": nn.spec.Tensor([1, n], "int32"),
                    "attention_mask": nn.spec.Tensor([1, n], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


huggingface = make_standard_hf_loader(
    model_cls=EmbeddingGemma2, include_qkv=False, include_gate_up=False
)
