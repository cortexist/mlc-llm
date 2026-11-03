"""Bounded, single-sequence LFM3 text backbone for d1 set-attention decisions.

No causal KV cache: global state representations depend on the questions.
The caller supplies the checkpoint's packed question IDs (-1 for state), masks
and answer positions. Readout/calibration remain checkpoint-defined host code.
"""

import dataclasses

from tvm import te
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op
from tvm_tirx import tirx

from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.support.config import ConfigBase


@dataclasses.dataclass
class D1Config(ConfigBase):
    text_config: dict = dataclasses.field(default_factory=dict)
    context_window_size: int = 256
    prefill_chunk_size: int = 256
    max_batch_size: int = 1
    tensor_parallel_shards: int = 1
    sliding_window_size: int = -1
    kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.max_batch_size != 1 or self.tensor_parallel_shards != 1:
            raise ValueError("d1 prototype requires one sequence/device")
        if self.context_window_size != self.prefill_chunk_size:
            raise ValueError("d1 requires the complete packed sequence")
        if self.text_config.get("conv_bias", False):
            raise ValueError("Biased convolution not supported")
        self.vocab_size = self.text_config["vocab_size"]


class MLP(nn.Module):
    def __init__(self, c):
        self.w1 = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.w3 = nn.Linear(c["hidden_size"], c["intermediate_size"], bias=False)
        self.w2 = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=False)

    def forward(self, x):
        return self.w2(op.silu(self.w1(x)) * self.w3(x))


class Depthwise(nn.Module):
    def __init__(self, h, width):
        self.weight = nn.Parameter((h, 1, width))
        self.width = width

    def forward(self, x):
        def compute(x, w):
            r = te.reduce_axis((0, self.width), "r")
            return te.compute(
                x.shape,
                lambda b, t, h: te.sum(
                    tirx.if_then_else(
                        t + r >= self.width - 1,
                        x[b, t + r - self.width + 1, h],
                        tirx.const(0, x.dtype),
                    )
                    * w[h, 0, r],
                    axis=r,
                ),
                name="d1_depthwise",
            )

        return op.tensor_expr_op(compute, "d1_depthwise", [x, self.weight])


class ShortConv(nn.Module):
    def __init__(self, c):
        h = c["hidden_size"]
        self.in_proj = nn.Linear(h, 3 * h, bias=False)
        self.out_proj = nn.Linear(h, h, bias=False)
        self.conv = Depthwise(h, c["conv_L_cache"])

    def forward(self, x, mask):
        x = x * op.unsqueeze(mask.astype(x.dtype), -1)
        b, gate, value = op.split(self.in_proj(x), 3, axis=-1)
        return self.out_proj(gate * self.conv(b * value))


class Attention(nn.Module):
    def __init__(self, c, kind):
        h, d = c["hidden_size"], c["head_dim"]
        self.heads, self.kv, self.dim = c["num_attention_heads"], c["num_key_value_heads"], d
        self.window = c["sliding_window"] if kind == "sliding_attention" else 0
        p = c["rope_parameters"][kind]
        self.theta, self.angles = p["rope_theta"], int(p.get("partial_rotary_factor", 1) * d) // 2
        self.q_proj = nn.Linear(h, self.heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.kv * d, bias=False)
        self.v_proj = nn.Linear(h, self.kv * d, bias=False)
        self.o_proj = nn.Linear(self.heads * d, h, bias=False)
        self.gate_proj = nn.Linear(h, self.heads, bias=False)
        self.q_norm = nn.RMSNorm(d, -1, c["norm_eps"], bias=False)
        self.k_norm = nn.RMSNorm(d, -1, c["norm_eps"], bias=False)

    def rope(self, x):
        d = self.dim

        def compute(x):
            def at(b, t, h, i):
                f = tirx.if_then_else(
                    i % (d // 2) < self.angles,
                    tirx.power(
                        tirx.const(self.theta, "float32"),
                        -(2 * (i % (d // 2))).astype("float32") / d,
                    ),
                    0.0,
                )
                angle = t.astype("float32") * f
                # HF rounds sin/cos to the activation dtype before FP32 rotation.
                co = tirx.cos(angle).astype(x.dtype).astype("float32")
                si = tirx.sin(angle).astype(x.dtype).astype("float32")
                other = x[b, t, h, (i + d // 2) % d].astype("float32")
                sign = tirx.if_then_else(i < d // 2, -1.0, 1.0)
                return (x[b, t, h, i].astype("float32") * co + other * si * sign).astype(x.dtype)

            return te.compute(x.shape, at, name="d1_rope")

        return op.tensor_expr_op(compute, "d1_rope", [x])

    def forward(self, x, mask, groups):
        _, s, _ = x.shape
        q = self.rope(self.q_norm(op.reshape(self.q_proj(x), (1, s, self.heads, self.dim))))
        k = self.rope(self.k_norm(op.reshape(self.k_proj(x), (1, s, self.kv, self.dim))))
        v = op.reshape(self.v_proj(x), (1, s, self.kv, self.dim))

        def scores(q, k, mask, groups):
            r = te.reduce_axis((0, self.dim), "d")
            dot = te.compute(
                (1, self.heads, s, s),
                lambda b, h, i, j: te.sum(
                    q[b, i, h, r].astype("float32")
                    * k[b, j, h // (self.heads // self.kv), r].astype("float32"),
                    axis=r,
                ),
            )

            def at(b, h, i, j):
                allowed = (
                    ((j <= i) & (j > i - self.window))
                    if self.window
                    else ((groups[b, i] < 0) | (groups[b, j] < 0) | (groups[b, i] == groups[b, j]))
                )
                return tirx.if_then_else(
                    (mask[b, j] != 0) & allowed,
                    dot[b, h, i, j] * self.dim**-0.5,
                    tirx.const(-3.4028234663852886e38, "float32"),
                )

            return te.compute(dot.shape, at, name="d1_mask")

        weights = op.softmax(
            op.tensor_expr_op(scores, "d1_scores", [q, k, mask, groups], attrs={"op_pattern": 8}),
            -1,
        ).astype(x.dtype)

        def values(w, v):
            r = te.reduce_axis((0, s), "s")
            return te.compute(
                (1, s, self.heads, self.dim),
                lambda b, i, h, d: te.sum(
                    w[b, h, i, r].astype("float32")
                    * v[b, r, h // (self.heads // self.kv), d].astype("float32"),
                    axis=r,
                ),
            )

        out = op.tensor_expr_op(values, "d1_values", [weights, v], attrs={"op_pattern": 8}).astype(
            x.dtype
        )
        out = out * op.unsqueeze(op.sigmoid(self.gate_proj(x)), -1)
        return self.o_proj(op.reshape(out, (1, s, self.heads * self.dim)))


class Layer(nn.Module):
    def __init__(self, c, kind):
        self.kind, self.sandwich = kind, c.get("use_sandwich_norm", True)
        if kind == "conv":
            self.conv = ShortConv(c)
        else:
            self.self_attn = Attention(c, kind)
        self.feed_forward = MLP(c)
        for name in ["operator_norm", "ffn_norm"] + (
            ["post_operator_norm", "post_ffn_norm"] if self.sandwich else []
        ):
            setattr(self, name, nn.RMSNorm(c["hidden_size"], -1, c["norm_eps"], bias=False))

    def forward(self, x, mask, groups):
        y = self.operator_norm(x)
        y = self.conv(y, mask) if self.kind == "conv" else self.self_attn(y, mask, groups)
        x = x + (self.post_operator_norm(y) if self.sandwich else y)
        y = self.feed_forward(self.ffn_norm(x))
        return x + (self.post_ffn_norm(y) if self.sandwich else y)


class Body(nn.Module):
    def __init__(self, c):
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = nn.ModuleList([Layer(c, k) for k in c["layer_types"]])
        self.embedding_norm = nn.RMSNorm(c["hidden_size"], -1, c["norm_eps"], bias=False)


class D1Text(nn.Module):
    def __init__(self, config):
        self.config = config
        self.language_model = Body(config.text_config)
        self.dtype = "float32"

    def prefill(self, input_ids, attention_mask, question_ids):
        m = self.language_model
        x = m.embed_tokens(input_ids)
        for layer in m.layers:
            x = layer(x, attention_mask, question_ids)
        return m.embedding_norm(x)

    def get_default_spec(self):
        n = self.config.context_window_size
        return nn.spec.ModuleSpec.from_raw(
            {
                "prefill": {
                    "input_ids": nn.spec.Tensor([1, n], "int32"),
                    "attention_mask": nn.spec.Tensor([1, n], "int32"),
                    "question_ids": nn.spec.Tensor([1, n], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


huggingface = make_standard_hf_loader(model_cls=D1Text, include_qkv=False, include_gate_up=False)
