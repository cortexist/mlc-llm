"""Independent float32 PyTorch reference for the converted Nemotron draft."""

from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
w = {k: v.float() for k, v in load_file(str(root / "dist/draft-hf/model.safetensors")).items()}
rng = np.random.default_rng(29)
noise = rng.normal(0, 1, (1, 3, 2560)).astype("float16")
features = rng.normal(0, 5, (1, 161, 12800)).astype("float16")
h = torch.from_numpy(noise).float()
features = torch.from_numpy(features).float()


def linear(x, name):
    return torch.nn.functional.linear(x, w[name + ".weight"])


def norm(x, name):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w[name + ".weight"]


def stat(name, x):
    print(
        name, "max_abs", x.abs().max().item(), "finite", torch.isfinite(x).all().item(), flush=True
    )


angles = torch.arange(164).float()[:, None] / (1e6 ** (torch.arange(64).float() / 64))
angles = torch.cat([angles, angles], dim=-1)[None, :, None, :]
cos, sin = angles.cos(), angles.sin()


def rope(x, start, stop):
    left, right = x.chunk(2, dim=-1)
    return x * cos[:, start:stop] + torch.cat([-right, left], dim=-1) * sin[:, start:stop]


context = norm(linear(features, "fc"), "hidden_norm")
stat("context", context)
for i in range(5):
    layer = f"layers.{i}"
    a = layer + ".self_attn"
    x = norm(h, layer + ".input_layernorm")
    q = norm(linear(x, a + ".q_proj").reshape(1, 3, 32, 128), a + ".q_norm")
    keys = torch.cat([linear(context, a + ".k_proj"), linear(x, a + ".k_proj")], dim=1)
    k = norm(keys.reshape(1, 164, 8, 128), a + ".k_norm")
    values = torch.cat([linear(context, a + ".v_proj"), linear(x, a + ".v_proj")], dim=1)
    v = values.reshape(1, 164, 8, 128).transpose(1, 2).repeat_interleave(4, dim=1)
    q = rope(q, 161, 164).transpose(1, 2)
    k = rope(k, 0, 164).transpose(1, 2).repeat_interleave(4, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (128**-0.5)
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 3, 4096)
    h = h + linear(out, a + ".o_proj")
    stat(f"layer{i}.attention_residual", h)
    x = norm(h, layer + ".post_attention_layernorm")
    gate = linear(x, layer + ".mlp.gate_proj")
    up = linear(x, layer + ".mlp.up_proj")
    activated = torch.nn.functional.silu(gate) * up
    stat(f"layer{i}.gate_product", activated)
    h = h + linear(activated, layer + ".mlp.down_proj")
    stat(f"layer{i}.mlp_residual", h)
out = norm(h, "norm")
stat("output", out)
np.savez(
    root / "logs/draft-reference.npz", features=features.numpy(), noise=noise, output=out.numpy()
)
