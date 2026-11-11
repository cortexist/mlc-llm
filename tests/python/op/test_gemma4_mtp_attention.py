"""Check the E4B assistant attention kernel on sm_72 against the exported arithmetic.

The reference rounds scores and weights to FP16 and accumulates in FP32, as the
TIR path does. The kernel adds in a different order, so outputs are compared
within two FP16 steps and the bit-equal share is printed. Keys and values
outside the valid range are NaN: reading one would poison the result.
"""

from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.op.gemma4_mtp import shared_attention

CAPACITY = 16384


class Probe(nn.Module):
    def local(self, q, k, v, n):
        return shared_attention(q, k, v, n, 512)

    def full(self, q, k, v, n):
        return shared_attention(q, k, v, n, 0)


def signature(dim):
    return {
        "q": nn.spec.Tensor([1, 2, 2, dim], "float16"),
        "k": nn.spec.Tensor([1, CAPACITY, 2, dim], "float16"),
        "v": nn.spec.Tensor([1, CAPACITY, 2, dim], "float16"),
        "n": nn.spec.Tensor([1], "int32"),
        "$": {"param_mode": "none", "effect_mode": "none"},
    }


def reference(q, k, v, n, window):
    begin = max(0, n - window - 1) if window else 0
    k, v = k[0, begin:n].astype("float32"), v[0, begin:n].astype("float32")
    scores = np.einsum("hgd,shd->hgs", q[0].astype("float32"), k).astype("float16")
    scores = scores.astype("float32")
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = (weights / weights.sum(-1, keepdims=True)).astype("float16").astype("float32")
    return np.einsum("hgs,shd->hgd", weights, v).astype("float16")[None]


target = Target({"kind": "cuda", "arch": "sm_72"}, host="llvm")
spec = {"local": signature(256), "full": signature(512)}
mod, _, external = Probe().export_tvm(spec=spec, allow_extern=True)
mod = AttachExternModules(external)(mod.with_attr("external_mods", []))
exe = relax.build(mod, target=target)
path = Path("dist/libs/gemma4-mtp-attention-test.so")
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), dev)
rng = np.random.default_rng(1111)
for name, dim, window in (("local", 256, 512), ("full", 512, 0)):
    for n in (1, 2, 63, 64, 65, 513, 514, 700, 4096, CAPACITY):
        q = (rng.standard_normal((1, 2, 2, dim)) * 0.3).astype("float16")
        k = rng.standard_normal((1, CAPACITY, 2, dim)).astype("float16")
        v = rng.standard_normal((1, CAPACITY, 2, dim)).astype("float16")
        begin = max(0, n - window - 1) if window else 0
        k[0, :begin] = k[0, n:] = v[0, :begin] = v[0, n:] = np.nan
        args = [tvm_tirx.runtime.tensor(x, device=dev) for x in (q, k, v, np.array([n], "int32"))]
        output = vm[name](*args).numpy()
        again = vm[name](*args).numpy()
        expected = reference(q, k, v, n, window)
        np.testing.assert_array_equal(output.view("uint16"), again.view("uint16"))
        error = np.abs(output.astype("float32") - expected.astype("float32"))
        step = np.spacing(np.abs(expected)).astype("float32")
        assert np.isfinite(output).all() and (error <= 2 * step).all(), (name, n, error.max())
        print(
            {
                "function": name,
                "length": n,
                "bit_equal": round(float((output.view("uint16") == expected.view("uint16")).mean()), 4),
                "max_error": float(error.max()),
            },
            flush=True,
        )
