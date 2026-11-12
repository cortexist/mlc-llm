"""Check the assistant attention kernel on sm_72 against the exported arithmetic.

The reference rounds scores and weights to FP16 and accumulates in FP32, as the
TIR path does. The kernel adds in a different order: outputs are compared
within two FP16 steps, plus what a score rounded to the neighboring FP16 value
does to its weight, and the bit-equal share is printed. Keys and values outside
the valid range are NaN: reading one would poison the result.
--output stores a hash of every output, to compare two builds of the kernel.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.op.gemma4_mtp import shared_attention

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layout", choices=["e4b", "unified"], default="e4b")
parser.add_argument("--output", type=Path)
arguments = parser.parse_args()
# K/V heads and query heads per K/V head, of a local and of a global layer.
LOCAL, FULL, WINDOW, CAPACITY = {
    "e4b": ((2, 2), (2, 2), 512, 16384),
    "unified": ((8, 2), (1, 16), 1024, 4096),
}[arguments.layout]


class Probe(nn.Module):
    def local(self, q, k, v, n):
        return shared_attention(q, k, v, n, WINDOW)

    def full(self, q, k, v, n):
        return shared_attention(q, k, v, n, 0)


def signature(heads, group, dim):
    return {
        "q": nn.spec.Tensor([1, heads, group, dim], "float16"),
        "k": nn.spec.Tensor([1, CAPACITY, heads, dim], "float16"),
        "v": nn.spec.Tensor([1, CAPACITY, heads, dim], "float16"),
        "n": nn.spec.Tensor([1], "int32"),
        "$": {"param_mode": "none", "effect_mode": "none"},
    }


def reference(q, k, v, n, window):
    begin = max(0, n - window - 1) if window else 0
    k, v = k[0, begin:n].astype("float32"), v[0, begin:n].astype("float32")
    rounded = np.einsum("hgd,shd->hgs", q[0].astype("float32"), k).astype("float16")
    scores = rounded.astype("float32")
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = (weights / weights.sum(-1, keepdims=True)).astype("float16").astype("float32")
    # One FP16 step of a score scales its weight by that step.
    moved = weights * np.spacing(np.abs(rounded)).astype("float32")
    slack = np.einsum("hgs,shd->hgd", moved, np.abs(v))[None]
    return np.einsum("hgs,shd->hgd", weights, v).astype("float16")[None], slack


target = Target({"kind": "cuda", "arch": "sm_72"}, host="llvm")
spec = {"local": signature(*LOCAL, 256), "full": signature(*FULL, 512)}
mod, _, external = Probe().export_tvm(spec=spec, allow_extern=True)
mod = AttachExternModules(external)(mod.with_attr("external_mods", []))
exe = relax.build(mod, target=target)
path = Path("dist/libs/gemma4-mtp-attention-test.so")
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), dev)
rng = np.random.default_rng(1111)
records = []
for name, (heads, group), dim, window in (("local", LOCAL, 256, WINDOW), ("full", FULL, 512, 0)):
    for n in sorted({1, 2, 63, 64, 65, 513, 514, 700, WINDOW + 1, WINDOW + 2, 4096, CAPACITY}):
        q = (rng.standard_normal((1, heads, group, dim)) * 0.3).astype("float16")
        k = rng.standard_normal((1, CAPACITY, heads, dim)).astype("float16")
        v = rng.standard_normal((1, CAPACITY, heads, dim)).astype("float16")
        begin = max(0, n - window - 1) if window else 0
        k[0, :begin] = k[0, n:] = v[0, :begin] = v[0, n:] = np.nan
        args = [tvm_tirx.runtime.tensor(x, device=dev) for x in (q, k, v, np.array([n], "int32"))]
        output = vm[name](*args).numpy()
        again = vm[name](*args).numpy()
        expected, slack = reference(q, k, v, n, window)
        np.testing.assert_array_equal(output.view("uint16"), again.view("uint16"))
        error = np.abs(output.astype("float32") - expected.astype("float32"))
        step = np.spacing(np.abs(expected)).astype("float32")
        assert np.isfinite(output).all() and (error <= 2 * step + slack).all(), (
            name,
            n,
            float((error - 2 * step - slack).max()),
        )
        records.append(
            {
                "function": name,
                "length": n,
                "bit_equal": round(
                    float((output.view("uint16") == expected.view("uint16")).mean()), 4
                ),
                "max_error": float(error.max()),
                "beyond_two_steps": int((error > 2 * step).sum()),
                "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            }
        )
        print(records[-1], flush=True)
if arguments.output:
    arguments.output.write_text(json.dumps(records, indent=1) + "\n")
