"""Exercise the actual Relax/FFI boundary, scratch planning, and output dtypes."""

from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.op.xavier import q4_linear


class Probe(nn.Module):
    def forward(self, x, w, s):
        return q4_linear(x, w, s), q4_linear(x, w, s, "float32")


target = Target({"kind": "cuda", "arch": "sm_72"}, host="llvm")
spec = {
    "forward": {
        "x": nn.spec.Tensor([1, "rows", 2560], "float16"),
        "w": nn.spec.Tensor([137, 320], "uint32"),
        "s": nn.spec.Tensor([137, 80], "float16"),
        "$": {"param_mode": "none", "effect_mode": "none"},
    }
}
mod, _, external = Probe().export_tvm(spec=spec, allow_extern=True)
mod = AttachExternModules(external)(mod.with_attr("external_mods", []))
exe = relax.build(mod, target=target)
path = Path("dist/libs/xavier-q4-test.so")
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), dev)
rng = np.random.default_rng(91)
w = rng.integers(0, 2**32, (137, 320), dtype="uint32")
s = rng.uniform(0.005, 0.02, (137, 80)).astype("float16")
unpacked = ((w[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(137, 2560)
weight = ((unpacked.astype("float16") - np.float16(7)) * np.repeat(s, 32, axis=1)).astype("float16")


def tensor(x):
    return tvm_tirx.runtime.tensor(x, device=dev)


for b in [1, 2, 3, 4]:
    for zero in [False, True]:
        x = (
            np.zeros((1, b, 2560), "float16")
            if zero
            else rng.normal(0, 0.2, (1, b, 2560)).astype("float16")
        )
        expected = x.astype("float32") @ weight.astype("float32").T
        y16, y32 = [y.numpy() for y in vm["forward"](tensor(x), tensor(w), tensor(s))]
        assert y16.dtype == np.float16 and y32.dtype == np.float32
        np.testing.assert_array_equal(y16, y32.astype("float16"))
        assert np.isfinite(y32).all()
        error = np.sqrt(np.mean((y32 - expected) ** 2))
        relative = float(error / max(float(np.sqrt(np.mean(expected**2))), 1e-9))
        assert relative < 0.015, (b, zero, relative)
        if zero:
            np.testing.assert_array_equal(y32, expected)
        print(dict(rows=b, zero=zero, relative_rmse=relative), flush=True)
