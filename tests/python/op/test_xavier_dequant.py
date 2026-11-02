"""Check the exported Relax/FFI boundary and exact q4f16_1 FP16 rounding on sm_72."""

from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.op.xavier import dequant_weight


class Probe(nn.Module):
    def forward(self, w, s):
        return dequant_weight(w, s)


target = Target({"kind": "cuda", "arch": "sm_72"}, host="llvm")
spec = {
    "forward": {
        "w": nn.spec.Tensor([137, 320], "uint32"),
        "s": nn.spec.Tensor([137, 80], "float16"),
        "$": {"param_mode": "none", "effect_mode": "none"},
    }
}
mod, _, external = Probe().export_tvm(spec=spec, allow_extern=True)
mod = AttachExternModules(external)(mod.with_attr("external_mods", []))
exe = relax.build(mod, target=target)
path = Path("dist/libs/xavier-dequant-test.so")
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), dev)
rng = np.random.default_rng(921)
w = rng.integers(0, 2**32, (137, 320), dtype="uint32")
unpack = ((w[:, :, None] >> (np.arange(8, dtype="uint32") * 4)) & 15).reshape(137, 2560)
for special in (False, True):
    s = rng.uniform(0.001, 0.03, (137, 80)).astype("float16")
    if special:
        s[:] = np.resize(np.array([0, -0.0, 2**-24, 2**-14, 1, -1, 4096], "float16"), s.shape)
    expected = (unpack.astype("float16") - np.float16(7)) * np.repeat(s, 32, axis=1)
    output = vm["forward"](
        tvm_tirx.runtime.tensor(w, device=dev), tvm_tirx.runtime.tensor(s, device=dev)
    ).numpy()
    np.testing.assert_array_equal(output.view("uint16"), expected.view("uint16"))
    print({"special_scales": special, "bit_exact": True}, flush=True)
