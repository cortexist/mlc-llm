"""Compare the register-resident gated delta kernel with the TIR kernel on sm_72.

Both are measured against a float64 evaluation of the same recurrence, and timed.
"""

import time
from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.model.qwen35.qwen35_model import create_gated_delta_net_func
from mlc_llm.op.xavier import gated_delta

KEY_HEADS, VALUE_HEADS, DIM = 16, 48, 128


class Probe(nn.Module):
    def fast(self, q, k, v, gate, beta, state):
        return gated_delta(q, k, v, gate, beta, state)

    def reference(self, q, k, v, gate, beta, state):
        batch, steps, _, _ = v.shape
        return op.tensor_ir_op(
            create_gated_delta_net_func(KEY_HEADS, VALUE_HEADS, DIM, DIM, "float16"),
            "gated_delta_net",
            [q, k, v, gate, beta, state],
            [
                Tensor.placeholder([batch, steps, VALUE_HEADS, DIM], "float32"),
                Tensor.placeholder([batch, VALUE_HEADS, DIM, DIM], "float32"),
            ],
        )


def exact(q, k, v, gate, beta, state):
    """The recurrence in float64: decay, predict, delta update, read out."""
    group = VALUE_HEADS // KEY_HEADS
    q, k, v = (np.asarray(x, "float64") for x in (q, k, v))
    state = np.asarray(state, "float64").copy()
    out = np.zeros(v.shape)
    for t in range(v.shape[1]):
        key, query = np.repeat(k[:, t], group, axis=1), np.repeat(q[:, t], group, axis=1)
        state *= gate[:, t, :, None, None]
        error = v[:, t] - np.einsum("bhrc,bhr->bhc", state, key)
        state += key[..., None] * (beta[:, t, :, None] * error)[:, :, None, :]
        out[:, t] = np.einsum("bhrc,bhr->bhc", state, query) / np.sqrt(DIM)
    return out, state


target = Target({"kind": "cuda", "arch": "sm_72"}, host="llvm")
arguments = {
    "q": nn.spec.Tensor(["batch", "steps", KEY_HEADS, DIM], "float16"),
    "k": nn.spec.Tensor(["batch", "steps", KEY_HEADS, DIM], "float16"),
    "v": nn.spec.Tensor(["batch", "steps", VALUE_HEADS, DIM], "float16"),
    "gate": nn.spec.Tensor(["batch", "steps", VALUE_HEADS], "float32"),
    "beta": nn.spec.Tensor(["batch", "steps", VALUE_HEADS], "float32"),
    "state": nn.spec.Tensor(["batch", VALUE_HEADS, DIM, DIM], "float32"),
    "$": {"param_mode": "none", "effect_mode": "none"},
}
mod, _, external = Probe().export_tvm(
    spec={"fast": arguments, "reference": arguments}, allow_extern=True
)
mod = AttachExternModules(external)(mod.with_attr("external_mods", []))
exe = relax.build(mod, target=target)
path = Path("dist/libs/xavier-gdn-test.so")
path.parent.mkdir(parents=True, exist_ok=True)
exe.export_library(str(path), options=["-L/usr/local/cuda/lib64", "-lcudart"])
dev = tvm_tirx.cuda(0)
vm = relax.VirtualMachine(tvm_ffi.load_module(str(path.resolve())), dev)
rng = np.random.default_rng(3527)


def unit(shape):
    x = rng.normal(size=shape)
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype("float16")


for batch, steps in [(1, 1), (1, 7), (2, 33), (1, 129), (1, 512)]:
    inputs = [
        unit((batch, steps, KEY_HEADS, DIM)),
        unit((batch, steps, KEY_HEADS, DIM)),
        rng.normal(0, 0.5, (batch, steps, VALUE_HEADS, DIM)).astype("float16"),
        rng.uniform(0.5, 1, (batch, steps, VALUE_HEADS)).astype("float32"),
        rng.uniform(0, 1, (batch, steps, VALUE_HEADS)).astype("float32"),
        rng.normal(0, 0.1, (batch, VALUE_HEADS, DIM, DIM)).astype("float32"),
    ]
    on_device = [tvm_tirx.runtime.tensor(x, device=dev) for x in inputs]
    expected = exact(*inputs)
    result, seconds = {}, {}
    for name in ["reference", "fast"]:
        for _ in range(3):
            dev.sync()
            start = time.perf_counter()
            outputs = vm[name](*on_device)
            dev.sync()
            seconds[name] = time.perf_counter() - start
        result[name] = [x.numpy() for x in outputs]
        assert all(x.dtype == np.float32 and np.isfinite(x).all() for x in result[name])
    report = dict(batch=batch, steps=steps)
    for i, part in enumerate(["output", "state"]):
        scale = np.abs(expected[i]).max()
        error = {n: float(np.abs(result[n][i] - expected[i]).max() / scale) for n in result}
        between = float(np.abs(result["fast"][i] - result["reference"][i]).max() / scale)
        # Summation order alone separates the two: a few FP32 rounding steps of the scale.
        assert error["fast"] <= max(4 * error["reference"], 1e-5), (part, error)
        assert between <= 1e-5, (part, between)
        report[part] = dict(error, between=between)
    report["us_per_step"] = {n: round(seconds[n] / steps * 1e6, 1) for n in seconds}
    print(report, flush=True)
