"""Compare the register-resident gated delta kernel with the TIR kernel on sm_72.

Both are measured against a float64 evaluation of the same recurrence, and timed. The commit
kernel must reproduce, bit for bit, the state after the first rows of a chunk.
"""

import time
from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm_ffi import Shape
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_external_modules import AttachExternModules
from mlc_llm.model.qwen35.qwen35_model import create_gated_delta_net_func
from mlc_llm.op.xavier import commit_rows, gated_delta, pack_rows

KEY_HEADS, VALUE_HEADS, DIM, TAPS = 16, 48, 128, 3
CHANNELS = (2 * KEY_HEADS + VALUE_HEADS) * DIM
WIDTH = CHANNELS + (KEY_HEADS + VALUE_HEADS) * DIM + 2 * VALUE_HEADS


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

    def pack(self, conv_input, k, v, gate, beta):
        return pack_rows(conv_input, k, v, gate, beta)

    def commit(self, held, rows, state, conv_state):
        return commit_rows(held, 1, rows, state, conv_state)


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
none = {"param_mode": "none", "effect_mode": "none"}
chunk = {
    "k": nn.spec.Tensor(["batch", "steps", KEY_HEADS, DIM], "float16"),
    "v": nn.spec.Tensor(["batch", "steps", VALUE_HEADS, DIM], "float16"),
    "gate": nn.spec.Tensor(["batch", "steps", VALUE_HEADS], "float32"),
    "beta": nn.spec.Tensor(["batch", "steps", VALUE_HEADS], "float32"),
}
forward = {
    "q": nn.spec.Tensor(["batch", "steps", KEY_HEADS, DIM], "float16"),
    **chunk,
    "state": nn.spec.Tensor(["batch", VALUE_HEADS, DIM, DIM], "float32"),
    "$": none,
}
spec = {
    "fast": forward,
    "reference": forward,
    "pack": {
        "conv_input": nn.spec.Tensor([1, "steps", CHANNELS], "float16"),
        **{name: nn.spec.Tensor([1, *x.shape[1:]], x.dtype) for name, x in chunk.items()},
        "$": none,
    },
    "commit": {
        "held": nn.spec.Tensor([2, "steps", WIDTH], "float32"),
        "rows": int,
        "state": nn.spec.Tensor([1, VALUE_HEADS, DIM, DIM], "float32"),
        "conv_state": nn.spec.Tensor([1, TAPS, CHANNELS], "float16"),
        "$": none,
    },
}
mod, _, external = Probe().export_tvm(spec=spec, allow_extern=True)
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


def sample(batch, steps):
    return [
        unit((batch, steps, KEY_HEADS, DIM)),
        unit((batch, steps, KEY_HEADS, DIM)),
        rng.normal(0, 0.5, (batch, steps, VALUE_HEADS, DIM)).astype("float16"),
        rng.uniform(0.5, 1, (batch, steps, VALUE_HEADS)).astype("float32"),
        rng.uniform(0, 1, (batch, steps, VALUE_HEADS)).astype("float32"),
        rng.normal(0, 0.1, (batch, VALUE_HEADS, DIM, DIM)).astype("float32"),
    ]


def call(name, *inputs):
    outputs = vm[name](
        *(x if isinstance(x, Shape) else tvm_tirx.runtime.tensor(x, device=dev) for x in inputs)
    )
    return [x.numpy() for x in outputs] if not hasattr(outputs, "numpy") else outputs.numpy()


for batch, steps in [(1, 1), (1, 7), (2, 33), (1, 129), (1, 512)]:
    inputs = sample(batch, steps)
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

# Held rows: layer 0 is noise, layer 1 the rows whose first `rows` are committed.
for steps in (4, 7):
    q, k, v, gate, beta, state = sample(1, steps)
    conv_input = rng.normal(0, 1, (1, steps, CHANNELS)).astype("float16")
    conv_state = rng.normal(0, 1, (1, TAPS, CHANNELS)).astype("float16")
    packed = call("pack", conv_input, k, v, gate, beta)
    parts = [x.astype("float32").reshape(1, steps, -1) for x in (conv_input, k, v, gate, beta)]
    np.testing.assert_array_equal(packed, np.concatenate(parts, axis=-1))
    held = np.concatenate([rng.normal(size=packed.shape).astype("float32"), packed])
    for rows in range(1, steps + 1):
        new_state, new_conv = call("commit", held, Shape([rows]), state, conv_state)
        _, forward_state = call("fast", *(x[:, :rows] for x in (q, k, v, gate, beta)), state)
        np.testing.assert_array_equal(new_state, forward_state)
        history = np.concatenate([conv_state[0], conv_input[0, :rows]])
        np.testing.assert_array_equal(new_conv[0], history[rows : rows + TAPS])
    print(dict(commit_steps=steps, bit_exact=True), flush=True)
