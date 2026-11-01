"""Check the native-width callback against TVM and independent FP32 attention."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import tvm
import tvm_ffi
import tvm_tirx
from tvm.relax.frontend.nn.llm.kv_cache import _attention_prefill
from tvm_tirx.target import Target

from mlc_llm.op.xavier_attention import source_module

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
args = p.parse_args()
target = Target(
    {"kind": "cuda", "arch": "sm_72", "max_shared_memory_per_block": 49152}, host="llvm"
)
functions = {}
for local in [0, 1]:
    name = f"baseline{local}"
    functions[name] = _attention_prefill(
        2, 8, 512, "float16", bool(local), {}, target, sliding_window_size=512
    ).with_attr("global_symbol", name)
reference = tvm.compile(tvm_tirx.IRModule(functions), target=target).jit()
module = source_module()
flags = [x for x in module.compile_options if x != "-c"]
src = Path("python/mlc_llm/op/cuda/xavier_attention.cu")
library = Path("dist/libs/xavier-attention-test.so")
subprocess.run(["nvcc", "-shared", *flags, str(src), "-o", str(library)], check=True)
native = tvm_ffi.load_module(str(library.resolve()))
fast = {
    local: native["xavier_wrap_attention"](reference[f"baseline{local}"], local) for local in [0, 1]
}
os.environ["MLC_XAVIER_ATTN_OFF"] = "1"
disabled = {
    local: native["xavier_wrap_attention"](reference[f"baseline{local}"], local) for local in [0, 1]
}
del os.environ["MLC_XAVIER_ATTN_OFF"]
dev = tvm_tirx.cuda(0)


def tensor(a):
    return tvm_tirx.runtime.tensor(a, device=dev)


rng = np.random.default_rng(903)
results = []


def timing(fn, inputs):
    for _ in range(3):
        fn(*inputs)
    dev.sync()
    start = time.perf_counter()
    for _ in range(20):
        fn(*inputs)
    dev.sync()
    return (time.perf_counter() - start) * 50


cases = [(length, rows, 0, 0, 1, 1) for length in [161, 448, 512, 1768] for rows in [1, 3]]
cases += [(448, 3, 7, 4, 1, 1), (448, 32, 0, 0, 1, 1), (0, 3, 0, 0, 1, 1), (1768, 3, 0, 0, 0, 1)]
cases += [(length, 3, 0, 0, causal, 0) for length in [161, 512, 1768] for causal in [0, 1]]
for context, rows, offset, sink, causal, local in cases:
    width = 256 if local else 512
    span = context + offset - sink
    pages_count = (span + 15) // 16
    physical = rng.permutation(pages_count).astype("int32")
    pages = np.zeros((max(pages_count, 1), 2, 2, 16, 512), "float16")
    kv = rng.normal(0, 0.2, (2, context, 2, width)).astype("float16")
    for j in range(context):
        slot = j if j < sink else j - sink + offset
        pages[physical[slot // 16], :, :, slot % 16, :width] = kv[:, j]
    q = np.zeros((rows, 8, 512), "float16")
    q[:, :, :width] = rng.normal(0, 0.2, (rows, 8, width)).astype("float16")
    length = np.array(
        [[span - (pages_count - 1) * 16 if pages_count else 0], [offset], [sink]], "int32"
    )
    if not local:
        length = length[0].copy()
    inputs = [
        tensor(q),
        tensor(np.array([0, rows], "int32")),
        tensor(pages),
        tensor(np.array([0, pages_count], "int32")),
        tensor(physical),
        tensor(length),
        tensor(np.zeros(1, "int32")),
        tensor(np.arange(rows, dtype="int32")),
        tvm_tirx.runtime.empty(q.shape, "float16", dev),
        tvm_tirx.runtime.empty((rows, 8), "float32", dev),
        causal,
        0,
        1.0,
        10000.0,
        1.0,
    ]
    reference[f"baseline{local}"](*inputs)
    old, old_lse = inputs[8].numpy(), inputs[9].numpy()
    fast[local](*inputs)
    output, lse = inputs[8].numpy(), inputs[9].numpy()
    np.testing.assert_allclose(output, old, rtol=0.002, atol=0.0003)
    np.testing.assert_allclose(lse, old_lse, rtol=2e-5, atol=0.0003)
    if local:
        np.testing.assert_array_equal(output[:, :, 256:], 0)
    expected = np.zeros_like(q, dtype="float32")
    expected_lse = np.full((rows, 8), -50000.0, "float32")
    for row in range(rows):
        first = max(0, context - max(512 - row - 1, 0)) if local and causal else 0
        last = min(context, context - rows + row + 1) if not local and causal else context
        for head in range(8):
            if first >= last:
                continue
            k = kv[0, first:last, head // 4].astype("float32")
            v = kv[1, first:last, head // 4].astype("float32")
            logits = k @ q[row, head, :width].astype("float32")
            maximum = logits.max()
            prob = np.exp(logits - maximum)
            expected[row, head, :width] = (prob / prob.sum()) @ v
            expected_lse[row, head] = (maximum + np.log(prob.sum())) / np.log(2)
    np.testing.assert_allclose(output, expected, rtol=0.002, atol=0.0003)
    np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=0.0003)
    disabled[local](*inputs)
    np.testing.assert_array_equal(inputs[8].numpy(), old)
    record = dict(
        context=context,
        queries=rows,
        offset=offset,
        sink=sink,
        causal=causal,
        local=local,
        baseline_ms=timing(reference[f"baseline{local}"], inputs),
        native_ms=timing(fast[local], inputs),
        max_abs_error=float(np.max(np.abs(output.astype("float32") - old))),
    )
    results.append(record)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(record), flush=True)
