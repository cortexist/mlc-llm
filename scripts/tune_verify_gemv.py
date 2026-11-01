"""Measure q4f16_1 small-batch GEMV schedules at E4B's dominant shapes, with numerical checks."""

import argparse
import inspect
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import tvm
import tvm_ffi
import tvm_tirx
from tvm import te
from tvm.s_tir import dlight as dl
from tvm.s_tir.dlight.gpu import low_batch_gemv
from tvm_tirx import tirx
from tvm_tirx.target import Target
from mlc_llm.quantization import QUANTIZATION

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--shapes", default="20480:2560,2560:10240,262144:2560")
p.add_argument("--rows", type=int, default=3)
args = p.parse_args()
device = tvm_ffi.device("cuda", 0)
target = Target(
    {
        "kind": "cuda",
        "arch": "sm_72",
        "max_num_threads": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    },
    host="llvm",
)
quant = QUANTIZATION["q4f16_1"]
rng = np.random.default_rng(123)
original = low_batch_gemv.LowBatchGEMV.sch_inner_reduction
source = textwrap.dedent(inspect.getsource(original))
results = []

for shape in args.shapes.split(","):
    n, k = map(int, shape.split(":"))
    qw = rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32)
    scales = rng.uniform(0.005, 0.02, (n, k // 32)).astype("float16")
    inputs = rng.normal(0, 0.2, (args.rows, 1, k)).astype("float16")
    sample = np.linspace(0, n - 1, min(128, n), dtype=int)
    unpacked = ((qw[sample, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(
        len(sample), k
    )
    dequant = (
        (unpacked.astype("float16") - np.float16(7)) * np.repeat(scales[sample], 32, axis=1)
    ).astype("float16")
    ref = (dequant.astype("float32") @ inputs.reshape(args.rows, k).T.astype("float32")).T
    tensors = [tvm_tirx.runtime.tensor(x, device=device) for x in (qw, scales, inputs)]
    output = tvm_tirx.runtime.empty((args.rows, 1, n), "float16", device)
    for bucket, ts, tr in [
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (4, 0, 0),
        (4, 4, 32),
        (4, 8, 32),
        (4, 16, 32),
    ]:
        label = f"b{bucket}-s{ts}-r{tr}"
        patched = source
        if ts:
            patched = patched.replace("TS, TR = 4, 64", f"TS, TR = {ts}, {tr}").replace(
                "TS, TR = 16, 32", f"TS, TR = {ts}, {tr}"
            )
        ns = {}
        exec(patched, low_batch_gemv.__dict__, ns)
        low_batch_gemv.LowBatchGEMV.sch_inner_reduction = ns["sch_inner_reduction"]
        batch = tirx.Var("batch_size", "int64")
        W = te.placeholder((n, k // 8), "uint32", name="q_weight")
        S = te.placeholder((n, k // 32), "float16", name="q_scale")
        A = te.placeholder((batch, 1, k), "float16", name="activation")
        D = quant._dequantize(W, S, 1, [n, k])
        r = te.reduce_axis((0, k), name="k")
        Y = te.compute(
            (batch, 1, n), lambda b, t, o: te.sum(A[b, t, r] * D[o, r], axis=r), name="NT_matmul"
        )
        f = te.create_function([W, S, A, Y]).with_attr("global_symbol", "main")
        mod = tvm_tirx.IRModule({"main": f})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(bucket))(mod)
            built = tvm.compile(mod, target=target).jit()
        for _ in range(4):
            built["main"](*tensors, output)
        device.sync()
        y = output.numpy().reshape(args.rows, n)[:, sample]
        err = float(np.max(np.abs(y - ref)))
        assert np.allclose(y, ref, rtol=0.015, atol=0.025), (shape, label, err)
        times = []
        for rep in range(3):
            device.sync()
            t = time.perf_counter()
            for _ in range(50):
                built["main"](*tensors, output)
            device.sync()
            times.append((time.perf_counter() - t) / 50)
        seconds = min(times)
        row = dict(
            rows=args.rows,
            n=n,
            k=k,
            schedule=label,
            ms=seconds * 1000,
            gb_s=(qw.nbytes + scales.nbytes) / seconds / 1e9,
            max_abs_error=err,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
low_batch_gemv.LowBatchGEMV.sch_inner_reduction = original
