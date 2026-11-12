"""Compare custom small-batch CUDA kernels against sampled FP32 dot products."""

import argparse
import ctypes as C
import hashlib
import json
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--library", default="dist/libs/xavier-q4-probe.so")
p.add_argument("--output", required=True)
p.add_argument("--repeats", type=int, default=40)
p.add_argument(
    "--modes",
    default="2,5",
    help="0=FP16, 1=untiled Q8, 2..5=Q8 tiles of 1/2/4/8 rows, 6..9=the same tiles by groups of 32",
)
p.add_argument("--rows", default="1,2,3,4")
p.add_argument(
    "--shapes",
    default="20480x2560,2560x10240,262144x2560",
    help="outputs x inputs of the matrices; the default is Nemotron's MLP and LM head",
)
args = p.parse_args()
cuda = C.CDLL("libcudart.so")
cuda.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
cuda.cudaFree.argtypes = [C.c_void_p]
lib = C.CDLL(str(Path(args.library).resolve()))
run = lib.xavier_q4_run
run.argtypes = [C.c_void_p] * 6 + [C.c_int] * 5 + [C.c_void_p]


def check(code):
    assert code == 0, code


def allocate(array):
    ptr = C.c_void_p()
    check(cuda.cudaMalloc(C.byref(ptr), array.nbytes))
    check(cuda.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, 1))
    return ptr


rng = np.random.default_rng(123)
results = []
for n, k in (map(int, shape.split("x")) for shape in args.shapes.split(",")):
    w = rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32)
    scales = rng.uniform(0.005, 0.02, (n, k // 32)).astype("float16")
    sample = np.linspace(0, n - 1, 128, dtype=int)
    unpacked = ((w[sample, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(128, k)
    dequant = (
        (unpacked.astype("float16") - np.float16(7)) * np.repeat(scales[sample], 32, axis=1)
    ).astype("float16")
    pw, ps = allocate(w), allocate(scales)
    for b in map(int, args.rows.split(",")):
        x = rng.normal(0, 0.2, (b, k)).astype("float16")
        ref = x.astype("float32") @ dequant.astype("float32").T
        y = np.empty((b, n), "float16")
        ptrs = [
            allocate(a)
            for a in [x, y, np.empty((b, k // 2), "uint32"), np.empty((b, k // 32, 2), "float32")]
        ]
        px, py, pq, pa = ptrs
        for mode in map(int, args.modes.split(",")):

            def launch():
                check(run(pw, ps, px, py, pq, pa, n, k, b, mode, 0, None))

            for _ in range(4):
                launch()
            check(cuda.cudaDeviceSynchronize())
            check(cuda.cudaMemcpy(y.ctypes.data, py, y.nbytes, 2))
            actual = y[:, sample].astype("float32")
            error = actual - ref
            rmse = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(ref**2)))
            assert np.isfinite(y).all()
            assert rmse < (0.015 if mode else 0.001), (n, k, b, mode, rmse)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(args.repeats):
                    launch()
                check(cuda.cudaDeviceSynchronize())
                times.append((time.perf_counter() - start) * 1000 / args.repeats)
            row = dict(
                n=n,
                k=k,
                rows=b,
                mode=mode,
                ms=min(times),
                relative_rmse=rmse,
                max_abs_error=float(np.max(np.abs(error))),
                # The tile of a Q8 mode chooses the outputs of a warp, not their arithmetic.
                sha256=hashlib.sha256(y.tobytes()).hexdigest(),
            )
            results.append(row)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        for ptr in ptrs:
            check(cuda.cudaFree(ptr))
    check(cuda.cudaFree(pw))
    check(cuda.cudaFree(ps))
