"""Isolated q4 expansion checks and FP32-accumulation cuBLASLt candidate timings."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import tvm
import tvm_ffi
import tvm_tirx
from tvm import te
from tvm.relax.frontend import nn
from tvm.s_tir import dlight as dl

from mlc_llm.op.xavier_prefill import ROOT, TARGET
from mlc_llm.quantization import QUANTIZATION

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--gemm", action="store_true")
p.add_argument("--rows", type=int, nargs="+", default=[448])
p.add_argument("--shapes", default="137:2560,20480:2560,2560:10240,2048:2560,10752:2560")
a = p.parse_args()
dev = tvm_tirx.cuda(0)
folder = ROOT / "dist/linear-experiments"
folder.mkdir(exist_ok=True)
flags = [x for x in nn.SourceModule.get_compile_options("cu") if x != "-c"]
subprocess.run(
    [
        "nvcc",
        "-shared",
        *flags,
        "-O3",
        "-arch=sm_72",
        str(ROOT / "python/mlc_llm/op/cuda/xavier_dequant.cu"),
        "-o",
        str(folder / "dequant.so"),
    ],
    check=True,
)
native = tvm_ffi.load_module(str(folder / "dequant.so"))
if a.gemm:
    subprocess.run(
        [
            "nvcc",
            "-shared",
            *flags,
            "-O3",
            "-arch=sm_72",
            str(ROOT / "python/mlc_llm/op/cuda/xavier_dense.cu"),
            "-lcublas",
            "-lcublasLt",
            "-o",
            str(folder / "dense.so"),
        ],
        check=True,
    )
    dense = tvm_ffi.load_module(str(folder / "dense.so"))
rng = np.random.default_rng(902)


def tensor(v):
    return tvm_tirx.runtime.tensor(v, device=dev)


def timing(fn):
    for _ in range(2):
        fn()
    dev.sync()
    start = time.perf_counter()
    for _ in range(5):
        fn()
    dev.sync()
    return (time.perf_counter() - start) * 200


results = []
for shape in a.shapes.split(","):
    n, k = map(int, shape.split(":"))
    w = rng.integers(0, 2**32, (n, k // 8), dtype="uint32")
    s = rng.uniform(0.001, 0.03, (n, k // 32)).astype("float16")
    dw, ds = tensor(w), tensor(s)
    o = tvm_tirx.runtime.empty((n, k), "float16", dev)
    wr = te.placeholder(w.shape, "uint32", "w")
    sr = te.placeholder(s.shape, "float16", "s")
    out = QUANTIZATION["q4f16_1"]._dequantize(wr, sr, axis=-1, out_shape=[n, k])
    func = te.create_function([wr, sr, out]).with_attr("global_symbol", "reference")
    with TARGET:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(tvm_tirx.IRModule({"reference": func}))
    ref = tvm.compile(mod, target=TARGET).jit()["reference"]
    ref(dw, ds, o)
    reference = o.numpy()
    native["xavier_dequant"](dw, ds, o)
    np.testing.assert_array_equal(o.numpy(), reference)
    # Independent nibble/scale reference over tail rows, including odd N.
    unpack = ((w[-17:, :, None] >> (np.arange(8, dtype="uint32") * 4)) & 15).reshape(-1, k)
    cpu = (unpack.astype("float16") - np.float16(7)) * np.repeat(s[-17:], 32, axis=1)
    np.testing.assert_array_equal(reference[-17:], cpu)
    row = {
        "n": n,
        "k": k,
        "dequant_bit_exact": True,
        "reference_ms": timing(lambda: ref(dw, ds, o)),
        "vector_ms": timing(lambda: native["xavier_dequant"](dw, ds, o)),
    }
    if a.gemm and n != 137:
        row["gemm"] = []
        for m in a.rows:
            x = rng.normal(0, 0.2, (m, k)).astype("float16")
            dx = tensor(x)
            y = tvm_tirx.runtime.empty((m, n), "float16", dev)
            count = dense["gemm_probe"](dx, o, y, -1)
            dense["gemm_probe"](dx, o, y, 0)
            base = y.numpy()
            cpu = x[:2].astype("float32") @ reference[:32].astype("float32").T
            np.testing.assert_allclose(base[:2, :32], cpu, rtol=0.003, atol=0.002)
            variants = []
            for idx in [*range(count), 32]:
                dense["gemm_probe"](dx, o, y, idx)
                output = y.numpy()
                err = float(np.max(np.abs(output.astype("float32") - base.astype("float32"))))
                valid = bool(np.allclose(output, base, rtol=0.003, atol=0.002))
                variants.append(
                    {
                        "index": idx,
                        "ms": timing(lambda: dense["gemm_probe"](dx, o, y, idx)),
                        "max_abs_error": err,
                        "within_tolerance": valid,
                        "config": {
                            name: dense["gemm_config"](m, n, k, idx, attribute)
                            for attribute, name in enumerate(
                                [
                                    "id",
                                    "tile",
                                    "split_k",
                                    "reduction",
                                    "swizzle",
                                    "custom",
                                    "stages",
                                ]
                            )
                        }
                        if idx != 32
                        else None,
                    }
                )
            best = min((v for v in variants if v["within_tolerance"]), key=lambda v: v["ms"])
            confirmations = []
            for idx in [0, best["index"], 0, best["index"]]:
                confirmations.append(
                    {"index": idx, "ms": timing(lambda: dense["gemm_probe"](dx, o, y, idx))}
                )
            row["gemm"].append({"m": m, "variants": variants, "confirmation": confirmations})
    results.append(row)
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
