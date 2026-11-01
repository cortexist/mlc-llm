"""Compile isolated prefill candidates without recompiling the model."""

import argparse
import subprocess

from tvm.relax.frontend import nn

from mlc_llm.op.xavier_prefill import ROOT, build_tir

p = argparse.ArgumentParser()
p.add_argument("--native-only", action="store_true")
args = p.parse_args()
folder = ROOT / "dist/prefill-experiments"
folder.mkdir(parents=True, exist_ok=True)
flags = [x for x in nn.SourceModule.get_compile_options("cu") if x != "-c"]
subprocess.run(
    [
        "nvcc",
        "-shared",
        *flags,
        "-O3",
        "-arch=sm_72",
        str(ROOT / "python/mlc_llm/op/cuda/xavier_prefill.cu"),
        "-o",
        str(folder / "native.so"),
    ],
    check=True,
)
for variant in (() if args.native_only else ("bounds", "width")):
    print("Building", variant, flush=True)
    print(build_tir(variant), flush=True)
