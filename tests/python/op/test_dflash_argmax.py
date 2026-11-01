"""Correctness and transfer-inclusive timing for DFlash's GPU greedy selection."""

import argparse
import json
import time

import numpy as np
import tvm
import tvm_tirx
from tvm import relax
from tvm_tirx.target import Target

from mlc_llm.compiler_pass.attach_sampler import _attach_argmax_logits


def run(arch="sm_72", repeats=50):
    device = tvm_tirx.cuda(0)
    target = Target({"kind": "cuda", "arch": arch}, host="llvm")
    bb = relax.BlockBuilder()
    _attach_argmax_logits(bb)
    mod = bb.finalize()
    vm = relax.VirtualMachine(relax.build(mod, target=target), device)
    rng = np.random.default_rng(42)
    results = []
    for rows, vocab in [(1, 33), (3, 257), (3, 262144), (8, 262144), (16, 151936)]:
        logits = rng.standard_normal((rows, vocab)).astype("float32")
        logits[0, :] = -np.inf  # Fully masked rows still select the first index.
        if rows > 1:
            logits[1, 7] = logits[1, 19] = np.inf  # Deterministic tie breaking.
        if rows > 2:
            logits[2, 2] = logits[2, vocab - 1] = 100
        data = tvm_tirx.runtime.tensor(logits, device=device)
        expected = np.argmax(logits, axis=-1).astype("int32")
        for _ in range(3):
            np.testing.assert_array_equal(vm["argmax_logits"](data).numpy(), expected)
        timings = {}
        # numpy's optimized host argmax is a conservative comparison against the
        # fork's scalar C++ loop. Both include the required GPU-to-CPU copy.
        for name, fn in {
            "cpu_copy_argmax": lambda: data.numpy().argmax(axis=-1),
            "gpu_argmax_copy_ids": lambda: vm["argmax_logits"](data).numpy(),
        }.items():
            device.sync()
            start = time.perf_counter()
            for _ in range(repeats):
                value = fn()
            device.sync()
            np.testing.assert_array_equal(value, expected)
            timings[name + "_ms"] = 1000 * (time.perf_counter() - start) / repeats
        row = dict(rows=rows, vocab=vocab, **timings)
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="sm_72")
    parser.add_argument("--output")
    args = parser.parse_args()
    results = run(args.arch)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
