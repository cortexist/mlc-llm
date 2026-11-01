"""Real-weight draft-kernel comparison; synthetic features, not an LLM throughput test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from tvm import relax
from tvm.contrib import tvmjs

import mlc_llm  # Registers the runtime dependencies of compiled MLC libraries.

parser = argparse.ArgumentParser()
parser.add_argument("--contexts", default="161,448,1768")
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--variant-suffix", default="-fp32res")
parser.add_argument("--cached-lib", help="Override the cached variant for numerical/timing checks")
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dev = tvm_tirx.cuda(0)
tensor = lambda array: tvm_tirx.runtime.tensor(np.ascontiguousarray(array), device=dev)
weights, _ = tvmjs.load_tensor_cache(str(root / "dist/draft"), dev)
variants = {}
for name in ["uncached", "cached"]:
    library = (
        root / args.cached_lib
        if name == "cached" and args.cached_lib
        else root / f"dist/libs/draft-{name}{args.variant_suffix}-sm72.so"
    )
    vm = relax.VirtualMachine(tvm_ffi.load_module(str(library)), dev)
    metadata = json.loads(vm["_metadata"]())
    variants[name] = (vm, [weights[p["name"]] for p in metadata["params"]])
rng = np.random.default_rng(29)
noise = tensor(rng.normal(0, 1, (1, 3, 2560)).astype("float16"))
results = []


def timing(fn):
    fn()
    dev.sync()
    start = time.perf_counter()
    for _ in range(args.repeats):
        fn()
    dev.sync()
    return (time.perf_counter() - start) * 1000 / args.repeats


for context in map(int, args.contexts.split(",")):
    features = rng.normal(0, 5, (1, context, 12800)).astype("float16")
    positions = np.arange(context + 3, dtype="float64")[:, None]
    angles = positions / (1e6 ** (np.arange(64) / 64))
    angles = np.concatenate([angles, angles], axis=-1)[None, :, None, :]
    outputs = {}
    row = {"context": context, "draft_tokens": 2}
    for name, (vm, params) in variants.items():
        # Respect the model's 512-token projection compilation bound.
        pieces = []
        for start in range(0, context, 512):
            piece = vm["project_target_hidden"](tensor(features[:, start : start + 512]), params)
            pieces.append(piece.numpy())
        projected = np.concatenate(pieces, axis=1)
        assert np.isfinite(projected).all(), name
        offset = max(0, context - 1023) if name == "cached" else 0
        projection = tensor(projected[:, offset:])
        cos = tensor(np.cos(angles[:, offset:]).astype("float32"))
        sin = tensor(np.sin(angles[:, offset:]).astype("float32"))
        mask = tensor(np.zeros((1, 1, 3, context + 3 - offset), "float32"))

        def forward():
            return vm["draft_forward"](noise, projection, cos, sin, mask, params)

        outputs[name] = forward().numpy()
        assert np.isfinite(outputs[name]).all(), name
        row[name + "_draft_ms"] = timing(forward)
        new_features = tensor(features[:, :3])
        row[name + "_project_3_tokens_ms"] = timing(
            lambda: vm["project_target_hidden"](new_features, params)
        )
    difference = outputs["cached"].astype("float32") - outputs["uncached"].astype("float32")
    row["max_abs_error"] = float(np.max(np.abs(difference)))
    row["relative_rmse"] = float(
        np.sqrt(np.mean(difference**2))
        / np.sqrt(np.mean(outputs["uncached"].astype("float32") ** 2))
    )
    assert row["relative_rmse"] < 0.02, row
    reference_file = root / "logs/draft-reference.npz"
    if context == 161 and reference_file.exists():
        reference = np.load(reference_file)
        # Compare only when this is the exact independent-reference fixture.
        if np.array_equal(features, reference["features"]):
            ref_output = reference["output"]
            row["torch_reference_relative_rmse"] = float(
                np.sqrt(np.mean((outputs["cached"].astype("float32") - ref_output) ** 2))
                / np.sqrt(np.mean(ref_output**2))
            )
            assert row["torch_reference_relative_rmse"] < 0.05, row
    results.append(row)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
