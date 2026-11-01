"""Compare prefill candidates with TVM and independently sampled FP32 attention."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm
import tvm_ffi
import tvm_tirx
from tvm.relax.frontend.nn.llm.kv_cache import _attention_prefill, _attention_prefill_ragged

from mlc_llm.op.xavier_prefill import ROOT, TARGET

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--stress", action="store_true", help="Unit-scale Q/K/V and small-batch fallback")
a = p.parse_args()
fns = {"ragged": _attention_prefill_ragged(2, 8, 512, 512, "float16", {}, TARGET)}
for local in (0, 1):
    fns[f"paged{local}"] = _attention_prefill(
        2, 8, 512, "float16", bool(local), {}, TARGET, sliding_window_size=512
    )
ref = tvm.compile(
    tvm_tirx.IRModule({k: v.with_attr("global_symbol", k) for k, v in fns.items()}), target=TARGET
).jit()
folder = ROOT / "dist/prefill-experiments"
native = tvm_ffi.load_module(str(folder / "native.so"))
modules = {v: tvm_ffi.load_module(str(folder / f"{v}.so")) for v in ("bounds", "width")}
modules["wmma"] = native
rng = np.random.default_rng(9331)
dev = tvm_tirx.cuda(0)
results = []


def tensor(x):
    return tvm_tirx.runtime.tensor(x, device=dev)


def timing(fn, inputs):
    for _ in range(2):
        fn(*inputs)
    dev.sync()
    start = time.perf_counter()
    for _ in range(5):
        fn(*inputs)
    dev.sync()
    return (time.perf_counter() - start) * 200


cases = []
for name in ("ragged", "paged0", "paged1"):
    for rows, context in ((17, 32), (161, 161), (448, 512), (512, 512), (232, 1536)):
        cases.append((name, [rows], [context], 0, 0, 1, 1.0))
    cases.extend(
        [
            (name, [33], [448], 7, 4, 0, 0.125),
            (name, [17, 33], [32, 97], 7, 4, 1, 1.0),
            (name, [17], [0], 0, 0, 1, 1.0),
        ]
    )
if a.stress:
    cases = [
        (name, [rows], [context], 0, 0, 1, 1.0)
        for name in ("ragged", "paged0", "paged1")
        for rows, context in ((3, 32), (17, 32), (161, 161), (512, 512))
    ]
input_std = 1.0 if a.stress else 0.2
output_atol = 0.002 if a.stress else 0.00025
for name, rows_list, contexts, offset, sink, causal, scale in cases:
    local = name == "paged1"
    paged = name != "ragged"
    width = 256 if local else 512
    batch = len(rows_list)
    rows = sum(rows_list)
    qptr = np.array([0, *np.cumsum(rows_list)], "int32")
    q = np.zeros((rows, 8, 512), "float16")
    q[:, :, :width] = rng.normal(0, input_std, (rows, 8, width)).astype("float16")
    kvs = [rng.normal(0, input_std, (2, c, 2, width)).astype("float16") for c in contexts]
    qo = tvm_tirx.runtime.empty(q.shape, "float16", dev)
    lseo = tvm_tirx.runtime.empty((rows, 8), "float32", dev)
    if paged:
        actual_offset = offset if local else 0
        actual_sink = sink if local else 0
        spans = [c + actual_offset - actual_sink if c else 0 for c in contexts]
        npages = [(s + 15) // 16 for s in spans]
        pptr = np.array([0, *np.cumsum(npages)], "int32")
        pv = rng.permutation(sum(npages)).astype("int32")
        pages = np.zeros((max(sum(npages), 1), 2, 2, 16, 512), "float16")
        for seq, c in enumerate(contexts):
            for j in range(c):
                slot = j if j < actual_sink else j - actual_sink + actual_offset
                pages[pv[pptr[seq] + slot // 16], :, :, slot % 16, :width] = kvs[seq][:, j]
        lengths = np.array([s - (n - 1) * 16 if n else 0 for s, n in zip(spans, npages)], "int32")
        if local:
            lengths = np.stack(
                (
                    lengths,
                    np.full(batch, actual_offset, "int32"),
                    np.full(batch, actual_sink, "int32"),
                )
            )
        inputs = [
            tensor(q),
            tensor(qptr),
            tensor(pages),
            tensor(pptr),
            tensor(pv),
            tensor(lengths),
            tensor(np.zeros(batch, "int32")),
            tensor(np.arange(rows, dtype="int32")),
            qo,
            lseo,
            causal,
            0,
            1.0,
            10000.0,
            scale,
        ]
    else:
        kv = np.zeros((2, sum(contexts), 2, 512), "float16")
        pos = 0
        for x in kvs:
            kv[:, pos : pos + x.shape[1], :, :width] = x
            pos += x.shape[1]
        kptr = np.array([0, *np.cumsum(contexts)], "int32")
        inputs = [
            tensor(q),
            tensor(qptr),
            tensor(kv[0]),
            tensor(kv[1]),
            tensor(kptr),
            tensor(np.arange(rows, dtype="int32")),
            tensor(np.zeros(batch, "int32")),
            qo,
            lseo,
            causal,
            0,
            1.0,
            10000.0,
            scale,
        ]
    valid_rows = np.zeros(rows, bool)
    samples = []
    for seq, (nr, c) in enumerate(zip(rows_list, contexts)):
        for r in range(nr):
            start = max(0, c - max(512 - r - 1, 0)) if local and causal else 0
            end = min(c, c - nr + r + 1) if not local and causal else c
            valid_rows[qptr[seq] + r] = end > start
        for r in sorted(set([0, min(1, nr - 1), min(15, nr - 1), min(16, nr - 1), nr - 2, nr - 1])):
            first = max(0, c - max(512 - r - 1, 0)) if local and causal else 0
            last = min(c, c - nr + r + 1) if not local and causal else c
            expected = np.zeros((8, 512), "float32")
            expected_lse = np.full(8, -50000.0, "float32")
            if last > first:
                for h in range(8):
                    logits = (
                        kvs[seq][0, first:last, h // 4].astype("float32")
                        @ q[qptr[seq] + r, h, :width].astype("float32")
                        * scale
                    )
                    top = logits.max()
                    prob = np.exp(logits - top)
                    expected[h, :width] = (prob / prob.sum()) @ kvs[seq][
                        1, first:last, h // 4
                    ].astype("float32")
                    expected_lse[h] = (top + np.log(prob.sum())) / np.log(2)
            samples.append((qptr[seq] + r, expected, expected_lse))
    ref[name](*inputs)
    baseline = qo.numpy()
    baseline_lse = lseo.numpy()
    record = dict(
        kind=name,
        queries=rows_list,
        context=contexts,
        offset=offset,
        sink=sink,
        causal=causal,
        scale=scale,
        baseline_ms=timing(ref[name], inputs),
        variants={},
    )
    for variant, module in modules.items():
        fn = native["wrap_prefill"](
            ref[name], module[name], int(variant == "width" and local), 8 if paged else 7
        )
        fn(*inputs)
        output = qo.numpy()
        lse = lseo.numpy()
        # The original finite mask sentinel gives spurious values for all-masked
        # rows; independently require zero and the empty-state LSE for these.
        np.testing.assert_allclose(
            output[valid_rows],
            baseline[valid_rows],
            rtol=0.006,
            atol=output_atol,
            err_msg=f"{variant} {name} {rows_list} {contexts}",
        )
        np.testing.assert_allclose(
            lse[valid_rows], baseline_lse[valid_rows], rtol=2e-5, atol=0.0004
        )
        np.testing.assert_array_equal(output[~valid_rows], 0)
        np.testing.assert_allclose(lse[~valid_rows], -50000.0, rtol=0, atol=0)
        for row, expected, expected_lse in samples:
            np.testing.assert_allclose(output[row], expected, rtol=0.006, atol=output_atol)
            np.testing.assert_allclose(lse[row], expected_lse, rtol=2e-5, atol=0.0004)
        if rows <= 4:
            np.testing.assert_array_equal(output, baseline)
            np.testing.assert_array_equal(lse, baseline_lse)
        record["variants"][variant] = dict(
            ms=timing(fn, inputs),
            max_abs_error=float(
                np.max(np.abs(output[valid_rows].astype("float32") - baseline[valid_rows]))
            )
            if valid_rows.any()
            else 0,
        )
    results.append(record)
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(record), flush=True)
