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
p.add_argument("--layer-width", action="store_true", help="Check layer-aware ragged widths")
p.add_argument("--layout", choices=["e4b", "unified"], default="e4b")
a = p.parse_args()
# Query and stored K/V heads. Of the Unified 12B, only the current chunk
# (--layer-width) takes these kernels; its cached pages take test_xavier_paged.py.
HQ, KV = (8, 2) if a.layout == "e4b" else (16, 8)
G = HQ // KV
assert a.layer_width or a.layout == "e4b"
fns = {"ragged": _attention_prefill_ragged(KV, HQ, 512, 512, "float16", {}, TARGET)}
for local in (0, 1):
    fns[f"paged{local}"] = _attention_prefill(
        KV, HQ, 512, "float16", bool(local), {}, TARGET, sliding_window_size=512
    )
ref = tvm.compile(
    tvm_tirx.IRModule({k: v.with_attr("global_symbol", k) for k, v in fns.items()}), target=TARGET
).jit()
folder = ROOT / "dist/prefill-experiments"
native = tvm_ffi.load_module(str(folder / "native.so"))
modules = {"wmma": native}
if not a.layer_width:
    modules.update({v: tvm_ffi.load_module(str(folder / f"{v}.so")) for v in ("bounds", "width")})
rng = np.random.default_rng(9331)
dev = tvm_tirx.cuda(0)
results = []
wide_fixture = None


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
if a.layer_width:
    cases = [case for case in cases if case[0] == "ragged"]
    cases += [("ragged256", *case[1:]) for case in list(cases)]
for name, rows_list, contexts, offset, sink, causal, scale in cases:
    narrow_ragged = name == "ragged256"
    if narrow_ragged:
        name = "ragged"
    local = name == "paged1"
    paged = name != "ragged"
    width = 256 if local or narrow_ragged else 512
    batch = len(rows_list)
    rows = sum(rows_list)
    qptr = np.array([0, *np.cumsum(rows_list)], "int32")
    q = np.zeros((rows, HQ, 512), "float16")
    q[:, :, :width] = rng.normal(0, input_std, (rows, HQ, width)).astype("float16")
    kvs = [rng.normal(0, input_std, (2, c, KV, width)).astype("float16") for c in contexts]
    qo = tvm_tirx.runtime.empty(q.shape, "float16", dev)
    lseo = tvm_tirx.runtime.empty((rows, HQ), "float32", dev)
    if paged:
        actual_offset = offset if local else 0
        actual_sink = sink if local else 0
        spans = [c + actual_offset - actual_sink if c else 0 for c in contexts]
        npages = [(s + 15) // 16 for s in spans]
        pptr = np.array([0, *np.cumsum(npages)], "int32")
        pv = rng.permutation(sum(npages)).astype("int32")
        pages = np.zeros((max(sum(npages), 1), 2, KV, 16, 512), "float16")
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
        kv = np.zeros((2, sum(contexts), KV, 512), "float16")
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
            expected = np.zeros((HQ, 512), "float32")
            expected_lse = np.full(HQ, -50000.0, "float32")
            if last > first:
                for h in range(HQ):
                    logits = (
                        kvs[seq][0, first:last, h // G].astype("float32")
                        @ q[qptr[seq] + r, h, :width].astype("float32")
                        * scale
                    )
                    top = logits.max()
                    prob = np.exp(logits - top)
                    expected[h, :width] = (prob / prob.sum()) @ kvs[seq][
                        1, first:last, h // G
                    ].astype("float32")
                    expected_lse[h] = (top + np.log(prob.sum())) / np.log(2)
            samples.append((qptr[seq] + r, expected, expected_lse))
    ref[name](*inputs)
    baseline = qo.numpy()
    baseline_lse = lseo.numpy()
    if a.layer_width and not narrow_ragged and rows > 4 and all(contexts):
        wide_fixture = (inputs, baseline, baseline_lse)
    record = dict(
        kind=name,
        queries=rows_list,
        context=contexts,
        offset=offset,
        sink=sink,
        causal=causal,
        scale=scale,
        logical_width=width,
        baseline_ms=timing(ref[name], inputs),
        variants={},
    )
    candidates = {"hybrid-width": native} if a.layer_width else modules
    for variant, module in candidates.items():
        fn = native["wrap_prefill"](
            ref[name],
            module["ragged_by_layer"]("lg") if a.layer_width else module[name],
            int(variant == "width" and local),
            8 if paged else 7,
        )
        if a.layer_width:
            unscoped = fn
            # Reproduce the layer-id entrypoint that invokes the ragged callback.
            scoped = native["wrap_layer"](tvm_ffi.convert(lambda _cache, _layer: unscoped(*inputs)))

            def fn(*unused):
                return scoped(0, 0 if narrow_ragged else 1)

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

if a.layer_width:
    # Scope state must not leak into a later global call, including exceptions.
    inputs, expected, expected_lse = wide_fixture

    def fail(_cache, _layer):
        raise RuntimeError("deliberate scope unwind")

    wrapped = native["wrap_layer"](tvm_ffi.convert(fail))
    try:
        wrapped(0, 0)
    except Exception as error:
        assert "deliberate scope unwind" in str(error)
    else:
        raise AssertionError("Expected scope test exception")
    ragged = native["ragged_by_layer"]("lg")
    ragged(*inputs)
    np.testing.assert_allclose(inputs[7].numpy(), expected, rtol=0.006, atol=output_atol)
    np.testing.assert_allclose(inputs[8].numpy(), expected_lse, rtol=2e-5, atol=0.0004)
    unknown = native["wrap_layer"](tvm_ffi.convert(lambda _cache, _layer: ragged(*inputs)))
    unknown(0, 100)
    np.testing.assert_allclose(inputs[7].numpy(), expected, rtol=0.006, atol=output_atol)
    print("Layer scope unwind and unknown-layer fallback passed", flush=True)
