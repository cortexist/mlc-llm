"""Check the tensor-core paged attention against FP32 attention and time it against TVM.

One sequence in the cache layout of the E4B, of the Unified 12B (both padded to
512) or of a Qwen3.8 model (256 wide, global layers only), through both the
prefill and the decode callback. Every slot that no query may attend holds NaN:
unused keys of the last page, pages of other sequences and the padding of local
heads.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm
import tvm_ffi
import tvm_tirx
from tvm.relax.frontend.nn.llm.kv_cache import _attention_decode, _attention_prefill

from mlc_llm.op.xavier_prefill import ROOT, TARGET

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
p.add_argument("--stress", action="store_true", help="Unit-scale Q/K/V")
# Query heads, stored K/V heads, local window (0: no local layers) and stored width
# of a head. The Unified 12B stores the one K/V head of a global layer in every
# cache head.
LAYOUTS = {
    "e4b": (8, 2, 512, 512),
    "unified": (16, 8, 1024, 512),
    "qwen27b": (24, 4, 0, 256),
    "qwen2b": (8, 2, 0, 256),
}
p.add_argument("--layout", choices=list(LAYOUTS), default="e4b")
a = p.parse_args()
HQ, KV, W, STRIDE = LAYOUTS[a.layout]
G = HQ // KV
fns = {}
for local in (0, 1) if W else (0,):
    fns[f"prefill{local}"] = _attention_prefill(
        KV, HQ, STRIDE, "float16", bool(local), {}, TARGET, sliding_window_size=W
    )
    fns[f"decode{local}"] = _attention_decode(KV, HQ, STRIDE, "float16", bool(local), {}, TARGET)
ref = tvm.compile(
    tvm_tirx.IRModule({k: v.with_attr("global_symbol", k) for k, v in fns.items()}), target=TARGET
).jit()
native = tvm_ffi.load_module(str(ROOT / "dist/prefill-experiments/native.so"))
rng = np.random.default_rng(1111)
dev = tvm_tirx.cuda(0)
input_std = 1.0 if a.stress else 0.2
output_atol = 0.002 if a.stress else 0.00025


def tensor(x):
    return tvm_tirx.runtime.tensor(x, device=dev)


def refuse(*_):
    raise AssertionError("The paged kernel declined a call it should take")


def timing(fn, inputs):
    for _ in range(2):
        fn(*inputs)
    dev.sync()
    start = time.perf_counter()
    for _ in range(5):
        fn(*inputs)
    dev.sync()
    return round((time.perf_counter() - start) * 200, 3)


# (local, decode, rows, cached keys, window offset, sink, causal, scale)
cases = [(0, 0, r, c, 0, 0, 0, 1.0) for r, c in ((1, 1), (1, 17), (2, 16), (3, 100), (4, 513))]
cases += [(0, 0, r, c, 0, 0, 0, 1.0) for r, c in ((5, 31), (8, 700), (9, 1536), (17, 32))]
cases += [(0, 0, r, c, 0, 0, 0, 1.0) for r, c in ((161, 161), (232, 1536), (512, 512))]
cases += [(0, 0, r, c, 0, 0, 0, 1.0) for r, c in ((512, 4000), (3, 8256), (512, 8256))]
# Rows around what a block of six query heads holds (two, or five of a chunk) and
# around where a chunk of such blocks stops splitting the keys.
cases += [(0, 0, r, c, 0, 0, 0, 0.0625) for r, c in ((4, 6600), (6, 200), (10, 97), (11, 2000))]
cases += [(0, 0, r, c, 0, 0, 0, 0.0625) for r, c in ((80, 300), (81, 300), (134, 6574))]
cases += [(0, 0, r, c, 0, 0, 1, 1.0) for r, c in ((5, 100), (17, 32), (161, 161))]
cases += [(1, 0, r, c, o, 0, 1, 1.0) for r, c, o in ((1, 1, 0), (3, 100, 0), (4, 512, 5))]
cases += [(1, 0, r, c, o, 0, 1, 1.0) for r, c, o in ((8, 512, 9), (17, 32, 0), (161, 161, 0))]
cases += [(1, 0, r, c, o, 0, 1, 1.0) for r, c, o in ((448, 512, 0), (512, 512, 3), (512, 300, 0))]
cases += [(1, 0, 33, 448, 7, 4, 0, 0.125), (1, 0, 3, 512, 11, 0, 0, 1.0)]
cases += [(0, 1, 1, c, 0, 0, 0, 1.0) for c in (1, 16, 17, 513, 2000, 8256, 16000)]
cases += [(1, 1, 1, c, o, 0, 0, 1.0) for c, o in ((1, 0), (300, 0), (512, 5), (512, 0))]
if W == 1024:
    cases += [(1, 0, r, c, o, 0, 1, 1.0) for r, c, o in ((3, 1024, 13), (512, 1024, 3))]
    cases += [(1, 0, r, c, o, 0, 1, 1.0) for r, c, o in ((448, 700, 0), (17, 1024, 0))]
    cases += [(1, 1, 1, c, o, 0, 0, 1.0) for c, o in ((1024, 5), (1024, 0), (700, 0))]
if not W:
    cases = [case for case in cases if not case[0]]
results = []
for local, decode, rows, count, offset, sink, causal, scale in cases:
    width = 256 if local else STRIDE
    q = np.zeros((rows, HQ, STRIDE), "float16")
    q[:, :, :width] = rng.normal(0, input_std, (rows, HQ, width)).astype("float16")
    repeated = int(a.layout == "unified" and not local)
    kv = rng.normal(0, input_std, (2, count, 1 if repeated else KV, width)).astype("float16")
    kv = np.ascontiguousarray(np.broadcast_to(kv, (2, count, KV, width)))
    span = count + offset - sink
    npages = (span + 15) // 16
    pv = rng.permutation(npages + 3)[:npages].astype("int32")
    pages = np.full((npages + 3, 2, KV, 16, STRIDE), np.nan, "float16")
    # Expired keys in front of a local window are real, finite data.
    expired = rng.normal(0, input_std, (2, KV, offset, width)).astype("float16")
    for slot in range(offset):
        pages[pv[slot // 16], :, :, slot % 16, :width] = expired[:, :, slot]
    for j in range(count):
        slot = j if j < sink else j - sink + offset
        pages[pv[slot // 16], :, :, slot % 16, :width] = kv[:, j]
    length = np.array([span - (npages - 1) * 16], "int32")
    if local:
        length = np.array([[length[0]], [offset], [sink]], "int32")
    qo = tvm_tirx.runtime.empty(q.shape, "float16", dev)
    lseo = tvm_tirx.runtime.empty((rows, HQ), "float32", dev)
    inputs = [tensor(q)] + ([] if decode else [tensor(np.array([0, rows], "int32"))])
    inputs += [tensor(pages), tensor(np.array([0, npages], "int32")), tensor(pv), tensor(length)]
    inputs += [tensor(np.zeros(1, "int32")), tensor(np.arange(rows, dtype="int32")), qo, lseo]
    inputs += ([] if decode else [causal]) + [0, 1.0, 10000.0, scale]

    row = np.arange(rows)[:, None]
    key = np.arange(count)[None]
    seen = np.ones((rows, count), bool)
    if causal and local:
        seen = key >= np.maximum(0, count - np.maximum(W - row - 1, 0))
    elif causal:
        seen = key < count - rows + row + 1
    expected = np.zeros((rows, HQ, STRIDE), "float32")
    expected_lse = np.full((rows, HQ), -50000.0, "float32")
    mask = np.repeat(seen, G, axis=0)
    some = mask.any(1)
    for head in range(KV):
        queries = q[:, head * G : head * G + G, :width].astype("float32").reshape(rows * G, width)
        logits = queries @ kv[0, :, head].astype("float32").T * np.float32(scale)
        logits[~mask] = -np.inf
        top = np.where(some, logits.max(1, initial=-np.inf), 0)
        prob = np.exp(logits - top[:, None])
        total = np.where(some, prob.sum(1), 1)
        out = (prob / total[:, None]) @ kv[1, :, head].astype("float32")
        expected[:, head * G : head * G + G, :width] = out.reshape(rows, G, width)
        lse = np.where(some, (top + np.log(total)) / np.log(2), -50000.0)
        expected_lse[:, head * G : head * G + G] = lse.reshape(rows, G)

    fn = native["wrap_paged"](
        tvm_ffi.convert(refuse), decode, HQ, KV, HQ if repeated else G, W if local else 0, STRIDE
    )
    fn(*inputs)
    output, lse = qo.numpy(), lseo.numpy()
    fn(*inputs)
    name = dict(local=local, decode=decode, rows=rows, keys=count, offset=offset, sink=sink)
    np.testing.assert_array_equal(output.view("uint16"), qo.numpy().view("uint16"), str(name))
    np.testing.assert_allclose(output, expected, rtol=0.006, atol=output_atol, err_msg=str(name))
    np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=0.0004, err_msg=str(name))
    empty = ~seen.any(1)
    np.testing.assert_array_equal(output[empty], 0)
    np.testing.assert_array_equal(output[:, :, width:], 0)
    record = dict(
        **name,
        causal=causal,
        scale=scale,
        max_abs_error=float(np.abs(output - expected).max()),
        ms=timing(fn, inputs),
        tvm_ms=timing(ref[f"{'decode' if decode else 'prefill'}{local}"], inputs),
    )
    results.append(record)
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(record), flush=True)
