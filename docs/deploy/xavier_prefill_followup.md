# Xavier layer-aware prefill follow-up — November 2, 2026

The `hybrid-width` variant removes padded work from local current-chunk
attention. On the existing Nemotron 3.5 Bolt E4B service libraries, prefill
improves about 13% at 448 tokens and 12% at 910 tokens. It does not change
weights, KV storage, context capacity or the small-batch decode/verification
kernels. No target or draft model recompilation is required.

## Matched results

AGX Xavier 32GB, MAXN, GPU 1377 MHz, q4f16_1 target, F16 DFlash assistant,
two proposals, batch one, 16K context, 512-token chunks, prefix cache disabled.
Each arm has one warmup and two measured repetitions per prompt, with 16
forced output tokens. The order is `hybrid`, `hybrid-width`, `hybrid`.
Controls below pool the four measurements bracketing the candidate.

| Prompt tokens | Control prefill tok/s | Candidate prefill tok/s | Change |
| --- | ---: | ---: | ---: |
| 161 | 245.24 | 253.53 | +3.38% |
| 448 | 397.24 | 449.41 | +13.13% |
| 910 | 332.76 | 372.44 | +11.92% |

All six output strings per case match exactly across the three arms. The
short control varies from 242.40 to 248.09 tok/s, so its small gain is less
conclusive. The 448-token controls are 397.40 and 397.08; the 910-token
controls are 330.75 and 334.78. An earlier screening run measured 436.83 tok/s
at 448 tokens, also faster than its 395.67 tok/s control. These are small
samples, not latency percentiles or sustained-load qualification.

This protocol differs from the November 1 8K-context, 128-output-token
comparison. Do not substitute these numbers into that historical table or
compare them directly with the Orin Nano's different engine/model/protocol.
The short decode samples are checks, not a new decode-performance claim.

Machine-readable results: [prefill-layer-width-summary.json](../benchmarks/xavier/prefill-layer-width-summary.json).
Raw private evidence is archived under `mlc-jp5/logs/prefill-followup-20261102/`
in the benchmark workspace, with a matching directory in the experimental checkout.

## Implementation

Previously, the ragged/current-chunk callback received padded 512-wide
Q/K/V tensors without the layer identity. Local layers have only 256 live
dimensions, but the callback ran the full 512-wide WMMA kernel.

Native wrappers now carry layer identity through the synchronous host-side
attention entrypoint using a scoped, thread-local width. The layout comes
from the target configuration's physical KV-owning layers; shared-KV calls
use their source layer ID. The callback chooses a 256- or 512-wide kernel
before enqueueing GPU work. Scope state is restored on normal return and
exception, and unknown IDs fall back to width 512. There is no per-layer
Python callback or GPU-to-CPU copy.

Physical storage strides stay 512. Narrow output tails are zeroed. Causal
masking and the existing paged-attention path stay unchanged. Calls with at
most four query rows retain their original callback. The opt-in variant
checks the 8-query/2-KV-head E4B geometry, layer partition and chunk/window
limits; it is not a general optimization for other model architectures.

## Validation

- Sixteen normal-scale fixtures cover both widths, ragged batches, partial
  tiles, differing Q/K lengths, causal/noncausal masking, scale changes and
  empty context. Outputs and log-sum-exp are compared with TVM and sampled
  independent FP32 attention.
- Eight unit-scale fixtures include small-batch fallback and 512-row inputs.
  Maximum observed absolute output error is 0.001953125, within the existing
  0.002 stress tolerance. Small-batch fallback is bit-identical.
- Exception unwinding and unknown-layer fallback pass explicit checks.
- The matched engine runs cover one- and two-chunk prompts and identical
  16-token continuations. They do not establish full-context model quality.
- The deployed API passes arithmetic (`391`), a native `get_weather` tool
  call, and a 768-dimensional EmbeddingGemma request with the resident
  specialists still enabled.

## Build, enable and roll back

Stop inference while building or running these GPU checks. From the
experimental checkout with its existing target/draft libraries and `width.so`:

```sh
source scripts/xavier/env.sh
python scripts/build_xavier_prefill.py --native-only
python tests/python/op/test_xavier_prefill.py --layer-width --output logs/prefill-width-tests.json
python tests/python/op/test_xavier_prefill.py --layer-width --stress --output logs/prefill-width-stress.json
```

For a fresh installation without the earlier prefill artifacts, omit
`--native-only` to build the TIR variants too. Benchmark with
`--prefill-variant hybrid-width`; pass the matching target configuration and
model libraries as in the saved results. The service supplies its explicit
16K target configuration to the adapter.

```sh
sudo install -m 644 deploy/xavier/prefill-width.conf /etc/systemd/system/mlc-nemotron.service.d/prefill-width.conf
sudo systemctl daemon-reload
sudo systemctl restart mlc-nemotron
```

Rollback removes only that optional drop-in, then reloads/restarts the unit.
The base unit still selects `hybrid`, which the rebuilt native library also
supports. Restore serving after standalone experiments.

## Remaining costs

A fresh CUPTI capture of a 448-token prompt plus two output tokens attributes
about 43% of summed GPU kernel time to dense GEMM, 23% to dequantization,
20% to current-chunk attention and 14% to other kernels. Activity delivery
was delayed: the file labeled prefill is empty and the file labeled decode
contains the request's prefill work too. Treat it as a combined diagnostic,
not a phase-separated profile or an uninstrumented performance result.

Weight expansion and dense GEMM remain the next substantial targets.
The subsequent [linear-kernel experiment](xavier_prefill_linear.md) measures
and deploys vectorized weight expansion, then tests dense GEMM selection separately.
Dequantization caching, fused prefill weight expansion/GEMM, larger chunks,
global-paged attention changes and matched Orin/Xavier GEMM measurements
were not implemented in this follow-up.
