# Xavier prefill weight expansion and dense GEMM — November 2, 2026

Vectorized weight expansion improves Nemotron prefill by 11–28% in matched
short tests and is deployed. This follows the
[layer-aware attention work](xavier_prefill_followup.md), retaining Nemotron
3.5 Bolt E4B's q4f16_1 checkpoint, 16K context, 512-token prefill chunks,
`hybrid-width` attention and two-proposal DFlash setup.

## Matched model results

AGX Xavier 32GB, MAXN, batch one, prefix cache disabled. The order is old
library, vector expansion, old library, with one warmup and two measured
requests per prompt in each arm. Each request forces 16 output tokens.
The control pools the four measurements bracketing the candidate.

| Prompt tokens | Current fork prefill (tok/s) | Vector expansion (tok/s) | Change |
| --- | ---: | ---: | ---: |
| 161 | 255.10 | 327.11 | +28.23% |
| 448 | 451.76 | 520.17 | +15.14% |
| 910 | 374.64 | 416.33 | +11.13% |

All six output strings per prompt match exactly. The controls before/after
are close: 254.42/255.77, 452.68/450.85 and 375.53/373.74 tok/s, respectively.
This is a short repeatability check, not a sustained-load or full-context
qualification. Sixteen-token decode samples are regression checks, not a
new decode-performance result. Forced continuations may extend past EOS.

The control already includes the November 2 local-attention width change.
These rates should not replace the older 8K-context/128-output-token README
benchmarks or be treated as a matched comparison with another engine/device.

## Weight expansion

`MLC_XAVIER_DEQUANT=1` enables a native CUDA expansion kernel when compiling
Gemma4 prefill entrypoints. Each thread unpacks one 32-bit word, multiplies
four FP16 pairs by their group scale and writes eight weights with one aligned
16-byte store. It preserves the NK layout, 32-weight groups, nibble offset
of seven and FP16 rounding. Decode/verification keep their existing native
small-batch path. Embedding lookup and the output head remain unchanged.

The replacement feeds the same dense cuBLAS operation. It does not retain
an expanded copy of every layer's weights between requests. The opt-in is
restricted to sm_72 and q4f16_1 with K divisible by 32; other compilation modes retain their existing
path.

An isolated comparison against TVM's q4f16_1 expansion with the fallback GPU
schedule measured the following times. Each timing has two warmup calls and
five measured calls; these are kernel screening measurements, not model rates.

| Weight matrix N × K | TVM expansion (ms) | Vector expansion (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 20,480 × 2,560 | 3.312 | 1.224 | 2.71× |
| 2,560 × 10,240 | 1.496 | 0.620 | 2.41× |
| 2,048 × 2,560 | 0.356 | 0.141 | 2.53× |
| 10,752 × 2,560 | 1.728 | 0.643 | 2.69× |

Every expanded element agrees exactly with TVM. An independent NumPy
unpacking reference checks tail rows, including an odd 137-row fixture.
The exported Relax/FFI test checks bit patterns, including signed zero,
subnormal scales and negative scales.

## Dense GEMM probe

`scripts/probe_xavier_linear.py --gemm` compares cuBLASLt's returned
heuristics and legacy `cublasGemmEx` on the same FP16 matrices. The workspace
limit is 32 MiB and the requested compute type is FP32, matching the current
TVM call. It records numerical differences as well as latency. FP32 compute
type alone does not exclude algorithms that store split-K partial sums in
FP16; the follow-up probe records algorithm ID and reduction configuration.

The initial 448-row screen finds the existing choice competitive for the
largest gate/up projection. Repeated tests find approximately 40% lower kernel
latency for the local query projection at 161/448 rows, and 8–15% lower latency
for the down projection at 448/512 rows. The 512-row query projection already
uses the preferred algorithm. These comparisons match the full output against
the first heuristic at `rtol=0.003, atol=0.002`; sampled entries also have an
independent FP32 reference. Some choices exceed the baseline-matching tolerance.
Several default algorithms also use FP16 partial sums, so differences alone
do not establish which choice gives better model quality.

A separate old/tuned/old model experiment uses an opt-in `LD_PRELOAD` hook
to select those configurations by algorithm properties, with the vector
dequantization library in all three arms:

| Prompt tokens | Default GEMM (tok/s) | Tuned GEMM (tok/s) | Change |
| --- | ---: | ---: | ---: |
| 161 | 328.99 | 331.25 | +0.69% |
| 448 | 518.30 | 534.54 | +3.13% |
| 910 | 417.84 | 420.37 | +0.61% |

All six output strings per prompt match. The sub-1% differences are too small
for a firm performance conclusion from these samples. The hook selects only
the tested row counts and projection shapes; other calls retain the default
heuristic. It is a diagnostic experiment, not a general selection policy,
and **is not enabled in serving**. The deployed change retains existing
cuBLAS selection. General row-count coverage and broader numerical/model
qualification remain follow-up work.

Machine-readable timings, algorithm configurations and artifact hashes:
[prefill-linear-summary.json](../benchmarks/xavier/prefill-linear-summary.json).
Private raw logs are archived under `mlc-jp5/logs/prefill-linear-20261102/`
in the benchmark workspace, with matching board-side logs.

## Reproduce

Stop GPU serving before compiling or benchmarking. From the initialized
Xavier checkout:

```sh
source scripts/xavier/env.sh
python scripts/probe_xavier_linear.py --gemm --output logs/linear-probe.json
python tests/python/op/test_xavier_dequant.py
bash scripts/build_xavier_prefill_linear.sh
```

The build writes `dist/libs/target-dflash-vector-dequant-16k-sm72.so` without
overwriting `target-dflash-dp4a-attn-16k-sm72.so`. It requires the existing
`dist/target-16k` model configuration. `MLC_XAVIER_DEQUANT` is a compile-time
switch; setting it in a running service does not change an existing library.

The matched model command is:

```sh
python scripts/bench_xavier_dflash.py --prefill-variant hybrid-width \
  --target dist/target-16k --target-lib dist/libs/target-dflash-vector-dequant-16k-sm72.so \
  --draft dist/draft-16k --draft-lib dist/libs/draft-cached-tc-16k-sm72.so \
  --context-size 16384 --draft-tokens 2 --tokens 16 \
  --cases chat,448,910 --warmups 1 --repeats 2 --output logs/vector-prefill.json
```

Use the original target library for the bracketing controls. To reproduce
the optional dense experiment, first compile the diagnostic hook:

```sh
g++ -O2 -shared -fPIC -I/usr/local/cuda/include scripts/xavier/gemm_probe_hook.cc \
  -L/usr/local/cuda/lib64 -lcublasLt -lcudart -ldl \
  -o dist/linear-experiments/gemm-probe-hook.so
```

Run the same benchmark with `LD_PRELOAD="$PWD/dist/linear-experiments/gemm-probe-hook.so"`
and `MLC_XAVIER_GEMM_PROBE=1` in that command's environment; use zero for the
bracketing controls. The log confirms each selected shape. This hook checks
CUDA 11.4, device zero/sm_72, FP16 inputs/output, FP32 compute, transpose flags,
batch one and default epilogue, and falls back if the expected candidate is absent.
It does not change the normal runtime installation.

## Deployment and rollback

After validating the new library, install the optional model-library drop-in
alongside `prefill-width.conf`:

```sh
sudo install -m 644 deploy/xavier/prefill-linear.conf \
  /etc/systemd/system/mlc-nemotron.service.d/prefill-linear.conf
sudo systemctl daemon-reload
sudo systemctl restart mlc-nemotron
```

`MLC_NEMOTRON_MODEL_LIB` selects the target library. Removing only
`prefill-linear.conf`, then reloading/restarting, restores the original
target library while retaining `hybrid-width` attention. The old library
is preserved. Restore serving after standalone tests.

The deployed service passes arithmetic (`391`), a declared `get_weather`
tool call and a 768-dimensional embedding request. Nemotron, EmbeddingGemma 2
and d1 remain registered and resident. No dense-tuning preload is deployed.
