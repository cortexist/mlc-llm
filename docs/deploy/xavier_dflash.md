# DFlash on brain1: experimental port and optimization record

Measured on 2026-11-01. This branch ports
[Lev Stambler's DFlash implementation](https://github.com/Lev-Stambler/mlc-llm--DFlash/tree/b6c7bf7d269fd4127fd1a48c478b01d9dc8ff42b)
onto MLC `44fd06d0`, retaining the measured Xavier fixes in `7e4cd2a5`.
The imported pipeline is recorded separately in `1f8903e1`.

The port executes Nemotron 3.5 Bolt E4B with a q4f16_1 target and the existing
F16 GGUF draft converted without requantization. It is an isolated experiment,
not an installed service. The original MLC checkout/wheel and llama.cpp remain
available. Validation covers greedy, one-sequence text generation.

## Native Xavier kernels (100 W port follow-up)

The user moved the board to the power station's USB-C port labeled 100 W and
rebooted it. This records the physical connection, not a verified PD contract.
The new experiments retain MAXN/fixed clocks and serialize compilation and GPU
work under `run_monitored.py`. Evidence is saved locally in
`mlc-jp5/logs/dflash/pd100-20261101/`.

The new kernels make DFlash a decode speedup on this board. Fresh controlled
runs use one warmup and two measured requests per case, 128 forced output
tokens, context 8192, chunk 512, one sequence and prefix caching disabled.
All rows below were measured on the new power connection, without competing
compilation or inference. Values are mean decode tokens/second:

| MLC variant | 161-token chat | 448-token prompt | 1768-token prompt |
|---|---:|---:|---:|
| Previous plain | 20.60 | 17.80 | 15.68 |
| Native matrices, plain | 23.24 | 19.63 | 17.18 |
| Previous DFlash, two draft tokens | 14.24 | 13.81 | 12.69 |
| Native matrices, DFlash, attention disabled | 25.29 | 21.57 | 19.75 |
| Native matrices + attention, DFlash | **31.59** | **33.42** | **31.08** |

Chat verification falls from **118.24 to 45.99 ms/round**; drafting falls from
21.03 to 15.75 ms. Mean output per round slightly decreases, 1.984 to 1.954,
so increased acceptance does not explain the 2.22x throughput gain. The first
128 chat output tokens match the previous DFlash run. Native plain decoding
produces different wording, demonstrating that these arithmetic paths are not
bit-identical even with the same checkpoint. Attention on/off produces identical
output text in all three cases, although small numerical differences change
draft acceptance by about one round per request.

Prefill improves modestly: **218.33 / 271.95 / 144.14 tok/s** for native DFlash,
versus 202.88 / 263.01 / 141.05 previously. The corresponding new request wall
times are 4.76 / 5.45 / 16.36 seconds. The 1768-token prompt still spends about
12.3 seconds before its first token. Forced continuations after EOS are useful
for timing but are not representative natural answers or an agent-quality test.

A fresh llama.cpp+DFlash F16 control uses the same prompt text, 128 output
tokens, one warmup/two repeats, no prompt cache and context 8192. It uses
Q4_K_M rather than MLC q4f16_1 plus Q8 activations, and its tokenizer counts
168 / 454 / 1774 input tokens rather than 161 / 448 / 1768:

| Case | MLC prefill tok/s | llama prefill tok/s | MLC decode tok/s | llama decode tok/s | MLC wall seconds | llama wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 218.33 | 189.90 | 31.59 | 26.80 | 4.76 | 5.63 |
| Medium | 271.95 | 316.46 | 33.42 | 25.09 | 5.45 | 6.51 |
| Long | 144.14 | 341.88 | 31.08 | 24.37 | 16.36 | 10.43 |

MLC now wins decode in these cases and the short/medium complete requests.
llama.cpp still wins the long request because of prefill. This is a practical
engine comparison with different quantization/tokenization, not a controlled
same-numerics accuracy comparison. The old 24.5 tok/s llama figure used a full
natural answer; the fresh 128-token control is the appropriate comparison here.
The experimental MLC path still requires prefix caching disabled and is tested
at one concurrent sequence. The existing llama.cpp launcher remains available;
no persistent service was changed.

The new draft-length sweep gives 27.36 / 31.59 / 32.35 tok/s on chat for
one / two / three draft tokens. Three drafts fall to 27.40 / 29.69 tok/s on
the 448 / 1768 cases, below two drafts. Keep **two draft tokens** as the general
setting; three is only a small chat-specific win in this sample.

Natural-stop checks with native plain, two-token and three-token DFlash return `391` for
the arithmetic prompt and `Done` for the long prompt. Their complete design-note
answers contain the requested three summary points and two additional risks,
but differ in wording and length (186 / 199 / 216 generated tokens). These are
functional smoke checks; the old and new arithmetic are not accuracy-equivalent
by construction, and no broad quality/perplexity benchmark has been run.

Two opt-in compilation flags add CUDA 11.4/sm_72 kernels:

- `MLC_XAVIER_Q4=1` quantizes activations to signed Q8 in groups of 32 and uses
  DP4A against the existing q4f16_1 checkpoint layout. A warp processes one
  output row for one token, or eight output rows for two through four tokens,
  sharing weight loads across the token batch. Accumulation is FP32. This
  replaces small-batch linear layers and the shared vocabulary projection;
  large prefill matrix multiplications retain the existing cuBLAS path.
- `MLC_XAVIER_ATTN=1` wraps the cache's paged-prefill callbacks for batches of
  one through four queries. Each query/head gets its own block. Local heads
  compute their actual 256 dimensions inside the existing 512-wide cache;
  global heads compute all 512. Larger batches call the original TIR kernel.
  Set `MLC_XAVIER_ATTN_OFF=1` **before engine creation** to disable this wrapper
  at runtime, allowing an attention ablation with the same model library.

The original single-token cache-decode callback is not replaced. The attention
change primarily accelerates verification; it does not promise faster large
prefill or ordinary one-token attention. The matrix path also benefits plain
decoding and the draft's shared target vocabulary head.

The cache layout, checkpoint, and tokenizer are unchanged. The Q8 activation
path changes arithmetic and is **not bit-identical to the old target**. Random
component checks found approximately 0.5–0.6% relative RMS error against an
independent FP32 dot product of the original FP16-dequantized weights. This is
not a perplexity or agent-quality evaluation. The supported experiment is
greedy, one sequence, at most three draft tokens; the native linear entry point
rejects batches outside one through four rows.

Standalone matrix timings (including activation quantization and launch overhead):

| Matrix, output × input | One token | Three tokens |
|---|---:|---:|
| FFN gate/up, 20480 × 2560 | 0.287 ms | 0.339 ms |
| FFN down, 2560 × 10240 | 0.166 ms | 0.174 ms |
| Vocabulary, 262144 × 2560 | 3.647 ms | 4.208 ms |

These demonstrate much better amortization across verification tokens, but do
not measure DRAM traffic or establish end-to-end throughput by themselves.

The attention tests compare outputs and base-2 log-sum-exp against both TVM
and NumPy, covering local/global attention, causal/noncausal masks, shuffled
pages, nonzero cache offsets/sinks, empty context, and the larger-batch fallback.
All 18 cases passed with maximum observed output differences at most 3.1e-5. Local
three-query attention at 448 cached tokens measured 1.044 ms with TIR versus
0.200 ms with the native callback. These are isolated component timings.
The cache already bounds normal local-layer history; the 1768-token local
component fixture deliberately stresses the mask and does not represent
normal retained local-cache length.

`tests/python/op/test_xavier_q4.py` exercises the Relax/FFI boundary, scratch
planning, FP16/FP32 outputs, tail rows and zero inputs. The compiler uses a
local external-module attachment pass that copies TVM's immutable attribute
array before adding native modules; the pinned upstream pass incorrectly
calls `append` directly on that array. A separate cache test compiles, exports,
reloads and instantiates the wrapped cache through the actual Relax VM. The
full model build subsequently completed successfully.

Repeated token limits 1, 2, 3 and 17 pass with both two- and three-token
DFlash, including decode accounting equal to completion tokens minus one.
The final CUPTI diagnostic completes with zero dropped records. Both old and
new 32-token chat traces contain 17 rounds:

| GPU work per round | Previous | Native |
|---|---:|---:|
| Target quantized linear layers, excluding vocabulary | 74.38 ms | 28.13 ms |
| Target vocabulary projection | 14.47 ms | 4.28 ms |
| Draft vocabulary projection | 9.81 ms | 4.28 ms |
| Target attention, paged plus ragged | 18.22 ms | 4.77 ms |
| Activation quantization | — | 1.16 ms |
| All recorded GPU kernels | 133.11 ms | 59.59 ms |

The native linear row combines FFN and other projections because they share a
kernel symbol. Vocabulary softcap operations, previously fused into the old
projection kernel, become separate small kernels. Instrumented GPU sums are
not wall-clock verification latency or throughput. Sources are the two trace
CSVs and `profile-comparison.json` in the archived evidence.

All final monitored runs exit successfully. Across the full build, comparison,
validation and llama control, CPU temperature peaks at 59 C and GPU at 65 C
(the MLC comparison peaks at 62.5 C GPU). No OC IRQ increment, current alarm,
or monitor stop is observed. Available memory stays above 18,726 MiB. Peak
summed monitored rails are 40.38 W and the lowest GPU supply sample is 19.616 V.
These are slow on-board samples, not wall power or proof of the negotiated PD
limit. The board remains up, MAXN and nvfancontrol remain active, and all test
processes have stopped. The new port completes the workload successfully; it
does not establish what caused the earlier unclean shutdown.

Build with the existing prepared weights/runtime:

```bash
cd ~/Workspaces/mlc-dflash-work
bash scripts/build_xavier_kernels.sh
# Output: dist/libs/target-dflash-dp4a-attn-sm72.so
# Optional separate matrix-only build: bash scripts/build_xavier_kernels.sh --q4-only

source ~/Workspaces/mlc-jp5/env.sh
export PYTHONPATH="$PWD/python" MLC_LIBRARY_PATH="$PWD/build"
python scripts/bench_xavier_dflash.py --draft-tokens 2 \
  --target-lib dist/libs/target-dflash-dp4a-attn-sm72.so \
  --cases chat,448,1768 --tokens 128 --warmups 1 --repeats 2 \
  --output logs/native-draft2.json
MLC_XAVIER_ATTN_OFF=1 python scripts/bench_xavier_dflash.py --draft-tokens 2 \
  --target-lib dist/libs/target-dflash-dp4a-attn-sm72.so \
  --cases chat,448,1768 --tokens 128 --warmups 1 --repeats 2 \
  --output logs/matrix-only-draft2.json
# Fresh cross-engine control; starts and stops its own localhost server.
python scripts/bench_xavier_llama.py --output logs/llama-control.json
```

## Prior controlled results, before the native kernels

**The completed DFlash port is slower than optimized plain MLC on Xavier.**
After the board restarted, the runtime was rebuilt and the final target library
compiled successfully. Controlled comparisons ran sequentially with no competing
compilation or inference: MAXN, fixed clocks, prefix cache disabled, context 8192,
prefill chunk 512, one warmup and two measured requests per case, forcing 128
output tokens. The two paths use identical weights, tokenizer and target library.

| Prompt tokens | Plain prefill tok/s | DFlash prefill tok/s | Plain decode tok/s | DFlash decode tok/s |
|---|---:|---:|---:|---:|
| 161 (design note) | 204.92 | 203.18 | **20.66** | 14.15 |
| 448 | 266.72 | 263.60 | **17.83** | 13.81 |
| 1768 | 141.98 | 141.44 | **15.66** | 12.70 |

DFlash uses two draft tokens plus the known frontier. One draft token is worse
on the design note: **13.75 tok/s**. With two draft tokens, the design note commits
1.984 output tokens per round (first draft accepted 60.94%, both 37.50%), taking
about 21.2 ms in the draft phase and **118.9 ms in verification**. Verification
includes target execution, selection, KV commit and the next feature update.
At the same acceptance and draft cost, it would need to fall below roughly
75 ms to match plain MLC. Further draft-only improvements cannot close this gap.

Both repeated chat and 448-token output strings match plain MLC exactly. The
1768-token forced continuation differs after its natural `Done` answer and end
marker. Different batch arithmetic can change greedy decisions; this comparison
does not establish identical token sequences for arbitrary inputs. Forcing EOS
to be ignored is useful for timing but produces artificial continuations.

The earlier plain-engine fixes remain the useful result: short-chat performance
rose from 11.90 to about 20.7 decode tok/s and 108 to about 209 prefill tok/s.
The earlier llama.cpp+DFlash result remains faster at 24.5 decode tok/s, and
llama.cpp remains preferable for the intended agent server. Those cross-engine
figures use different quantization/tokenization and are not a matched rerun.

Evidence: `final-draft{0,1,2}.json`, `final-comparison.log` and
`final-comparison-telemetry.jsonl` in the local recovery archive below.

The first working port, before the final optimizations, produced the expected
design-note answer prefix, matching plain MLC. Plain MLC measured 20.84/20.82
decode tok/s and 208.58/208.78 prefill tok/s. The initial DFlash port measured
about 11.66 decode tok/s, with mean accepted output length 2.0 per round
(first draft accepted 60.94%, both accepted 39.06%). Target verification was
the dominant cost, about 137 ms per three-token round. These DFlash runs
overlapped CPU-only compilation and exposed a max-token/accounting bug, so
they are exploratory measurements, not the final controlled comparison.

The numerical component checks passed:

- Repeated requests obey exact max_tokens limits 1, 2, 3 and 17, with decode
  accounting equal to completion_tokens minus one. Natural stopping produces
  `391` for the arithmetic prompt, `Done` for the 1768-token prompt, and a
  complete 215-token design-note answer with three summary bullets and two risks.
  That complete answer matches the previously saved optimized plain-MLC answer
  exactly; DFlash decodes it at 13.60 tok/s. These are functional smoke tests.
- GPU argmax matches NumPy for vocabulary sizes through 262144, including
  finite/infinite ties and fully masked rows. Only token IDs return to the CPU.
- A small independent CPU comparison verifies cached versus recomputed draft
  K/V, incremental appends, rejected-prefix slicing, and sliding-window views.
  The tested outputs match exactly.
- An independent FP32 PyTorch calculation using the real draft weights exposed
  residual magnitudes above FP16's 65504 limit. FP32 residuals and norm-before-
  cast eliminate the non-finite outputs in the real-weight fixture.
- The tensor-core projection variant's output has 2.08% relative RMS error
  against that FP32 reference, versus 2.17% for the earlier FP32-projection
  implementation. This is a synthetic-input numerical check, not an accuracy
  or agent-quality evaluation.

Idle real-weight component timings, synthetic inputs, two draft tokens:

| Context | Recomputed K/V draft | Cached/windowed draft | Cached draft with tensor-core feature projection |
|---|---:|---:|---:|
| 161 | 11.13 ms | 9.15 ms | 9.17 ms |
| 448 | 16.08 ms | 11.24 ms | 11.50 ms |
| 1768 | 32.26 ms | 15.43 ms | 15.56 ms |

The first two columns isolate caching/windowing with the old FP32 feature
projection. Their real-weight outputs match exactly at 161/448 context; trimming
the long context changes reduction arithmetic, with 1.03% relative RMS error.
Feature updates are timed separately: the cached variant's three-token update
falls from 4.85–4.88 ms to 1.29–1.31 ms with tensor-core projection. These are
component measurements, not end-to-end decode rates. The final tensor-core
cached output differs from the old uncached output by 1.06–1.29% relative RMS.
Evidence: `final-core-cache-control.json` and `final-core-tc.json`.

The separate q4 GEMV probe passed sampled FP32-reference comparisons. For
three actual tokens, bucket 3 took 0.888 ms versus bucket 4's 1.069 ms at
20480 x 2560, and 0.436 versus 0.529 ms at 2560 x 10240. The optional full-model
three-token specialization is included in the final measured target above.

## Remaining GPU costs

The isolated 32-token CUPTI run completed normally with zero dropped records.
Its decode interval contains 17 verification rounds; the two FFN kernels each
run 714 times (42 layers per round). Approximate GPU-kernel time breakdown:

| Work | Time per round | Share of recorded decode GPU time |
|---|---:|---:|
| Target FFN gate/up + down | 59.87 ms | 45.0% |
| Target attention | 18.22 ms | 13.7% |
| Target vocabulary projection | 14.47 ms | 10.9% |
| Draft vocabulary projection | 9.81 ms | 7.4% |

The three-position quantized FFN GEMVs remain the largest opportunity. Improving
their weight reuse across tokens, then the large vocabulary projections and
legacy attention kernels, is more promising than additional host-side cleanup.
The GPU argmax kernel accounts for less than 1% of this trace. Its separate
transfer-inclusive check takes 0.66 ms for three 262144-way rows, versus 1.17 ms
for copying logits and using NumPy argmax; all tie/masked-row checks pass.

These timings describe an instrumented diagnostic, not a throughput benchmark.
Streaming boundaries are approximate, and compiler warmup affects wall timing.
Evidence: `final-profile.json`, `final-profile.chat.32.0.{prefill,decode}.csv`,
`final-argmax.json`, and `final-validation.log`. The full validation sequence
exited successfully; all test processes and telemetry collectors have stopped.

## Changes

- Port the fork to the current split TVM/tirx/ffi APIs and current serving API.
- Add Gemma4 token-aware prefill/verify exports that expose layers
  `[5, 11, 17, 23, 35]` after their residual/scalar updates. Preserve per-layer
  token embeddings in target verification.
- Read mask ID 4, head size 128, RoPE theta 1e6, and window 1024 from the draft
  config. Fail on missing target/projection exports instead of using zeros.
- Make expensive debug copies/statistics opt-in with `MLC_DFLASH_DEBUG=1`.
- Select greedy tokens on the GPU. Slice away the known frontier before the
  expensive draft vocabulary projection.
- Project each newly committed context row into all layers' K/V once, reuse
  it, broadcast GQA groups, and attend only to the relevant sliding window.
  Rejected rows are not appended. Retained context storage is still linear in
  the full request length, rather than a bounded ring buffer.
- Grow RoPE/mask storage geometrically rather than retaining an allocation for
  every context length. Keep absolute RoPE positions when trimming context.
  Compact offset views into reusable GPU buffers because these TIR entry points
  require zero byte offsets.
- Keep residual sums and projection outputs in FP32; normalize before casting
  to FP16. Use FP16 operands with FP32 accumulation/output for feature projection
  instead of materializing FP32 weights each round.
- Add opt-in CUDA graph hints for DFlash verify, get_logits, and feature
  projection (`MLC_DFLASH_CUDA_GRAPH=1`). Deliberately avoid retaining a graph
  for every growing draft context length.
- Add opt-in exact-three-token GEMV (`MLC_JP5_VERIFY_THREE=1`).
- Clean request-local context after completion/abort and reset on fresh prefill.
  Require prefix caching disabled: target prefix entries do not retain draft
  features. Cap committed blocks at max_tokens/context limits and count the
  prefill completion once per request, not once per model state.

## Abrupt shutdown investigation (restarted 2026-10-31 23:56 CDT)

The previous persistent journal ends at 23:51:38 CDT, shortly after the profiled
engine loaded. There is no shutdown request, thermal-trip message, kernel panic,
or OOM event near that time. An OOM at 19:34 was hours earlier and does not
explain this event. The new boot performed ext4 journal/orphan recovery, and
pstore contains no crash record. This establishes an unclean stop, not its cause.

The root port `0003:00:00.0` reports corrected physical-layer RxErr events about
343–402 times per minute throughout the last 21 complete minutes. It connects
the Intel 7265 Wi-Fi adapter, not the NVMe SSD or integrated GPU. The same errors
continue at idle after reboot. They deserve separate investigation, but the
steady rate and corrected severity do not identify the shutdown trigger.

The user reports USB-C power from a power station's **PD 65 W port**. The station's
2000 W headline rating does not establish this port's available power. The board
was in MAXN with fixed maximum clocks, and CPU compilation overlapped GPU
profiling. A delivery transient/protection trip is a plausible hypothesis; it
is not proven. There was no continuous temperature/voltage record at the failure,
so thermal protection and other hardware/software failures cannot be excluded.

The new boot reports `SYS_RESET_N`, reset status `0x0`, and PMIC NVERC `0x44`;
the previous boot also reported `0x44`, so that value does not distinguish the
new event. UCSI reports 5 V/3 A while the measured high-voltage rails are about
19.8 V; do not use that UCSI value as proof of the negotiated PD limit. The NVMe
reports zero media errors and no critical warning. Its 95 unsafe shutdowns are
a cumulative historical count without a pre-event baseline.

A bounded fan test at PWM 160 reported 2329 RPM; nvfancontrol was restored
immediately. Its quiet profile intentionally permits zero RPM at low temperature.
Serial target compilation peaked at 52 C CPU / 50 C GPU. The full comparison
peaked at 56 C CPU / 60.5 C GPU, with no OC IRQ increments or reported current
alarms. Peak summed monitored rails were 38.32 W, the lowest GPU rail reading
was 19.624 V, and available RAM stayed above 19.9 GiB. These slow samples cannot rule out brief
power transients, and summed rails are not an external wall-power measurement.

Recovery uses one workload at a time, CPU compilation restricted to two cores,
and `scripts/run_monitored.py` to stream telemetry to the local workstation and
terminate its own workload at 85 C or below 2 GiB available RAM. GPU comparisons
retain the original MAXN clocks for comparability but do not overlap compilation.
No power-limit, boot, PCIe, or persistent fan configuration was changed.

Evidence is archived locally under
`~/Workspaces/brain1-llm-bench/mlc-jp5/logs/dflash/recovery-20261031/`, including the preceding journal,
new boot kernel log, fan test, and live telemetry. The pre-recovery experiment
logs are copied under `~/Workspaces/brain1-llm-bench/mlc-jp5/logs/dflash/remote-before-recovery/`.

Reference: NVIDIA's [power/thermal management documentation](https://docs.nvidia.com/jetson/archives/r35.6.0/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonXavierNxSeriesAndJetsonAgxXavierSeries.html)
describes separate software and hardware thermal shutdown mechanisms. The dev
kit's original [65 W supply specification](https://forums.developer.nvidia.com/t/agx-power-adapter/83776)
also means that a 65 W label alone is not evidence of an inadequate supply.

## Reproduce on brain1

The source lives at `~/Workspaces/mlc-dflash-work`; it uses the existing pinned
JetPack-5-compatible dependencies and environment from `~/Workspaces/mlc-jp5`.
The local review copy is `~/Workspaces/brain1-llm-bench/mlc-dflash-work`.

```bash
cd ~/Workspaces/mlc-dflash-work
# Build current runtime, prepare weights, compile final variants and run checks.
bash scripts/build_xavier_dflash.sh

source ~/Workspaces/mlc-jp5/env.sh
export PYTHONPATH="$PWD/python"
export MLC_LIBRARY_PATH="$PWD/build"

# Run sequentially, with all compilation and other inference stopped.
python scripts/bench_xavier_dflash.py --draft-tokens 0 \
  --target-lib dist/libs/target-dflash-graph-b3-sm72.so \
  --cases chat,448,1768 --tokens 128 --repeats 2 --output logs/final-plain.json
python scripts/bench_xavier_dflash.py --draft-tokens 2 \
  --target-lib dist/libs/target-dflash-graph-b3-sm72.so \
  --draft-lib dist/libs/draft-cached-tc-sm72.so \
  --cases chat,448,1768 --tokens 128 --repeats 2 --output logs/final-dflash.json
python scripts/bench_xavier_dflash.py --draft-tokens 2 \
  --target-lib dist/libs/target-dflash-graph-b3-sm72.so \
  --draft-lib dist/libs/draft-cached-tc-sm72.so \
  --cases short,chat --tokens 512 --allow-eos --repeats 1 \
  --output logs/final-natural-stop.json
```

Each case gets a warmup by default. The runner records output text, wall timing, usage,
and before/after engine metrics, and asserts exact requested token counts when
EOS is ignored. `--draft-tokens 2` means a three-position block including the
known frontier. `--draft-tokens 1` runs the one-token control;
`--limits 1,2,3,17 --warmups 0` exercises repeated boundary requests in one engine.
`scripts/summarize_xavier_dflash.py logs/final-draft*.json` extracts timing and
acceptance from the saved metrics. To monitor a workload, run
`python scripts/run_monitored.py --workload-log logs/workload.log -- COMMAND`;
rail readings and CUPTI need sudo on this board. Redirect monitor stdout to a
file on the workstation over SSH so telemetry survives a board power loss.

Historical remote logs: `plain-chat.json`,
`dflash-chat.json`, `debug-chat.log`, `argmax.json`, `context-cache-test-3.log`,
`draft-reference.npz`, `draft-core-tc.json`, `tune-verify-gemv.json`,
`compile-target-graph*.log`, `compile-draft-*tc.log`, `build-9.log`, and
`profile-dflash-v2.*`. `profile-dflash.log` is a failed earlier attempt that
exposed the nonzero-byte-offset constraint, not a performance result. Earlier
draft libraries without the `fp32res` or `tc` suffix fail the finite-output
check and must not be used.

The measured libraries are `target-dflash-graph-b3-sm72.so` and
`draft-cached-tc-sm72.so`. The optional uncached tensor-core library was not
needed for the comparisons above; `build_xavier_dflash.sh` can compile it.
Stochastic sampling, grammar constraints, multi-request serving, preemption,
and broad model-quality evaluation remain unvalidated.
