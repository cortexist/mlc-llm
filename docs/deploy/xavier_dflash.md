# DFlash on brain1: experimental port and optimization record

Measured on 2026-11-01. This branch ports
[Lev Stambler's DFlash implementation](https://github.com/Lev-Stambler/mlc-llm--DFlash/tree/b6c7bf7d269fd4127fd1a48c478b01d9dc8ff42b)
onto MLC `44fd06d0`, retaining the measured Xavier fixes in `7e4cd2a5`.
The imported pipeline is recorded separately in `1f8903e1`.

The port executes Nemotron 3.5 Bolt E4B with a q4f16_1 target and the existing
F16 GGUF draft converted without requantization. It is an isolated experiment,
not an installed service. The original MLC checkout/wheel and llama.cpp remain
available. Validation covers greedy, one-sequence text generation.

## Status and observed results

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
