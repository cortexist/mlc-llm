# DFlash on brain1: experimental port and optimization record

Work in progress, 2026-10-31. This branch ports
[Lev Stambler's DFlash implementation](https://github.com/Lev-Stambler/mlc-llm--DFlash/tree/b6c7bf7d269fd4127fd1a48c478b01d9dc8ff42b)
onto MLC `44fd06d0`, retaining the measured Xavier fixes in `7e4cd2a5`.
The imported pipeline is recorded separately in `1f8903e1`.

The port executes Nemotron 3.5 Bolt E4B with a q4f16_1 target and the existing
F16 GGUF draft converted without requantization. It is an isolated experiment,
not an installed service. The original MLC checkout/wheel and llama.cpp remain
available. Only greedy, one-sequence text generation is being validated.

## Status and observed results

**No DFlash speedup over the optimized plain MLC engine has been established.**
brain1 became unreachable during the final target builds and diagnostic run.
The latest runtime changes and final graph/specialization variants therefore
still need end-to-end validation. Do not treat the component results as server
throughput or use an unverified library as the default server.

The first working port, before the final optimizations, produced the expected
design-note answer prefix, matching plain MLC. Plain MLC measured 20.84/20.82
decode tok/s and 208.58/208.78 prefill tok/s. The initial DFlash port measured
about 11.66 decode tok/s, with mean accepted output length 2.0 per round
(first draft accepted 60.94%, both accepted 39.06%). Target verification was
the dominant cost, about 137 ms per three-token round. These DFlash runs
overlapped CPU-only compilation and exposed a max-token/accounting bug, so
they are exploratory measurements, not the final controlled comparison.

The numerical component checks passed:

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

Observed real-weight component timings, synthetic inputs, two draft tokens:

| Context | Recomputed K/V draft | Cached/windowed draft | Old feature update (3 tokens) | Tensor-core feature update |
|---|---:|---:|---:|---:|
| 161 | 11.12 ms | 9.13 ms | 4.12 ms | 1.31 ms |
| 448 | 15.82 ms | 11.35 ms | 4.11 ms | 1.31 ms |
| 1768 | 31.53 ms | 15.71 ms | 4.10 ms | 1.39 ms |

These measurements also overlapped CPU-only compilation; rerun idle before
publishing final timings. The comparison combines cached K/V/windowing with
the feature-projection change. A new uncached tensor-core control was building
when the board disconnected.

The separate q4 GEMV probe passed sampled FP32-reference comparisons. For
three actual tokens, bucket 3 took 0.888 ms versus bucket 4's 1.069 ms at
20480 x 2560, and 0.436 versus 0.529 ms at 2560 x 10240. The optional full-model
three-token specialization is still awaiting end-to-end measurement.

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

## Reproduce and finish validation on brain1

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

Each case gets a warmup. The runner records output text, wall timing, usage,
and before/after engine metrics, and asserts exact requested token counts when
EOS is ignored. `--draft-tokens 2` means a three-position block including the
known frontier. Test one draft token as a further end-to-end control. Test
max_tokens 1/2/3 and repeated requests after the final runtime rebuild.

Relevant remote logs (retrieve after connectivity returns): `plain-chat.json`,
`dflash-chat.json`, `debug-chat.log`, `argmax.json`, `context-cache-test-3.log`,
`draft-reference.npz`, `draft-core-tc.json`, `tune-verify-gemv.json`,
`compile-target-graph*.log`, `compile-draft-*tc.log`, `build-9.log`, and
`profile-dflash-v2.*`. `profile-dflash.log` is a failed earlier attempt that
exposed the nonzero-byte-offset constraint, not a performance result. Earlier
draft libraries without the `fp32res` or `tc` suffix fail the finite-output
check and must not be used.

Outstanding: finish/check the interrupted builds, verify the final runtime's
short/long/stop-limit behavior, run idle comparisons and profiling, and measure
actual acceptance. Stochastic sampling, grammar constraints, multi-request
serving, preemption, and broad model-quality evaluation remain unvalidated.
