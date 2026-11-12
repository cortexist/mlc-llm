# Official Gemma4 Unified 12B MTP experiment

The fork can convert and compile the original
`Gemma4UnifiedAssistantForCausalLM` checkpoint, separately from DFlash.
This is an experimental direct-VM driver, not an `MLCEngine` speculative mode.
An optional [HTTP adapter](../../deploy/xavier/README.md) serves it as
`gemma4-12b` alongside resident Nemotron, with one shared generation queue. The deployed contexts are 16K Nemotron and 4K 12B.
The sections below record the November 1 experiment; its controller, kernels
and speed changed on November 11, see
[All Xavier options](#all-xavier-options--november-11-2026).

The tested pair is the local QAT assistant
`~/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant` (BF16 source, converted
to `q0f16`) and the Heretic 12B target converted to `q4f16_1`. A checkpoint's
provenance does not guarantee acceptance against a modified, quantized target;
the trial below measures this particular pairing.

## Implementation and limits

- The four-layer assistant uses its own tied embedding/LM-head weights,
  pre/post projections, and the target's last local/global KV states. It has
  query/output projections, without independent K/V projections.
- Each draft step concatenates the scaled target embedding and previous hidden
  state. Shared KV and position stay fixed within a draft round; the assistant's
  projected hidden state feeds its next step. This follows the installed
  Transformers 5.22.1 candidate generator and
  [Unified assistant implementation](https://huggingface.co/docs/transformers/model_doc/gemma4_unified_assistant).
- New target exports return normalized hidden states and unpadded, unbroadcast
  KV deltas. Verification accepts the greedy matching prefix and rolls back
  rejected target-cache entries. The controller retains only committed deltas.
  Prompt prefill uses chunks of at most 512, preserving shared KV between chunks.
- One sequence, greedy generation, one or two proposals (three since
  November 11), 4,096 total tokens.
  Local assistant layers mask older shared KV using the reference sliding-window
  rule; global layers retain the complete valid history.
- Assistant KV inputs are padded to the compiled capacity (4,096 in deployment) and masked by valid length. Explicit
  attention reductions avoid CUDA 11.4 cuBLAS-Lt strided-batch dispatch errors;
  fixed buffers avoid a dynamic stack allocation limitation in CUDA codegen.
- In this experiment the Python controller copied KV, hidden states, logits and
  embeddings through CPU memory, and the reported times include that overhead.
  Since November 11 they stay on the GPU. Native engine integration remains
  future work.

## Validation

The CPU test runs two recurrent steps of an entire small assistant with mapped
Transformers weights, including its own LM head. Maximum logit errors are
`5.14e-7` and `8.94e-7`. It also checks target state-export dimensions.

A captured GPU step with the **actual assistant checkpoint** was compared with
Transformers on CPU in FP16. Both chose token 45518. Logit cosine similarity was
`0.9999979`, maximum error `0.03125`; projected hidden cosine was `0.9999985`,
maximum error `0.021484375`. FP16 GPU/CPU operations differ slightly; this is a
single-step reference check, not broad accuracy qualification.

Initial warmed, natural-stop arithmetic trial (25 prompt tokens, eight output
tokens): plain **12.04 tok/s**, MTP **17.40 tok/s**, six of six verified proposals
accepted. Both output the same token IDs and answer `391`. Counters include
verified proposals beyond an EOS encountered while emitting the final block.

The first 16-token design-note trial (161 prompt tokens) measured **11.30 tok/s**
plain and **16.19 tok/s** MTP, with **9/12** proposals accepted and identical
output token IDs. These short samples are exploratory, not a full throughput or
quality benchmark. They include easy channel/header tokens.

The follow-up used one four-token arithmetic warmup per mode, one measured
request per case/mode, and 24 forced output tokens. Both modes used the same 1K target library with both weight sets
resident; output token IDs matched exactly in both cases:

| Prompt tokens | Plain decode tok/s | MTP decode tok/s | Ratio | Accepted / proposed | Plain / MTP request time |
|---|---:|---:|---:|---:|---:|
| 161 | 11.12 | 16.62 | 1.49× | 14 / 17 | 3.861 / 3.167 s |
| 448 | 10.20 | 10.32 | 1.01× | 11 / 21 | 6.330 / 6.314 s |

The 448-token prompt asks for one word, so much of that forced continuation is
past EOS; its near-flat result is not a normal long-answer chat rate. Neither
case establishes sustained throughput. The controller and generic target
verification still have substantial overhead. See the
[saved result summary](../benchmarks/xavier/gemma4-12b-mtp-summary.json) for
token IDs, timings, acceptance counts and reference-check results.

## Build and run on Xavier

First build the fork's Xavier runtime/compiler as described in the README, and
convert the target using [the 12B instructions](xavier_gemma4_12b.md). Stop the
inference service before GPU compilation/testing; restore it afterward. Run
compilation and inference sequentially. Activate the same environment for both.

```bash
source scripts/xavier/env.sh
export MLC_GEMMA4_TEXT_ONLY=1
assistant_src="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant"
assistant_out="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant-q0f16-MLC"
target_weights="$HOME/Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic-q4f16_1-MLC"
sudo systemctl stop mlc-nemotron
trap 'sudo systemctl start mlc-nemotron' EXIT
bash scripts/build_xavier_mtp.sh "$assistant_src" "$target_weights" "$assistant_out"

export MLC_JP5_SINGLE_TOKEN=1 MLC_JP5_Q4_CUBLAS=1 MLC_XAVIER_Q4=1
unset MLC_XAVIER_ATTN
python scripts/bench_xavier_mtp.py --assistant "$assistant_out" \
  --cases short,chat --tokens 16 --capture logs/mtp-step.npz \
  --output logs/mtp-pilot.json
```

The build script reuses existing converted assistant shards; it regenerates
configuration and libraries. It creates `dist/gemma4-12b-mtp-target` with links
to target weights and a dedicated 1K config, plus libraries
`dist/libs/gemma4-12b-mtp-{target,assistant}-sm72.so` and the token-selection
runtime of the controller, `dist/libs/xavier-mtp-runtime.so`.

For the brief forced-output comparison, use `--cases chat,448 --tokens 24
--ignore-eos`. The runner warms both modes, compares each MTP output with the
same direct-VM target-only path, saves results before checking equality, and
fails on differing token IDs. Prefix state is cleared between runs; both weight
sets remain resident in both modes. Decode rate excludes the first token and
includes all subsequent controller work. Results are not interchangeable with
the earlier 4K/128-token MLCEngine benchmark.

After GPU work, check the actual checkpoint step and run the CPU regression:

```bash
python scripts/check_gemma4_mtp_step.py --model "$assistant_src" \
  --capture logs/mtp-step.npz --output logs/mtp-reference.json
python tests/python/model/test_gemma4_mtp.py
sudo systemctl start mlc-nemotron
trap - EXIT
```

The reference check requires a Transformers version containing the Unified
assistant (tested with 5.22.1) and CPU PyTorch. No reference weights or inference
implementation are loaded from remote code.

Xavier runs MAXN/fixed clocks. During this experiment its persistent fan
profile changed from `quiet` to `cool`; `pid` and `open_loop` were retained.
The known power-supply issue remains deferred. Logs are retained in
`~/Workspaces/xavier-llm-bench/mlc-jp5/logs/gemma4-mtp-20261101` on benchmark-client
and `~/Workspaces/mlc-dflash-work/logs/gemma4-mtp-20261101` on xavier.


## Extended context and tools

The 16K / 4K deployment uses `scripts/build_xavier_contexts.sh`; existing source
weights and converted shards are reused. The compiled assistant capacity can
exceed its local attention window. The driver reads capacity from both library
metadata records, retains absolute RoPE positions, masks local attention to the
reference's recent window, and masks padding in all layers. Four recurrent CPU
reference steps include a 1,100-token history; maximum logit errors are
`2.91e-7`, `1.21e-6`, `4.54e-5`, `1.71e-4`. High-position differences reflect the
FP32 inverse-frequency rounding already covered by the target RoPE test.

The native Heretic checkpoint template declares tools and formats calls/results
using the same Gemma4 markers as Nemotron. The HTTP adapter accepts OpenAI tool
schemas and validated call/result histories, parses only complete declared
calls, and emits OpenAI `tool_calls`/`finish_reason: tool_calls`. JSON arguments
remain strings on the API; tool execution happens in the client. Auto/none
choices are supported. Text and tool responses use buffered SSE with keepalives;
malformed/incomplete/undeclared calls fail instead of exposing partial calls.
See the deployment runbook for Pi setup and the measured context trial.

## All Xavier options — November 11, 2026

The kernels written for Nemotron now take the 12B layout: 16 query heads, 8
stored K/V heads, a 1,024-token local window, and one global K/V head that the
cache stores eight times. The MTP path has additions of its own. With the
service stopped, `bash scripts/build_xavier_contexts.sh` builds the 4K
libraries with every build option below, and
[`deploy/xavier/gemma4-12b-full.conf`](../../deploy/xavier/gemma4-12b-full.conf)
enables the rest in the adapter.

| Option | Set for | Effect |
|---|---|---|
| GPU-resident controller | always (`scripts/bench_xavier_mtp.py`, shared with the E4B path) | Shared K/V states, hidden rows and token selection stay on the device. A round no longer pads and uploads 4K states (14 ms) or reads logits back (5 ms). Same tokens. |
| `MLC_GEMMA4_MTP_PREFILL=paged` | adapter | Tensor-core attention over the cached pages, for prefill and verification. The layer kind is a compile-time parameter of the kernel; E4B results are bit-identical to the published build. |
| `MLC_XAVIER_DEQUANT=1` | target build | Vectorized expansion of the 4-bit weights for prefill. Bit-preserving. |
| `MLC_XAVIER_Q4=2` | target and 4-bit assistant build | Small-batch projections with grouped arithmetic: a lane owns 32 weights instead of 8, and the two scales of a product are applied once per group. Rounds differently from `1`, which Nemotron keeps. |
| `MLC_E4B_MTP_ATTN=1` | assistant build | The assistant attends the valid part of its shared K/V states instead of the padded 4K. Same tokens. |
| `MLC_GEMMA4_MTP_ASSISTANT=q4f16_1` | adapter | The assistant with 4-bit weights: 238 MB instead of 838 MB, a draft step of 4 ms instead of 10 ms. |
| `MLC_MTP_VERIFY_LEN=4`, `MLC_GEMMA4_MTP_DRAFTS=3` | target build, adapter | Three proposals per round. A target built this way still serves two, with the same tokens. |

Each step was measured on the previous one: the 161 / 448 / 910-token fixture
and the prose / code / scheduling prompts of the Nemotron comparison (64 / 72 /
94 tokens with the 12B template), 128 forced outputs, one
warmup and two measured repetitions, nothing cached.

| Step | First token (s), fixture | MTP tok/s, fixture | MTP tok/s, prose / code / scheduling |
|---|---:|---:|---:|
| Baseline: `main` `90c147d3`, deployed libraries | 1.77 / 4.03 / 11.97 | 13.26 / 12.33 / 9.46 | 11.78 / 14.18 / 11.85 |
| + GPU-resident controller | 1.75 / 4.02 / 11.98 | 14.82 / 13.67 / 10.20 | 14.23 / 17.19 / 14.26 |
| + tensor-core attention | 1.55 / 2.54 / 5.51 | 17.64 / 19.73 / 19.06 | 15.97 / 18.89 / 16.17 |
| + vectorized expansion | 1.17 / 2.17 / 4.72 | 17.32 / 19.46 / 18.94 | 16.10 / 18.88 / 16.33 |
| + assistant attending the valid context | 1.16 / 2.19 / 4.74 | 20.92 / 22.89 / 21.98 | 19.39 / 22.65 / 19.70 |
| + 4-bit assistant | 1.15 / 2.21 / 4.74 | 23.59 / 26.60 / 24.38 | 20.75 / 25.30 / 22.04 |
| + grouped projections | 1.16 / 2.21 / 4.79 | 27.26 / 29.67 / 26.78 | 23.16 / 28.26 / 23.47 |
| **+ three proposals** | 1.16 / 2.21 / 4.79 | **34.37 / 36.70 / 33.73** | **25.72 / 35.07 / 26.11** |

Plain decoding moves from 11.50 / 10.40 / 9.00 to 12.67 / 12.10 / 11.52 tok/s
on the fixture. One call each, by the number of cached tokens:

| Cached tokens | 64 | 512 | 1,024 | 2,048 | 3,584 |
|---|---:|---:|---:|---:|---:|
| Verification of 3 rows, baseline (ms) | 98.8 | 163.0 | 237.2 | 257.1 | 288.0 |
| Verification of 3 rows, all options (ms) | 81.2 | 86.3 | 92.4 | 95.3 | 97.5 |
| Verification of 4 rows, all options (ms) | 82.1 | 86.7 | 92.8 | 97.2 | 99.3 |
| One row, all options (ms) | 76.1 | 82.8 | 87.8 | 88.5 | 89.3 |
| Assistant step, baseline (ms, plus 14 ms of upload) | 19.3 | 19.8 | 19.2 | 19.9 | 19.3 |
| Assistant step, all options (ms) | 3.5 | 4.0 | 4.0 | 4.2 | 4.5 |

A 512-token prefill chunk behind 0 / 512 / 1,024 / 3,072 cached tokens takes
2.35 / 2.78 / 3.02 / 3.34 s instead of 4.70 / 9.45 / 14.20 / 16.85 s, and a
3,584-token prompt 21.9 s instead of 92.9 s. Of every prefill call, 0.54 s
(0.95 s before) is the expansion of the weights, whatever the number of rows.

### Why three proposals now pay

A one-row step reads all 6.7 GB of weights and is bound by memory. Further
rows only add arithmetic, and the arithmetic of the first small-batch kernel
was the larger part: its projections took 70 / 85 / 79 / 91 ms for one to
four rows of a 12B step. Grouped, they take 65 / 69 / 68 / 67 ms
(`scripts/bench_xavier_q4.py`, modes 6–9; the tile of each row count is the
one measured fastest on the 12B shapes). With the first kernel a fourth
verification row cost 12 ms, and three proposals were slower than two on prose
and scheduling (19.6 against 20.2 and 20.2 against 21.5 tok/s). Now a round of
three costs one assistant step more than a round of two.

How many proposals a round accepts does not predict the next round (after a
fully accepted round of three, 4 of 14 prose rounds were fully accepted again;
after any other, 11 of 39), so the number of proposals is fixed.

### Equivalence

The controller, the vectorized expansion, the assistant's attention and the
four-row library change no token. The controller and the assistant's attention
reproduce the recorded outputs and acceptance counts of the step before them;
the expansion leaves every teacher-forced logit unchanged; a library built with
`MLC_MTP_VERIFY_LEN=4` gives the outputs of the three-row one with two
proposals.

The tensor-core attention and the grouped projections round differently from
what they replace. With tokens forced to the baseline's plain outputs of the
fixture, in one-row and three-row steps (765 predictions):

| Compared | Same prediction | Margins of the others in the first (logits) | Largest change of a shared best logit |
|---|---:|---|---:|
| Tensor-core attention against the TIR kernels | 760 | 0.011, 0.038, 0.159, 0.190, 0.199 | 0.570 |
| Grouped projections against the first kernel | 762 | 0.040, 0.071, 0.071 | 0.565 |
| **All options against the baseline** | **761** | 0.011, 0.038, 0.088, 0.159 | 0.703 |
| For scale: the baseline's one-row against its three-row steps | 379 of 381 | | 0.620 |

Every changed prediction is the runner-up of the other side. Of the baseline's
recorded 128-token outputs, 3 of 6 plain and 3 of 6 two-proposal outputs are
reproduced exactly; the others diverge after 4 to 114 tokens. No task-quality
benchmark was run.

The 4-bit assistant proposes slightly differently: on the fixture 76 / 82 / 80
of its proposals were accepted in two-proposal rounds, against 73 / 80 / 80 of
the FP16 assistant's. Which rows a verification step holds follows from what
was accepted, and a row attends cached keys and keys of its own step through
different kernels, so proposals reach the output at near-ties: 4 of the 6
two-proposal outputs are those obtained with the FP16 assistant. Plain outputs
do not depend on the assistant.

Against sampled FP32 dot products both projection kernels have a relative RMS
error of 0.5–0.6%; the tiles of one kernel are bit-identical to each other.
For Nemotron nothing changes: the attention kernel reproduces the published
build's bits in 35 E4B cases at 0.97–1.02 of its time, the assistant's
attention in 20, and a Nemotron target compiled from these sources gives all
six recorded outputs of its full profile. Both models were loaded in one
process as the service loads them, the 12B through its adapter with every
option: each gave the tokens of its single-model runs in two turns.

### Measured, and left out of the adapter

- **Conversation reuse.** With the conversation cache of the Nemotron adapter
  driving the 12B libraries, the first token of a follow-up turn arrives in
  1.3–1.6 s on 300 / 1,000 / 1,700 cached tokens; a new prompt of that size
  takes 1.7 / 4.9 / 9.5 s. Stored conversations are copied, and a 12B token
  holds 786 KB of K/V state: a 4,096-token budget is 3.2 GB, and an 8,192-token
  one failed to allocate next to the model.
- **Four proposals.** The small-batch kernels take one to four rows.
- **The grouped projections for Nemotron.** On its shapes the microbenchmark
  gives 6–9% per projection, but its tokens would move at near-ties and the
  published full profile would need to be measured again.
