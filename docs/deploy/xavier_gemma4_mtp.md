# Official Gemma4 Unified 12B MTP experiment

The fork can convert and compile the original
`Gemma4UnifiedAssistantForCausalLM` checkpoint, separately from DFlash.
This is an experimental direct-VM driver, not an `MLCEngine` speculative mode.
An optional [HTTP adapter](../../deploy/cerebellum/README.md) serves it as
`gemma4-12b` alongside resident Nemotron, with one shared generation queue. The deployed contexts are 16K Nemotron and 4K 12B.
Nemotron continues to use its existing DFlash path.

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
- One sequence, greedy generation, one or two proposals, 4,096 total tokens.
  Local assistant layers mask older shared KV using the reference sliding-window
  rule; global layers retain the complete valid history.
- Assistant KV inputs are padded to the compiled capacity (4,096 in deployment) and masked by valid length. Explicit
  attention reductions avoid CUDA 11.4 cuBLAS-Lt strided-batch dispatch errors;
  fixed buffers avoid a dynamic stack allocation limitation in CUDA codegen.
- The Python controller copies KV, hidden states, logits and embeddings through
  CPU memory. Reported times include this overhead. Persistent GPU buffers,
  cache views, device sampling and native engine integration remain future work.

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
`dist/libs/gemma4-12b-mtp-{target,assistant}-sm72.so`.

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

Cerebellum runs MAXN/fixed clocks. During this experiment its persistent fan
profile changed from `quiet` to `cool`; `pid` and `open_loop` were retained.
The known power-supply issue remains deferred. Logs are retained in
`~/Workspaces/cerebellum-llm-bench/mlc-jp5/logs/gemma4-mtp-20261101` on nowhere
and `~/Workspaces/mlc-dflash-work/logs/gemma4-mtp-20261101` on cerebellum.


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
