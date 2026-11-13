# Nemotron with Gemma E4B MTP on Xavier

This path runs the existing Nemotron `q4f16_1` target with the official Gemma
E4B assistant converted to `q0f16`. The assistant consumes Nemotron embeddings,
hidden states and shared KV. Its trained vocabulary filter chooses 32 of 2048
centroids, then evaluates 4096 candidate tokens. Target verification controls
which proposals are committed.

## Measurements

The November 7 short-context comparison uses three fixed prompts of 60, 68
and 90 tokens, 128 forced outputs, one warmup and two measured repetitions.
Thinking and prefix reuse are disabled; runtime capacity is 1024, with one
sequence and 512-token prefill chunks. Hardware is AGX Xavier 32GB, JetPack
5.1.8 / CUDA 11.4 / sm_72, MAXN with fixed clocks.

| Decoder | Pooled decode tok/s |
|---|---:|
| Native MLCEngine plain | 24.504 |
| Native MLCEngine DFlash, two drafts | 30.978 |
| Direct-VM GPU controller plain | 24.376 |
| Gemma MTP, one draft | 32.236 |
| Gemma MTP, two drafts | **37.283** |

The original MTP prototype measured 17.309 / 14.801 tok/s with one / two
drafts. The generic MoE top-k operator processed the centroid row on one
thread. Parallel CUDA selection reduced an actual assistant step from
37.012 to 1.884 ms, preserving its logits, hidden output and candidate IDs
bit for bit. GPU-resident shared state and GPU argmax eliminate repeated
full-state host transfers. The two-draft implementation improves 2.52x and
exceeds fresh native DFlash throughput by 20.4% on these fixtures.

All optimized measured token sequences and acceptance counts match their
original same-width MTP runs. Acceptance is 59.9% / 46.0% for one / two
drafts. Plain/speculative sequences can differ, and forced continuation can
extend past EOS. This is a throughput comparison, not broad model-quality
validation. The 16K serving configuration is separate from this 1K timing
experiment; do not present 37.28 tok/s as a long-context serving guarantee.

[Per-run timings, prompt token IDs and hashes](../benchmarks/xavier/nemotron-gemma-mtp-summary.json)
include the native controls and output fingerprints.

## Build

Use the existing Xavier environment and `dist/target-16k` converted Nemotron
weights from the main deployment runbook. Put the official assistant
safetensors and configuration in `~/Models/gemma-4-E4B-IT-assistant`.

Stop the resident GPU service before compilation, and compile sequentially:

```bash
sudo systemctl stop mlc-nemotron
bash scripts/build_xavier_e4b_mtp.sh ~/Models/gemma-4-E4B-IT-assistant
sudo systemctl reset-failed mlc-nemotron
sudo systemctl start mlc-nemotron
```

The script converts the assistant once, preserves integer token ordering,
and creates target/configuration views by symlinking the existing weights.
It builds 1K and 16K target libraries, assistant libraries for 1K/2K/4K/8K/16K,
and the small GPU argmax helper. `MLC_E4B_MTP_TOPK=1` enables the Xavier CUDA
centroid kernel at compile time; the generic selector remains available.
No base C++ runtime rebuild is needed.

To repeat the short comparison, stop the resident service, source
`scripts/xavier/env.sh`, and run each mode sequentially:

```bash
python scripts/bench_xavier_nemotron_heads.py --mode mtp --output mtp.json
python scripts/bench_xavier_nemotron_heads.py --mode plain --output plain.json
python scripts/bench_xavier_nemotron_heads.py --mode dflash --output dflash.json
```

Restore the service afterward. The recorded source and library hashes refer
to the original experiment; later builds should record their own identities.

## Service profile

The service adapter uses a 16K target and the smallest assistant bucket that
fits the current committed prefix. All five assistant VMs share one parameter
array, with exact parameter-metadata checks before sharing. It remains a
direct-VM Python implementation, not a native MLCEngine speculative mode.

The profile retains the model IDs, authenticated endpoint, protected server
policy, resident embedding/d1 workers and shared FIFO GPU queue. It pins the
server prefix and caches two conversations using the existing safe linear
KV-fork implementation. The 32768-token cache budget includes the active
request, pinned prefix and recycled conversations; old entries are evicted
before allocating a request that would exceed that budget.

A chat request with the header `X-Xavier-Scratch: 1` is a scratch request, for
one-offs such as a [routing decision](xavier_pi_evidence_router.md#scratch-requests).
It is held in a place of its own, where the next scratch request starts from it
and replaces it: it never puts a conversation out, and it is the first to go
when the budget is short.

```bash
sudo install -m 644 deploy/xavier/nemotron-mtp.conf \
  /etc/systemd/system/mlc-nemotron.service.d/nemotron-mtp.conf
sudo systemctl daemon-reload
sudo systemctl reset-failed mlc-nemotron
sudo systemctl restart mlc-nemotron
```

`/healthz` reports `speculative.mode=gemma_mtp`, `draft_tokens` (2 unless
`MLC_NEMOTRON_MTP_DRAFTS` is set), configured context length, cached-conversation count and pinned-prefix length.

Supported behavior includes greedy chat/raw completions, incremental text
SSE, existing buffered native tool calls and client tool-result continuation,
stop strings and validated buffered JSON objects. Generation is restricted to
`temperature=0`, `n=1`; penalties, logit bias, logprobs, echo, suffix and grammar
constraints are rejected explicitly. One verification slot per draft is
reserved in the context budget. Cancellation drains the synchronous GPU worker before
the FIFO admits another model.

### Deployment validation (November 7)

The cold 16K target reproduced all three 128-token benchmark sequences using
the optimized short assistant. Repeated arithmetic requests returned 391 and
reused 342 pinned tokens, then 372 tokens from the prior request. API checks
cover incremental text SSE, tool invocation/result continuation, validated
JSON, authentication, context/token rejection, cancellation followed by
embedding and d1 requests, and the two-conversation cache. Twenty-two focused
CPU checks cover request bounds, tools, policy/JSON handling and queue draining.

A bounded 1280-token fixture exercises the next assistant bucket. With many
repeated meeting notes before the arithmetic question, both plain and MTP
incorrectly answered 401 instead of 391, with and without prefix reuse. This
is retained as a model-quality failure; successful execution is not a quality
guarantee. No full-16K stress or quality sweep was performed.

## All Xavier options — November 11

`deploy/xavier/xavier-full.conf` adds five options to the service profile:

| Setting | Effect |
|---|---|
| `MLC_NEMOTRON_PREFILL=paged` | `hybrid-width` for the current chunk; a Volta tensor-core kernel (`op/cuda/xavier_paged_kernel.cuh`) for every call over the cached pages, prefill and decode. Decode and verification rows keep each probability as two FP16 terms. |
| `MLC_NEMOTRON_MTP_ASSISTANT=gemma4-e4b-mtp-assistant-native-16k-sm72.so` | One 16K assistant built with `MLC_E4B_MTP_ATTN=1`, which attends only the valid keys of its shared K/V state, in place of the five context buckets. |
| `MLC_NEMOTRON_MTP_KV_PREFILL=1` | Prompt tokens except the last run `mtp_prefill_kv`: the 24 layers that store K/V. The last token takes a one-row verification step. |
| `MLC_NEMOTRON_MTP_TARGET=nemotron-e4b-mtp-target-grouped-16k-sm72.so` | A target built with `MLC_XAVIER_Q4=2`: the 4-bit projection kernel for 1–4 rows applies the scales of a product once per group of 32 weights (`mat_q4_grouped` in `op/cuda/xavier_q4.cu`), not per chunk of 8. Built with `MLC_MTP_VERIFY_LEN=4`, so a verification step takes four rows. |
| `MLC_NEMOTRON_MTP_DRAFTS=3` | Three drafts per round. Needs the four-row target. |

Without the last two lines the service uses the default target
(`MLC_XAVIER_Q4=1`, three rows) and two drafts: the profile as first published
on November 11. Every target compiled from this source contains
`mtp_prefill_kv`.

### Build and enable

`scripts/build_xavier_e4b_mtp.sh` builds everything, including the 16K
assistant, the grouped target and the attention module
(`dist/prefill-experiments`). With the
service stopped, as above:

```bash
sudo systemctl stop mlc-nemotron
bash scripts/build_xavier_e4b_mtp.sh ~/Models/gemma-4-E4B-IT-assistant
sudo install -m 644 deploy/xavier/xavier-full.conf \
  /etc/systemd/system/mlc-nemotron.service.d/xavier-full.conf
sudo systemctl daemon-reload
sudo systemctl reset-failed mlc-nemotron
sudo systemctl start mlc-nemotron
```

The drop-in's name sorts after `prefill-width.conf`, whose attention variant
it replaces. The startup log reports `'variant': 'paged'`. Removing the file
and restarting returns the November 7 profile; the bucket assistants are still
built.

### Measurements

AGX Xavier 32GB, MAXN with fixed clocks, 16K libraries, temperature 0, forced
outputs. "First three" is the profile without the grouped target and the third
draft. The long-context rows are from one text (this repository's README and
benchmark guide), each profile in its own process.

| | November 7 profile | First three | All options |
|---|---:|---:|---:|
| MTP decode, 60/68/90-token prompts, pooled tok/s | 37.27 | 38.94 | **42.27** |
| Plain decode through the MTP controller, pooled tok/s | 24.25 | 25.29 | **26.30** |
| Prefill, 161 / 448 / 910 tokens, tok/s | 329 / 522 / 424 | 504 / 807 / 808 | **506 / 832 / 808** |
| Decode at 2,500 / 9,500 tokens of context, tok/s | 24.0 / 10.5 | 36.0 / 29.1 | **42.7 / 40.8** |
| First token, new 9,500-token prompt | 68.6 s | 14.7 s | **14.8 s** |
| First token, 200 tokens added to a cached 9,500-token prompt | 2.92 s | 0.62 s | **0.62 s** |
| A 512-token prefill chunk after 9,280 cached tokens | 6.40 s | 0.97 s | **0.97 s** |

| Verification step, ms | 1 row | 2 rows | 3 rows | 4 rows |
|---|---:|---:|---:|---:|
| 64 cached tokens, first three | 40.2 | 47.9 | 45.7 | — |
| 64 cached tokens, all options | 37.3 | 40.8 | 42.1 | 41.3 |
| 9,792 cached tokens, first three | 49.4 | 57.5 | 55.9 | — |
| 9,792 cached tokens, all options | 47.3 | 51.2 | 52.5 | 51.4 |

Where the time went: a one-row step reads the Q4 weights at memory bandwidth
(about 40 ms), so short-context decode was already near its floor. What grew
with context was attention over the cached pages, the bucket assistants (33 ms
per draft step at 16K) and prefill through 18 layers whose output a prompt
token does not need. Further rows of a verification step were bound by the
arithmetic of the projection kernel, not by memory: grouping it made a fourth
row cost what the third did, which is what lets a third draft pay.

K/V-only prefill is exact. The tensor-core kernel is not bit-identical to the
November 7 kernels. Decode activations are quantized to int8, so small
attention differences flip near-tie greedy choices: 748 of 768 teacher-forced
predictions match (two previously accepted attention variants agree on 755),
and one of the three recorded 128-token outputs is reproduced; the others
diverge after 88 and 124 tokens. The grouped projections round differently
from the kernel they replace: 754 of 768 predictions match it, every changed
one within 0.28 of a tie (the top logit moves by 0.05 at the median and by
2.0 at most), and three of its six recorded outputs are reproduced; the
others diverge after 5, 8 and 80 tokens. Service checks pass
in both forms of the profile (arithmetic 391 with prefix reuse; the
1,280-token fixture still answers 401, as recorded on November 7). No
task-quality benchmark was run.

Because outputs diverge at near-ties, the grouped kernel alone moves the rate
with two drafts by −5% to +8% per prompt, although every round is 5–6%
shorter. With the third draft all ten prompts measured gain, 6% to 40% over
the first three options. A third draft did not pay before: with the earlier
kernel a four-row step cost about 6 ms more than a three-row one.

Tried and not kept: 8- instead of 16-page key segments (within 1%).
Per-run measurements and hashes:
[first three options](../benchmarks/xavier/nemotron-mtp-full-summary.json),
[grouped projections and three drafts](../benchmarks/xavier/nemotron-mtp-grouped-summary.json).

## Rollback

The DFlash weights and libraries are retained. Removing the MTP override
selects the original DFlash service branch:

```bash
sudo mv /etc/systemd/system/mlc-nemotron.service.d/nemotron-mtp.conf \
  /etc/systemd/system/mlc-nemotron.service.d/nemotron-mtp.conf.disabled
sudo systemctl daemon-reload
sudo systemctl reset-failed mlc-nemotron
sudo systemctl restart mlc-nemotron
```

Check authenticated model inventory and a short generation after either
switch. Full-capacity throughput and long stability sweeps are not implied
by the compiled 16K capacity.
