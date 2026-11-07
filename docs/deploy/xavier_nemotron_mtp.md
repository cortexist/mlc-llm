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

```bash
sudo install -m 644 deploy/xavier/nemotron-mtp.conf \
  /etc/systemd/system/mlc-nemotron.service.d/nemotron-mtp.conf
sudo systemctl daemon-reload
sudo systemctl reset-failed mlc-nemotron
sudo systemctl restart mlc-nemotron
```

`/healthz` reports `speculative.mode=gemma_mtp`, `draft_tokens=2`, configured
context length, cached-conversation count and pinned-prefix length.

Supported behavior includes greedy chat/raw completions, incremental text
SSE, existing buffered native tool calls and client tool-result continuation,
stop strings and validated buffered JSON objects. Generation is restricted to
`temperature=0`, `n=1`; penalties, logit bias, logprobs, echo, suffix and grammar
constraints are rejected explicitly. Two verification slots are reserved in
the context budget. Cancellation drains the synchronous GPU worker before
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
