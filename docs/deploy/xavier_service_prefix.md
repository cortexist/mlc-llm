# Protected service prompt and conversation reuse on Xavier

The optional `service-prefix.conf` profile gives Nemotron one stable server policy,
warmed and pinned at startup. It covers concise general answers, local evidence,
measurement provenance and tool use. Client instructions and tool definitions follow
this prefix. Pi continues retrieving local documents and sending them as request data.
Search evidence belongs in conversation messages; it must not change
the system prompt between search and general questions.

The profile retains up to two idle conversation sequences, with a 32,768-token
aggregate KV budget. Each request still has a 16,384-token context limit, including
the injected policy. One shared FIFO GPU queue remains in use. Cached entries can
be evicted for capacity; the pinned policy is protected until engine reset/restart.
New client instructions, tools and uncached history still require prefill. This is
bounded reuse, not a latency guarantee for arbitrary clients or context sizes.

## Safe prefix copies

The original per-layer sliding-window attention path can select incorrect keys
when a sequence shares several KV block depths. The conservative
[DFlash cache profile](xavier_dflash_prefix_cache.md) avoids forks completely.
A pinned prefix requires copying from an active retained sequence, so this profile
adds an opt-in TVM path: copy the matched physical KV pages into one independent
linear block, with the same absolute positions. Existing copy kernels and stream
barriers handle the device transfer. Projected DFlash context is copied to the
same token boundary. Reusing an idle sequence still rewinds it in place.

`MLC_XAVIER_LINEAR_FORK=1` enables this path only for per-layer sliding-window
caches without global cache sliding. The MLC engine requires a capability marker
from the patched TVM runtime before enabling DFlash forks. Other deployments keep
the conservative recycling policy. Copies consume real memory and transfer time;
this implementation does not assume shared physical pages.

The timestamp-to-sequence eviction index now uses an ordered map. The previous
unordered map's `begin()` could evict a recent conversation while retaining an
older one. The two-entry regression checks both recent-entry retention and actual
eviction after unrelated completions.

## Build and deploy

Use the initialized JetPack 5 checkout and model artifacts described in the
[current benchmark procedure](xavier_current_benchmarks.md). Back up the runtime
libraries and preserve existing service drop-ins. Stop inference before building:

```sh
source scripts/xavier/env.sh
trap 'sudo systemctl start mlc-nemotron' EXIT
sudo systemctl stop mlc-nemotron
bash scripts/build_xavier_service_prefix.sh
MLC_XAVIER_LINEAR_FORK=1 python scripts/check_xavier_service_prefix.py \
  --output /tmp/protected-prefix.json
```

The build script applies `patches/tvm-linear-sliding-prefix.patch`, rebuilds TVM
and MLC, and installs the runtime into both the checkout and the Python TVM
package. Updating only the checkout copy is insufficient: Python imports its
package copy first. No model-library rebuild or weight conversion is needed.

Replay private long conversation fixtures against an uncached reference using
`check_dflash_prefix_conversation.py`, with `MLC_XAVIER_LINEAR_FORK=1`, before
accepting a new runtime. The protected-prefix checker covers distinct clients,
repeat hits, idle-entry eviction, unrelated raw completions and cancellation.

Install the profile after validation:

```sh
sudo install -m 644 deploy/xavier/service-prefix.conf \
  /etc/systemd/system/mlc-nemotron.service.d/service-prefix.conf
sudo systemctl daemon-reload
sudo systemctl restart mlc-nemotron
```

`/healthz` reports the policy hash and protected token count. The tool adapter
exposes `usage.prompt_tokens_details.cached_tokens`, including in SSE usage,
so Pi can distinguish cached input from new prefill. Tool-enabled generations
remain buffered before SSE delivery.

Clients should keep system instructions stable across retrieval and general
questions. If a Pi installation has a custom workspace extension, retain its
tools and evidence handling while moving shared behavioral guidance into the
server policy. Use `/reload` after extension edits and `/new` for a clean timing
comparison. Keep long device histories outside automatically loaded context files.

Rollback: remove only `service-prefix.conf`, reload systemd and restart. The older
`prefix-cache.conf` profile then retains one recycled sequence without forks. Restore
the previous Pi guidance extension if the server policy is removed. Model artifacts
and unrelated service profiles are unchanged.

## Measurements

Nemotron 3.5 Bolt E4B on AGX Xavier 32 GB, q4f16_1 target, F16 DFlash
assistant, two proposals, 512-token prefill chunks and one active generation.
These are individual warmed-service latency samples, not medians or sustained-load
results. The Pi RPC session used its normal tools and automatic retrieval, with no
project context files loaded. Large client instruction files add uncached input
and can dominate the first request; keep automatically loaded guidance concise.

| Turn, in the same Pi conversation | Total (s) | Newly prefilled tokens | Cached input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Workspace Nemotron benchmark question | 27.26 | 3,678 | 342 | 211 |
| France question immediately after search | 0.66 | 19 | 4,230 | 4 |
| Undo `git add .` | 1.47 | 17 | 4,252 | 18 |
| Workspace Orin Nano benchmark question | 14.03 | 1,259 | 4,286 | 88 |
| France question immediately after second search | 0.64 | 19 | 5,632 | 3 |

Both France answers were `Paris.`; Git returned `git restore --staged .` with one
sentence explaining that edits are preserved. The workspace answers reported the
recorded prefill/plain-decode values and source paths. No model-issued tool calls
occurred in these five turns; automatic retrieval supplied the local evidence.
System and tool-definition hashes were identical across all five requests.

The prior recycling-only Pi run needed 21.17 seconds for its first France question
after search. The new adjacent-search transitions needed 0.66 and 0.64 seconds.
Search answer wording and output lengths differ between runs, so their totals are
not a fixed-workload throughput comparison. These samples establish the tested
routing behavior, not broad model quality or prompt-injection resistance.

The exact prior 4,346-token continuation produced the same 160-token response text
as its uncached reference, including all benchmark values. It prefills 1,876 tokens
and takes 16.22 seconds, compared with 24.16 seconds uncached.

The 13-case standalone check retains the 342-token protected prefix after distinct
clients, true LRU eviction, three unrelated raw completions and cancellation.
Client proof strings remain distinct. Repeating a 375-token client prompt reuses
374 tokens and takes 0.44 seconds. After unrelated requests evict its conversation,
342 shared tokens remain reusable and the same proof response takes 0.61 seconds.

HTTPS checks also passed native weather tools, supplied tool-result proofs,
repeated-result reuse, cancellation followed by arithmetic, JSON-only output,
incremental plain SSE (8 content chunks) and 768-dimensional embeddings. All
three resident models remained available. Final available RAM was 10,274 MiB
(about 10.0 GiB), with 3 MiB of swap occupied; this is a short-session snapshot.

The checked runtime has not undergone full-16K or sustained multi-client stress
qualification. Cache entries are bounded; changing earlier history, client/tool
instructions, or exceeding the cache budget can still require substantial prefill.
See [`service-prefix-summary.json`](../benchmarks/xavier/service-prefix-summary.json)
for counters and artifact hashes.


## Quick raw-throughput regression check

A short old/current/old runtime comparison used Nemotron E4B with identical
161- and 910-token input IDs, prefix reuse disabled, no injected policy, 64 forced
greedy output tokens, 512-token chunks, one warmup and two measured requests per
case/mode/phase. Both plain and DFlash decoding ran sequentially at pinned clocks.
The old MLC runtime predates DFlash prefix reuse; its TVM runtime predates linear
prefix copies. Target/draft weights, model libraries and attention/dequantization
kernels were identical. The current phase used its deployed 32K aggregate KV
budget versus the old 16K budget; individual context stayed 16K in both.

Old values below average the controls before and after the current phase:

| Prompt tokens | Plain prefill, old → current (tok/s) | Plain decode, old → current (tok/s) | DFlash decode, old → current (tok/s) |
| --- | ---: | ---: | ---: |
| 161 | 332.17 → 332.12 | 23.33 → 23.32 | 33.03 → 33.36 |
| 910 | 424.29 → 423.17 | 18.22 → 18.41 | 32.94 → 32.94 |

No raw-throughput regression was detectable in this check: the largest negative
change was 0.26%, smaller than observed control variation. DFlash prefill changed
by +1.09% and +0.08%. Every compared output matched within its decoding mode.
This is a short check with two measured requests per phase/case, not a statistical
exclusion of sub-percent effects. It does not measure cache-hit copy overhead,
Pi/network latency, or full-context performance. Its 64-token outputs also differ
from the 128-token protocol in the main benchmark tables.

Counters, individual samples, artifact hashes and protocol are recorded in
[`raw-regression-summary.json`](../benchmarks/xavier/raw-regression-summary.json).
