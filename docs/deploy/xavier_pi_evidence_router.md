# Optional d1 evidence routing for Pi

[`pi-evidence-router.mjs`](../../deploy/xavier/pi-evidence-router.mjs) asks the
resident d1 worker where a user request needs evidence. It returns `workspace`,
`inventory` or `agent`; the caller then retrieves documentation, reads live
`/v1/models` metadata, or lets its normal agent answer/act. Routing itself reads
no workspace documents and executes no requested operations. d1 produces a
classification, not a rewritten search query or conversational answer.

This keeps routing inference out of Nemotron's target/draft state and prefix
cache. Both workers still share the GPU FIFO, so queue time applies. The
[shared-weight short bucket](xavier_d1_decisions.md#optional-shared-weight-short-input-bucket)
reduces short routing requests from roughly 2.5 seconds to 0.3 seconds. Larger
packed inputs use the main 2K bucket. This adds routing latency; it does not make
classification free or impose a service latency guarantee.

## Client integration

Use a configuration file with the same schema as the
[Pi specialist extension](xavier_pi_specialists.md):

```json
{
  "baseUrl": "https://xavier.example.com/v1",
  "keyFile": "/home/YOUR_USER/.config/xavier/api-key"
}
```

`PI_XAVIER_CONFIG` overrides its default `~/.config/pi-xavier.json` location.
Use system CA trust. Keys are read at request time and are excluded from results.
HTTPS without embedded credentials is required; redirects are rejected.

```js
import { EvidenceRouter } from './pi-evidence-router.mjs';
const router = new EvidenceRouter();
const result = await router.route(
  userRequest,
  'Xavier is the local inference server; the client has project repositories.'
);
// result.action: workspace | inventory | agent; null means unavailable.
```

Wire this into the client's before-request hook. Supply a short factual
orientation appropriate to the deployment. Add retrieved evidence as conversation
data, preserving a stable system prompt and the user's original request. Retain
normal agent tools for follow-up reads and failures. The module is an optional
building block; installation-specific workspace helpers are separate client code.

The four internal intents are local records, live model inventory, general
information and actions. General information/actions delegate to the agent.
When general information wins but workspace has at least 0.15 of the option
score, the routing policy conservatively requests read-only evidence. This
threshold was selected on development questions; it is not a calibrated
probability of correctness. Scores must be finite, bounded and sum to one.

Exact request/orientation pairs are cached for five minutes, with an LRU limit
of 128 entries. A cached route does not cache model inventory or search results.
A failed or malformed classification returns a null action, allowing a caller's
availability fallback. Slash commands and requests longer than 1,600 characters
bypass routing. The module classifies the supplied request and orientation; it
does not inspect the full conversation automatically.

### Bounded inventory fallback

An inventory route is not proof that `/v1/models` contains the requested fact.
After fetching and validating that response, the client can call:

```js
const coverage = await router.inventoryCoverage(userRequest, orientation, models);
// models contains id, task and context_length from the live response.
// inventory: attach that response; workspace: perform one bounded search.
```

This separate d1 check receives actual inventory values and the meaning of
`context_length`. It distinguishes available deployment fields from missing or
historical facts. It runs only on the inventory path; ordinary questions incur
no second classification. Its cache key includes the request, orientation and
current inventory. Invalid/unavailable coverage decisions request search rather
than claiming inventory is sufficient. There is no internal retry loop.

In the local client trial, a fallback adds the inventory's model names to the
original query to resolve phrases such as “all three workers”, performs one
search with two candidate passages, then supplies evidence to Nemotron. This
does not make the search results relevant by definition. A companion
[`pi-evidence-source.mjs`](../../deploy/xavier/pi-evidence-source.mjs) can reread a
stale candidate when its exact cached passage still occurs uniquely in a current
file under the workspace root. It bounds files to 64 KiB and excerpts to 32 lines
and 2,400 characters; missing, changed, ambiguous or symlinked passages are
rejected. A verified passage is current file text, not a verified answer.

The local integration provides `PI_INVENTORY_FALLBACK=0` for rollback. This is
client orchestration; the server policy, model weights and inference kernels
remain unchanged. d1 coverage remains a classifier and can make mistakes.

## Measurements and limits

An authored 30-question development set exposed failures in an initial
three-choice prompt (27/30), a two-boolean variant (18/30), and a four-intent
prompt (28/30 top-choice routing). Applying the conservative retrieval rule
recovered the development misses. These are tuning results.

On a separate authored 32-question set, with prompt and policy fixed beforehand:

| Router | Correct evidence actions |
| --- | ---: |
| Existing client keyword gate | 16/32 |
| d1 top-choice routing | 30/32 |
| d1 plus conservative retrieval policy | 31/32 |

The remaining miss sent a past RAM measurement to live model inventory instead
of document search. An end-to-end replay of that question finished normally
in 4.14 seconds, acknowledged that inventory metadata contained no RAM data,
and stopped without calling the workspace search tool. It did not recover or
answer the requested fact. Two later server-prompt variants also failed to
trigger recovery; the original prompt was restored.

The bounded client fallback recovered that exact historical-RAM question in a
single real Pi replay: inventory route → coverage rejection → one search →
current-source verification → an answer citing **15,032 MiB (14.7 GiB)** from the
initial three-worker trial. This is a historical snapshot, not current RAM.

| Question in the replay conversation | Total time | Result |
| --- | ---: | --- |
| Historical three-worker RAM | 19.12 s | Correct recorded amount and source |
| Paris, immediately after search | 0.82 s | Correct, no retrieval |
| Git unstage command | 1.64 s | Correct, no retrieval |
| Deployed context limits | 3.08 s | Correct, inventory retained |
| Current model list | 2.77 s | Correct, inventory retained |

Coverage added approximately 0.29–0.30 seconds on the inventory turns, all in the
256-token bucket. The first answer had only the shared 342-token server prefix
cached; these are single end-to-end samples, not throughput benchmarks. Four
development coverage questions informed the data/field-description format, so
the recovery is a targeted regression check, not held-out accuracy validation.

A separate current-free-RAM question took 25.08 seconds and followed the existing
workspace route, bypassing this inventory fallback. It correctly said current
RAM was unavailable but then quoted unrelated hardware measurements. That is
still an answer-quality failure; this change does not fix general relevance or
provide live memory telemetry. An absent ECC-count control declined to invent a
value in 16.63 seconds; it also followed the existing workspace route, so it does
not independently validate the new inventory branch on absent records.

This is a small, same-author validation set, not a general
production accuracy estimate or independent benchmark. Do not hide missing
records by treating supplied context as proof that no records exist.

All measured validation requests fit the 256-token bucket. Median full HTTPS
classification was approximately 300 ms; a repeated client-cached decision took
0.018 ms in one sample. A dedicated cache check retained 723 of 724 Nemotron
prompt tokens after three d1 requests, with identical output. The repeated
Nemotron request took 0.49 seconds versus 1.44 for the initial request.

A real Pi conversation retrieved a previously omitted embedding timing table,
answered the three live context limits, returned Paris and gave the Git unstage
command. Short-answer totals were around 1–2 seconds including d1; inventory was
about 3 seconds. Retrieval can still select a poor passage, and synthesis can
omit workload qualifications. In particular, an embedding-service timing does
not establish full workspace-search latency. Semantic routing does not by itself
qualify end-to-end factual answer quality.

Run the CPU client checks with:

```sh
node --test tests/node/test_pi_evidence_router.mjs tests/node/test_pi_evidence_source.mjs
```

The two parameter-sharing CPU checks run in the MLC Python environment:

```sh
source scripts/xavier/env.sh
python tests/python/serve/test_xavier_shared_params.py
```

Raw user questions, captures, local configuration and detailed experiment logs
remain outside the public source tree. Generic aggregate results are recorded in
[`d1-router-summary.json`](../benchmarks/xavier/d1-router-summary.json).
