# Optional evidence routing for Pi

[`pi-evidence-router.mjs`](../../deploy/xavier/pi-evidence-router.mjs) asks
resident Nemotron where a user request needs evidence. It returns `workspace`,
`inventory` or `agent`; the caller then retrieves documentation, reads live
`/v1/models` metadata, or lets its normal agent answer/act. Routing itself reads
no workspace documents and executes no requested operations. The model replies
with one option name, not a rewritten search query or conversational answer.

The question goes through the chat route (`max_tokens` 8) as a
[scratch request](#scratch-requests), which keeps it out of the two conversations
Nemotron holds for reuse. All workers share the GPU FIFO, so queue time applies.
This adds routing latency; it does not make classification free or impose a
service latency guarantee.

Until 2026-11-12 the module asked the [d1 decision worker](xavier_d1_decisions.md).
d1 does not fit in memory beside the
[Qwen3.5 pair](../../deploy/xavier/README.md#qwen35-in-place-of-the-12b), and Nemotron,
which is always resident, routes the same validation set as well. The d1 results
are kept below for comparison.

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
The reply has to be one of the option names; emphasis marks and a final period
are ignored, and anything else counts as an unavailable router. The chat route
gives no option scores, so d1's rule of also searching when workspace held at
least 0.15 of the score is gone.

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

This separate check receives actual inventory values and the meaning of
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
remain unchanged. Coverage remains a classifier and can make mistakes.

## Scratch requests

Nemotron's MTP service keeps two conversations for reuse, and a new request puts
out the older one. Routing through the chat route would therefore cost the
client its own conversation: in a probe on the device, the conversation's next
turn reused 1,040 of 1,057 prompt tokens after one routing request, but only the
352 tokens of the server prefix after two in a row, which is what an inventory
question sends (0.98 s to its first token instead of a 0.25 s turn).

A chat request with the header `X-Xavier-Scratch: 1` is held in a place of its
own. The next scratch request starts from it and takes its place; it never puts
a conversation out, and it is the first to go when K/V slots run short. Only
Nemotron's MTP service reads the header.

With the header and both conversation places taken (two probe runs, 2026-11-12;
reuse as the server reports it, time for the whole HTTPS request of 8 tokens):

| Request | Reused prompt tokens | Time |
| --- | ---: | ---: |
| Conversation turn after two routing requests | 1,053 of 1,067 | 0.27–0.32 s |
| The other conversation's next turn | 776 of 794 | 0.39 s |
| Routing request after another routing request | 506 of 527–534 | 0.27–0.33 s |
| First routing request after a service start | 352 of 528 | 0.46 s |

## Measurements and limits

The validation set is a separate authored set of 32 questions. The d1 prompt and
policy were fixed before it was used; for Nemotron the same question and options
were given the closing line `Reply with one word: the option name.` before any
case was run.

| Router | Correct evidence actions |
| --- | ---: |
| Existing client keyword gate | 16/32 |
| d1 top-choice routing | 30/32 |
| d1 plus conservative retrieval policy | 31/32 |
| Nemotron option name (this module) | 31/32 |

Nemotron was measured with the module as committed, over HTTPS, on 2026-11-12:
every reply was an option name, and a decision took 0.27 s at the median (0.39 s
at most). Its one miss sent a request to add a project to the search index to
document search, which costs a retrieval and leaves the action with the agent.
It routes d1's miss, the past RAM measurement, to document search. On the
30-question development set it has 27 right (0.27 s median, 0.48 s at most):
a past client setting went to inventory, and two instructions went to search and
inventory. d1's 0.15 rule was chosen on that set, so the two are not comparable
there.

Inventory coverage: the four development questions of the d1 check came out as
with d1 (two inventory, two search) against the live inventory of three models,
in 0.27–0.42 s each. This is a regression check on four cases, not validation.

Four single questions through a real Pi client, a new process and conversation
each: a capital in 1.0 s, a Git command in 1.5 s and the context limits from the
live inventory in 2.1 s (two decisions), all correct. A question about a
recorded measurement was routed to document search and took 6.6 s; the passages
found did not hold the answer, and the model said so. That is a retrieval limit,
not a routing result.

This is a small, same-author validation set, not a general
production accuracy estimate or independent benchmark. Do not hide missing
records by treating supplied context as proof that no records exist.

### d1 router (2026-11-03)

An authored 30-question development set exposed failures in an initial
three-choice prompt (27/30), a two-boolean variant (18/30), and a four-intent
prompt (28/30 top-choice routing). Applying the conservative retrieval rule
recovered the development misses. These are tuning results.

d1's remaining miss on the validation set sent a past RAM measurement to live
model inventory instead of document search. An end-to-end replay of that
question finished normally in 4.14 seconds, acknowledged that inventory metadata
contained no RAM data, and stopped without calling the workspace search tool. It
did not recover or answer the requested fact. Two later server-prompt variants also failed to
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

## Checks

Run the CPU client checks with:

```sh
node --test tests/node/test_pi_evidence_router.mjs tests/node/test_pi_evidence_source.mjs
```

The cache rules of scratch requests and d1's two parameter-sharing checks run on
the CPU in the MLC Python environment:

```sh
source scripts/xavier/env.sh
python -m unittest tests/python/serve/test_nemotron_mtp.py
python tests/python/serve/test_xavier_shared_params.py
```

Raw user questions, captures, local configuration and detailed experiment logs
remain outside the public source tree. Generic aggregate results are recorded in
[`nemotron-router-summary.json`](../benchmarks/xavier/nemotron-router-summary.json) and
[`d1-router-summary.json`](../benchmarks/xavier/d1-router-summary.json).
