# Resident d1 text decisions on Xavier

This experimental port runs the **text backbone of LiquidAI/d1-Omni-2B** as a
resident decision worker alongside Nemotron and, optionally, EmbeddingGemma 2.
It implements LFM3 in TVM/MLC; it does not use a PyTorch GPU inference backend
or modify the C++ serving scheduler. Audio, images, video and extraction types
are not supported by this first text milestone.

The endpoint is a custom API extension, **not** OpenAI chat completions. It
accepts a JSON `state` and 1–4 named questions (`noul` or single `choice`).
`POST /v1/systemone` is a JEV-style alias of `/v1/decisions`, with the same
authentication, model ID and GPU queue. The supplied specialist profile uses a
2048-token bucket; the default worker remains 256 tokens unless configured.
The complete formatted request must fit the configured capacity; oversized input is rejected,
never truncated. Each request computes one complete forward pass, with no token
generation, causal KV reuse, MTP or DFlash. All resident workers share the same
FIFO for the lifetime of a request, including cancellation cleanup.

## Build and correctness checks

Use the JetPack 5 build environment described in the main README. The original
local checkpoint must include its Python files, tokenizer and `calibration.json`.
Its custom code is loaded from that trusted local directory. This text path
loads the tokenizer directly, avoiding unnecessary vision dependencies.

Stop GPU serving before compilation or direct GPU checks, and restore it after.
Run from the experimental checkout:

```bash
source scripts/xavier/env.sh
export D1_CHECKPOINT="$HOME/Models/d1-Omni-2B"
python tests/python/model/test_d1.py
python tests/python/serve/test_d1_inputs.py
python tests/python/serve/test_xavier_model_queue.py

# FP32 baseline, then the FP16 candidate. GPU work must remain sequential.
bash scripts/build_xavier_d1.sh
python scripts/check_d1_reference.py "$D1_CHECKPOINT" logs/d1-poc
python scripts/check_d1_gpu.py . logs/d1-poc
MLC_D1_PRECISION=f16 bash scripts/build_xavier_d1.sh
MLC_D1_PRECISION=f16 python scripts/check_d1_gpu.py . logs/d1-poc
```

Conversion retains 286 text tensors (1,728,801,536 parameters) and writes
`source-sha256.json`, hashing the original checkpoint files. Original weights
are preserved. FP32 weights occupy about 6.44 GiB; FP16 about 3.22 GiB, excluding
working memory. Artifact locations:

- `dist/d1-text-hf`: extracted original text checkpoint and provenance.
- `dist/d1-text-q0f32` / `dist/d1-text-q0f16`: converted text weights/config.
- `dist/libs/d1-text-{f32,f16}-s256-sm72.so`: compiled libraries.

The CPU check uses the checkpoint's own LFM3 implementation and d1 attention
hook, with actual weights. The GPU check compares hidden states and calibrated
decisions. The small-model test separately exercises convolution, local/global
attention, partial RoPE, sandwich normalization, multiple groups and padding.
FP16 acceptance requires finite output, hidden-state cosine above 0.99999 and
probability drift below 0.005 on the initial fixtures; these are engineering
checks, not a broad confidence-calibration qualification.

## Serving

Enable `MLC_D1=1`, `MLC_D1_PRECISION=f16` and `MLC_D1_CONTEXT=2048`
in the service environment after building the 2K library (below).
`MLC_D1_CHECKPOINT` optionally overrides the original checkpoint directory.
Keep `MLC_GEMMA4_MTP=0` when reserving memory for this trial. Existing Nemotron
and embedding endpoints retain their behavior. Set `MLC_D1=0` and restart to
remove the decision worker. A systemd drop-in change also needs daemon-reload.

With the local CA trusted by the operating system:

```bash
curl https://xavier.example.com/v1/decisions \
  -H "Authorization: Bearer $(cat ~/.config/xavier/api-key)" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "d1-omni-2b",
    "state": "The server cannot resolve DNS names.",
    "questions": {
      "network": {"type": "noul", "instructions": "Is this a network issue?"}
    }
  }'
```

`noul` is P(yes), not a generated answer. A single-choice question uses
`"type":"choice"` and a `criteria` object mapping option names to descriptions.
Its answer includes `choice`, `confidence` (maximum option probability), and
calibrated `probabilities`. A `noul` question may include
`"criteria":{"true":"what yes means","false":"what no means"}`; the
checkpoint formats both descriptions into the question. Unknown question fields,
malformed criteria and unsupported question types are rejected. Usage reports
actual formatted input tokens and zero output tokens. Timing/instance headers
separate GPU forward plus output copy, total worker time, and queue wait.
Pi uses this worker through the
[specialist extension](xavier_pi_specialists.md), alongside the embedding worker;
it is not registered as a conversational model in Pi's `/model` picker.

## Implementation details

Global attention uses the checkpoint's set mask: state reads all questions;
questions read state and their own group. Sliding attention and width-7 gated
convolutions stay causal across the whole packed sequence. LFM3 partial RoPE
retains the full head's frequency spacing and zeroes the unused frequencies.
The output projection scores only required answer-token rows on the CPU in
FP32, followed by the checkpoint's option grouping and calibration. The common
full-vocabulary log-softmax normalizer cancels in that option softmax.

## Experiments and limitations

Initial measurements on Xavier (six warm forward samples per precision):

| Text worker | Median forward + output copy | Reference check |
|---|---:|---|
| FP32 | 995.1 ms | Maximum hidden error 9.35e-5 |
| FP16, original 256-token build | 276.8 ms | Hidden cosine >=0.9999987; maximum probability drift 0.000267 |

Two actual-checkpoint fixtures contain 39 and 81 formatted tokens; both execute
the same padded 256-token bucket. FP16 is 3.59x faster in these samples. The
twelve authored evidence cases scored 12/12 for both d1 and Nemotron; median
HTTPS time was 291.6 ms for d1 and 596.1 ms for Nemotron. One d1 worker instance
served the entire trial. Available system RAM after the trial was 15,032 MiB
(14.7 GiB), with all three workers resident. This is a snapshot, not peak-RAM
qualification. A concurrent HTTP check held d1 in the shared queue for 2,093 ms
until Nemotron finished. [Raw results and limitations](../benchmarks/xavier/d1-text-poc-summary.json).

Run the small evidence-sufficiency/switching trial from any trusted client:

```bash
python scripts/bench_xavier_d1.py \
  --endpoint https://xavier.example.com \
  --key-file ~/.config/xavier/api-key --output d1-trial.json
```

The twelve question/evidence pairs are authored development examples, not a
held-out dataset. Nemotron's generated yes/no answers and d1's probabilities
are preserved, including failures. Repeat-versus-alternating timings from
different inputs are descriptive, not isolated CUDA context-switch latency.
These initial measurements describe the 256-token build, not the later 2K
variant. Audio, sustained workloads and broad calibration/quality remain
unqualified. Do not infer them from short text results or benchmarks on newer
NVIDIA hardware.

Reference: [model and code](https://huggingface.co/LiquidAI/d1-Omni-2B).


## JEV-style mail decisions and the 2K build

This implements the subset used by agentic-mail, not all JEV or native d1 types.
No model retraining is needed. JSON state is passed to the checkpoint's native
formatter, preserving object insertion order; yes/no criteria retain their
native semantics. The response contains `model`, named `answers` and `usage`
with actual packed input tokens and zero generated output tokens. Unsupported
media, extraction, scores and multiple choice are rejected. Choices support
2–26 options. The input bounds are 65,536 characters of JSON-serialized state and 16,384
characters of questions, followed by the stricter token check.

Build from the same unchanged converted FP16 weights, with serving stopped:

```bash
MLC_D1_PRECISION=f16 MLC_D1_CONTEXT=2048 MLC_D1_REUSE_WEIGHTS=1 \
  bash scripts/build_xavier_d1.sh
```

Reproduce the two larger reference comparisons with serving still stopped
(the CPU reference takes about six minutes on this board):

```bash
python scripts/check_d1_reference.py "$HOME/Models/d1-Omni-2B" logs/d1-systemone/reference \
  --context 2048 --fixtures docs/benchmarks/xavier/d1-systemone-reference.json
MLC_D1_PRECISION=f16 MLC_D1_CONTEXT=2048 \
  python scripts/check_d1_gpu.py . logs/d1-systemone/reference --repeats 1
```

This writes `dist/d1-text-config-f16-s2048/mlc-chat-config.json` and
`dist/libs/d1-text-f16-s2048-sm72.so`. It preserves the 256-token config/library.
The serving worker loads the existing `dist/d1-text-q0f16` weights and checks
the selected library's capacity. Without a short-bucket profile, all inputs execute the fixed padded bucket;
a short request also pays for 2048 positions. The optional shared-weight profile
below avoids that cost for small requests. Set `MLC_D1_CONTEXT=256` and
restart to restore the earlier latency/capacity tradeoff.

For agentic-mail, configure its decision API root to
`https://xavier.example.com/v1` (the full `/v1/systemone` URL also works), model
`d1-omni-2b`, and the same API key as the other resident workers. Keep the
language-model endpoint on `/v1` with `nemotron` for drafting. Account feature
toggles are separate. An oversized mail request gets HTTP 400 without silently
losing evidence. The app currently leaves failed decisions unjudged for retry;
it does not automatically fall back to Nemotron after a decision API error.

The development trial uses synthetic mail only, with the application's original
questions and thresholds: junk/phishing 0.85, priority 0.70, reference relevance
0.60, and replyability 0.95/0.05 with the existing address-pattern fallback.
It does not read, classify or send real mail. Run it with:

```bash
python scripts/bench_xavier_systemone.py \
  --endpoint https://xavier.example.com/v1 \
  --key-file ~/.config/xavier/api-key \
  --cases docs/benchmarks/xavier/d1-systemone-cases.json \
  --output systemone-trial.json
```

These authored cases check integration and expose obvious mistakes. They are
not a held-out evaluation or evidence that JEV's confidence thresholds transfer
to d1 on a real inbox. Retain raw probabilities and failures when evaluating.


### Measured 2K results

On Xavier, the 14 synthetic requests took **2,595 ms median over HTTPS** and
**2,485 ms median forward plus output copy**, with 160–1,336 actual input tokens.
All execute the padded 2K bucket. This is substantially slower for small inputs
than the original 256-token build; it provides capacity rather than a speedup.
The original-checkpoint CPU FP32 comparison at 428 and 1,340 tokens passed:
minimum hidden cosine 0.9999913, maximum probability drift 0.002145.

Thirteen of fourteen cases matched every expected label (28/30 labels). The
unsolicited-marketing example was correctly classified as spam but incorrectly
as phishing (0.892) and high priority (0.843). The original thresholds were
retained. The automated-but-monitored example's replyability result used the
app's existing pattern fallback, not a confident model decision.

Six input checks and six queue/budget checks passed. HTTPS checks covered both
endpoint aliases, single choice, authentication, unknown model, invalid criteria,
unsupported media and overflow. The application's actual Rust `DecisionClient`
passed over HTTPS using OS certificate trust. Nemotron drafting and 768-dimension
embeddings still passed. A final three-worker snapshot had 14,563 MiB available;
this is not a peak-memory or sustained-stability qualification.

[Summary](../benchmarks/xavier/d1-systemone-summary.json),
[raw HTTP results](../benchmarks/xavier/d1-systemone-results.json), and
[synthetic cases](../benchmarks/xavier/d1-systemone-cases.json).


## Optional shared-weight short-input bucket

When both 256- and 2048-token FP16 libraries are built from the same checkpoint,
`MLC_D1_SHORT_CONTEXT=256` enables the smaller library for requests whose complete
native packing fits 256 tokens. The public input capacity remains 2048; larger
requests continue using the main library. Install
[`d1-short.conf`](../../deploy/xavier/d1-short.conf) as a service drop-in, reload
systemd and restart. Remove that drop-in and restart to restore the fixed bucket.

The two VMs share one parameter array. Exact parameter metadata (names, order,
shapes, dtype and other recorded attributes) must match before sharing. Both
buckets warm sequentially at startup, with the main bucket first. No weight
conversion or model-library rebuild is needed when both artifacts already exist.
`X-Xavier-Decision-Bucket-Tokens` reports the bucket actually used.

Two short actual-checkpoint comparisons between the buckets had hidden-state
cosines at least 0.9999971 and maximum decision-probability drift 0.001667.
Their decisions agreed. Short execution was about 281 ms versus 2.46–2.52 s
for the 2K bucket. A 536-token request selected the main bucket. Six existing
short synthetic mail cases retained their thresholded verdicts, with maximum
probability drift below 0.000766. Floating-point results can differ between
buckets; these checks do not establish general decision equivalence.

The optional [Pi evidence-router example](xavier_pi_evidence_router.md) uses this
fast path to classify where a question's evidence should come from, without
starting a Nemotron generation or using one of its cached conversation entries.
