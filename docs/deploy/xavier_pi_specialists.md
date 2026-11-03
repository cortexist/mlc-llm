# Use the three resident workers from Pi

Nemotron is Pi's conversational model. EmbeddingGemma 2 and d1 are available
through tools and slash commands supplied by
[`pi-specialists.ts`](../../deploy/xavier/pi-specialists.ts).
Pi's `/model` selects chat models; listing a vector encoder or decision model
there would send an incompatible chat request.

| Resident model | API | Pi access |
|---|---|---|
| `nemotron` (Pi alias `nemotron:e4b`) | `/v1/chat/completions` | Normal conversation and tools |
| `embeddinggemma-2` | `/v1/embeddings` | `xavier_rank`, `/rank`, `/embed` |
| `d1-omni-2b` | `/v1/decisions` | `xavier_decide`, `/decide` |

All endpoints share existing TLS authentication and the GPU request queue.
The optional service drop-in [`specialists.conf`](../../deploy/xavier/specialists.conf)
explicitly enables these three workers and disables the resident 12B worker.
Install it as a systemd drop-in after building the required libraries; reload
systemd and restart the service if its effective environment has changed.

## Client setup

Use Pi 1.4.0 and Node's system CA trust (`NODE_USE_SYSTEM_CA=1`). Install the
extension under `~/.pi/agent/extensions/xavier-specialists.ts`, either by copying
it or by symlinking to your checkout. Pi discovers it on startup or `/reload`.

Create `~/.config/pi-xavier.json` with your actual endpoint and an **absolute**
path to the existing API-key file, then set its permissions to 0600:

```json
{
  "baseUrl": "https://xavier.example.com/v1",
  "keyFile": "/home/YOUR_USER/.config/xavier/api-key"
}
```

`PI_XAVIER_CONFIG` can override the configuration path. The extension reads
the key when needed, uses verified HTTPS, rejects redirects, and starts no
network connections during discovery. No certificate verification is disabled.
Keep Nemotron in Pi's custom provider registry and remove inactive 12B entries
from the active model picker (retain a backup to restore later).

## Try it

Inside Pi:

```text
/local-models
/decide Is this a network issue? | DNS lookup failed.
/embed task: search result | query: password reset
/rank {"query":"reset password","documents":["Reset your password using the account recovery link.","Saturn has bright rings."]}
```

`/local-models` queries the actual API inventory and explains each worker's
role. `/decide` returns P(yes); `/embed` shows the 768-dimensional vector's norm
and a short preview; `/rank` returns passages ordered by cosine similarity.
Direct commands do not need a Nemotron generation. For raw vector data, use
the embeddings API.

Nemotron can also invoke the specialist tools during ordinary conversation:

```text
Use xavier_decide to assess whether "DNS lookup failed" describes a network issue.

Use xavier_rank to rank these two passages for "reset password":
1. Reset your password using the account recovery link.
2. Saturn has bright rings.
```

`xavier_decide` accepts `state`, `question`, and optional `options` (objects
with `name` and `description`) for single-choice classification. `xavier_rank`
accepts `query` and 1–15 `documents`. It computes similarity locally after
obtaining embeddings. The local note files stay on the client; only supplied
passage text is sent to the inference API. The tool does not scan or index
the filesystem automatically.

Limits are enforced by the server: d1's entire formatted request must fit
the configured capacity (2048 tokens with the current specialist profile;
256 with the original build); each embedding input must fit 128 tokens including its task prefix.
Use short passages. No truncation is performed. The current d1 port is text-only.
Similarity is not factual verification, and a decision probability is not proof.

## Verified integration

The actual Pi → Nemotron → specialist → Nemotron path was tested for both tools.
d1 returned P(network issue)=0.8914 and Nemotron reported it correctly.
EmbeddingGemma ranked the password-recovery passage above the astronomy passage
(cosines 0.8966 versus 0.5654). Both direct commands and API inventory discovery
were also exercised. These are integration checks, not quality benchmarks.

Liquid's [LFM3 announcement](https://www.liquid.ai/blog/lfm3) describes Pi agent
support for **LFM3-1.7B**, a generative sibling. The
[d1-Omni-2B model card](https://huggingface.co/LiquidAI/d1-Omni-2B) explicitly
describes a decision model that does not generate conversational tokens.
