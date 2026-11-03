# Resident model HTTPS service on Xavier

Hostnames, addresses and machine-specific paths below are examples. Replace
`xavier.example.com`, `192.0.2.10`, the tailnet address and SSH aliases with your
own values. Historical network identifiers are anonymized in public summaries;
benchmark measurements are unchanged.

The optional **embedding switching PoC** profile is currently active: resident
`nemotron` chat and `embeddinggemma-2` text embeddings share the same GPU queue.
The 12B worker is unloaded by `embedding-poc.conf`; its artifacts and the base
dual-chat configuration below are retained. See the
[PoC measurements, API and restore instructions](../../docs/deploy/xavier_embedding_switch_poc.md).

The deployed OpenAI-compatible API is **`https://xavier.example.com/v1`**, with
model IDs **`nemotron`** and **`gemma4-12b`**. `/healthz` reports readiness. Model listing and
generation require `Authorization: Bearer <API key>`.

Both targets and their assistants stay resident. Select the model in each
chat request; switching does not unload/reload weights. One shared FIFO queue
serializes complete generations across both models, including streaming and
Nemotron tool calls. `X-Xavier-Queue-Wait-Ms` reports time spent waiting.
The existing Pi default remains Nemotron.
The existing `nemotron:e4b` client ID is also accepted as a Nemotron alias.

| API model | Decoder | Context | Tools | Response streaming |
|---|---|---:|---|---|
| `nemotron` | DFlash | 16,384 | Yes | Incremental chat; buffered tool turns |
| `gemma4-12b` | Official QAT MTP, two proposals | 4,096 | Yes | Buffered SSE with keepalives |

The 12B route uses the Heretic target and the original Unified QAT assistant.
It supports text `/v1/chat/completions`, greedy single-output generation,
and native Gemma4 tool calls, and reserves two context slots for verification. `max_tokens` must fit the
remaining context; its default is the smaller of 512 or the available budget.
Oversized inputs and forced tool choices, stop strings, penalties, logprobs,
response constraints and debug settings return HTTP 400. Unknown model IDs
return HTTP 404. This HTTP adapter uses the direct-VM implementation;
it does not add an `MLCEngine` MTP speculative mode. See
[the MTP build and validation notes](../../docs/deploy/xavier_gemma4_mtp.md).

## Access from clients

- On the LAN, the router resolves `xavier.example.com` to `192.0.2.10`.
- On the existing Headscale tailnet, MagicDNS's extra record resolves the same
  name to `100.64.0.10`. Enable Tailscale DNS on the client. Xavier itself
  retains LAN DNS with `--accept-dns=false`.
- Clients must trust their local CA certificate, or pass that file as their CA
  bundle. Ensure the CA's name constraints permit your chosen hostname and
  the leaf certificate includes it. Use the same hostname over the tailnet.
- The API key is stored with mode 0600 at
  `benchmark-client:~/.config/xavier/api-key` and
  `xavier:~/.config/mlc-nemotron/api-key`. Copy it securely to authorized
  clients; keep it out of source control.

On a client with the root certificate and key file copied locally:

```bash
export XAVIER_API_KEY="$(cat ~/.config/xavier/api-key)"
curl --cacert ./local-ca.crt https://xavier.example.com/v1/models \
  -H "Authorization: Bearer $XAVIER_API_KEY"
curl -N --cacert ./local-ca.crt https://xavier.example.com/v1/chat/completions \
  -H "Authorization: Bearer $XAVIER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"stream":true}'
```

For an SDK or application, set its base URL to `https://xavier.example.com/v1`,
its model to `nemotron` or `gemma4-12b`, and its API key to the saved key. Configure its TLS
trust store with the local CA. An explicit tailnet connectivity test can use
`curl --resolve xavier.example.com:443:100.64.0.10` with the same CA and credentials.

## Installed services

`mlc-nemotron.service` runs as `zero` from the validated experimental checkout
at `~/Workspaces/mlc-dflash-work`. It uses that checkout's Python
code, runtime, converted weights and compiled model libraries. The API listens
on `127.0.0.1:8000`. Nginx terminates TLS on port 443, permits LAN/loopback and
tailnet clients, and passes streaming responses without buffering. Public IPv6
clients are denied. No router port forwarding or Tailscale Funnel is configured.

The model configuration is the measured Nemotron target, F16 DFlash draft,
two proposals (`spec_draft_length=3`), `hybrid-width` prefill via the optional
`prefill-width.conf` drop-in, one active sequence,
16K context. The optional `prefix-cache.conf` drop-in enables
`MLC_NEMOTRON_PREFIX_CACHE=radix` with one recycled prefix; the code defaults to
`disable`. See the [DFlash prefix-reuse checks and rollback](../../docs/deploy/xavier_dflash_prefix_cache.md).
The optional [protected service-prefix profile](../../docs/deploy/xavier_service_prefix.md)
adds a server-owned policy, safe independent KV prefix copies and two recent
conversation entries. Its aggregate KV budget is 32K; each request still has a
16K context. Pi's search routing no longer changes the system prompt.
Multiple HTTP clients can queue requests. The service restarts on failure, with a bounded restart rate.
Nginx, MLC and tailscaled are enabled at boot.

The [November 2 prefill follow-up](../../docs/deploy/xavier_prefill_followup.md)
documents the native rebuild, checks and rollback to the base unit's `hybrid`
variant. The new variant changes local current-chunk attention width; weights,
context capacity and small-batch decode/verification kernels are retained.

The [weight-expansion follow-up](../../docs/deploy/xavier_prefill_linear.md)
adds `prefill-linear.conf`, selecting the vectorized Q4 expansion target via
`MLC_NEMOTRON_MODEL_LIB`. It improves matched prefill rates by 11–28%.
Build and validate the library before installing the drop-in. Removing this
drop-in restores the old target library; the attention-width setting remains.
The separate dense GEMM selection experiment is not enabled in the service.

`mlc-nemotron.service.d/dual-model.conf` sets `MLC_GEMMA4_MTP=1`, enabling the
second resident target/assistant and an arithmetic startup check. The API
becomes ready after both models load. Startup loads and warms 12B first, then
initializes Nemotron: a Nemotron-first restart encountered a late 1 GiB NvMap
allocation failure despite available system RAM. The MTP VM is constructed and called on
one dedicated thread; cancellation drains that worker before releasing the
shared GPU queue. Its chat template comes from the original Heretic checkpoint,
because MLC conversion omits `chat_template.jinja`.

To install the optional second model after building the MTP artifacts, copy
`scripts/gemma4_mtp_service.py`, `scripts/bench_xavier_mtp.py`,
`scripts/xavier_serve_queue.py` and the updated `scripts/serve_xavier_nemotron.py`
to the deployed checkout, then install the drop-in:

```bash
sudo install -d /etc/systemd/system/mlc-nemotron.service.d
sudo install -m 644 deploy/xavier/dual-model.conf \
  /etc/systemd/system/mlc-nemotron.service.d/dual-model.conf
sudo systemctl daemon-reload
sudo systemctl restart mlc-nemotron
```

For Nemotron-only operation, set `Environment=MLC_GEMMA4_MTP=0` in that drop-in,
reload systemd and restart the service. Keep the shared-queue helper beside the
serving script in either mode. No nginx, certificate, DNS or API-key change is
needed when enabling/disabling the second model.

Generation defaults to `temperature: 0`. Explicit nonzero temperatures or
`n` values other than 1 return HTTP 400: the experimental Xavier DFlash
sampler is qualified for greedy, single-output requests. Non-greedy sampling
can otherwise terminate its background worker.

### Pi on benchmark-client

`~/.pi/agent/models.json` registers `xavier/nemotron:e4b`, reads the API key
from the private credential file, and declares the 16384-token context and
2048-token output limit. `~/.pi/agent/bin/pi-xavier` selects it and loads
the system trust store through `NODE_USE_SYSTEM_CA=1` before Node starts:

```bash
pi-xavier
# Or select the provider directly:
NODE_USE_SYSTEM_CA=1 \
  pi --provider xavier --model nemotron:e4b --thinking off
```

On benchmark-client, the root is installed at
`/etc/ca-certificates/trust-source/anchors/local-ca.crt` and extracted by
`sudo update-ca-trust`. `/etc/environment` and
`/etc/profile.d/local-node-system-ca.sh` enable Node system-store trust for new
login sessions. Existing shells can run `export NODE_USE_SYSTEM_CA=1`. Some
applications maintain independent trust stores and need their own configuration.

Pi uses `openai-completions`, bearer authentication, `max_tokens`, system-role
messages and greedy sampling. Its per-model compaction settings reserve 2048
tokens and keep 4096 recent tokens. The default provider is `xavier` and the default client model is
`nemotron:e4b`. The `pi-llama-cpp` package remains installed with
`extensions: []`, suppressing its independent localhost:9931 startup probe;
re-enable it through `pi config` when using that integration. The ordinary Pi
launcher defaults `NODE_USE_SYSTEM_CA=1` for existing terminal environments.
Tools are enabled: the launcher has no `--no-tools` flag and the model's
`samplingParams` contains `temperature: 0, top_p: 1` without `tool_choice: none`.
Start a new Pi process after changing its model configuration.

`models.json` also registers `xavier/gemma4-12b` (4,096 context, 512 output
tokens), with compaction reserve/retention of 768/1024. Reopen `/model` to reload
Pi's local registry. Nemotron remains the default. For a fresh 12B session on benchmark-client:

```bash
~/.pi/agent/bin/pi-gemma4
```

The launcher enables read/bash/edit/write with a short system prompt and omits
MCP, extensions, skills and workspace context files to fit the smaller window.
Long existing coding transcripts may still exceed 4K. Both models accept native
OpenAI tool-call histories and text content blocks. Tools execute on the client.
12B buffers tool calls before emitting complete OpenAI SSE deltas.

Pi 1.4.0's fixed 4,096-token safety margin clamps a 4K model's output to one
token, so the 12B model sets `samplingParams.max_tokens: 512` explicitly. Server
checks still enforce the actual prompt/output/verification budget.

`pi-clm` also remains installed with `extensions: []`. Its default guard uses
`window − 4096 − reserve` (2048 input tokens on this model), so it withheld even
a one-line read result and then withheld the model's attempt to retrieve it.
Pi's own configured compaction remains active. Re-enable the extension through
`pi config` when using a suitable context size or after tuning its policy.

### Native Gemma4 tool calling

The Xavier serving script installs `scripts/gemma4_tools.py` ahead of MLC's
ordinary chat route. Requests with tools or tool history use the Nemotron
checkpoint's NVIDIA chat template, including native tool declarations, calls
and results. Raw completion generation uses the same optimized engine and
DFlash configuration. A parser converts complete native calls to OpenAI
`tool_calls` with JSON argument strings, and accepts the corresponding
`role: tool` results on subsequent requests. Pi executes tools on the client
computer; the service only generates calls.

The adapter loads tokenizer/template files from
`~/Models/nemotron-3.5-Bolt-E4B-BF16`, or `MLC_NEMOTRON_TOKENIZER` when set.
It does not load that checkpoint's BF16 weights. Keep its tokenizer files and
`chat_template.jinja`; this deployment uses Transformers 5.22.1 from the
existing MLC virtual environment. Copy the adapter beside the serving script
when deploying, along with the protocol update accepting string arguments.

Supported tool choices are `auto` (the default) and `none`; forced/required
tool choices, tool-conversation images, logprobs and response-format constraints
are not implemented. Tool generations are buffered until the turn completes,
then emitted as OpenAI SSE deltas with usage and finish reasons. SSE comments
keep the connection active during generation. Malformed/incomplete calls and
undeclared tool names produce errors rather than executable partial calls.
Ordinary chat without tools retains MLC's existing streaming path. The model context limits and greedy-only restriction still apply.

CPU regression checks:

```bash
python tests/python/serve/test_gemma4_tools.py
# In the deployed MLC environment, this also exercises typed requests and SSE:
PYTHONPATH=python MLC_LIBRARY_PATH=build \
  ~/Workspaces/mlc-venv/bin/python tests/python/serve/test_gemma4_tools.py
```

Verified over HTTPS: a native `read` request returned a structured tool call;
the next streaming turn accepted its matching result and quoted its proof word.
All seven CPU checks pass in the deployed environment. A real Pi run with the
ordinary tools and workspace context read a local file and returned its exact
random proof word. No file contents were included in the user's test prompt.

```bash
ssh xavier 'sudo systemctl status mlc-nemotron nginx tailscaled'
ssh xavier 'sudo journalctl -u mlc-nemotron -f'
ssh xavier 'sudo systemctl restart mlc-nemotron'
```

Templates are included beside this file. If moving the deployment, update the
checkout, venv and credential paths in the systemd unit. Keep compilation and
benchmarks serialized with the serving process; restore the service afterward.

## Certificates and DNS

The leaf is signed by `~/LocalAuthority/local-ca.crt` on benchmark-client. Its SANs are
`DNS:xavier.example.com` and `IP:192.0.2.10`. The initial leaf expires on
2027-11-01. The CA private key stays on benchmark-client; only the leaf key is installed
on xavier, root-owned at `/etc/nginx/tls/xavier.key` (0600).

Renew on benchmark-client using the saved CSR and extensions; OpenSSL prompts for the
CA key passphrase:

```bash
cd ~/LocalAuthority
openssl x509 -req -in crts/xavier.csr \
  -CA local-ca.crt -CAkey local-ca.key -CAcreateserial \
  -out crts/xavier.crt -days 365 -sha256 -extfile exts/xavier.ext
scp crts/xavier.crt xavier:.config/mlc-nemotron/xavier.crt
ssh xavier 'sudo install -m 644 ~/.config/mlc-nemotron/xavier.crt /etc/nginx/tls/xavier.crt && sudo nginx -t && sudo systemctl reload nginx'
```

The tailnet record lives in
`headscale-server:/etc/headscale/extra-records.json`; Headscale watches that file.
Its config references the file through `dns.extra_records_path`. The preceding
configuration is backed up at
`/etc/headscale/config.yaml.before-xavier-https`. The node was enrolled
under user `zero` with a single-use key; enrollment key files were removed.

## Verification

Verified from benchmark-client: CA/hostname validation, LAN IP SAN validation,
authenticated `/v1/models`, unauthorized HTTP 401, ordinary and streaming
chat (`391`, three SSE events), two queued clients (`42` and `121`), and chat
through the tailnet DNS name. TLS verification remains enabled throughout.
These are deployment checks; hardware stability diagnosis remains deferred.

The dual-resident trial on 2026-11-01 additionally verified alternating models
without reloading, both JSON/SSE paths, unknown-model/context/sampling errors,
cross-model queueing, a cancelled 12B stream followed by Nemotron, a 623-token
12B prompt using chunked prefill, and a Nemotron `get_weather` tool call. Four
queue/budget/cancellation CPU tests and the seven existing tool tests pass.
During the initial HTTP checks, sampled total system RAM peaked at 18,803 MiB
(18.36 GiB), with at least 11,894 MiB available. The final resident snapshot was
18,630 MiB used / 11,978 MiB available, with 5 MiB of swap occupied. These are
whole-device readings, not isolated worker memory, and short functional checks,
not a sustained-load or maximum-context qualification. A second restart with
12B-first warmup also passed, including the existing Pi model alias. Detailed local logs are
under `benchmark-client:~/Workspaces/xavier-llm-bench/mlc-jp5/logs/dual-resident-20261101`;
see the [saved summary](../../docs/benchmarks/xavier/dual-resident-summary.json).


## Building the 16K / 4K context libraries

With the original converted artifacts present and the service stopped, run
`bash scripts/build_xavier_contexts.sh` from the repository root. It creates
separate `dist/target-16k`, `dist/draft-16k`, `dist/gemma4-12b-mtp-target-4k`
and `dist/gemma4-12b-mtp-assistant-4k` configurations linking existing weights.
No weight reconversion or C++ runtime rebuild is required. The serving script
uses the corresponding `*-16k-sm72.so` and `*-4k-sm72.so` libraries. The original
8K/1K artifacts remain available for rollback with the prior serving script.

The 12B assistant keeps full shared KV history in bounded 4K buffers. Its local
layers mask old states using the Transformers assistant's sliding-window rule;
global layers see the full valid prefix. Absolute RoPE positions and frozen
KV/position within each draft round are preserved. The driver derives capacity
from library metadata and keeps prompt chunks at 512. Four recurrent CPU steps,
including a 1,100-token history with deliberately large masked local values,
are checked against Transformers; high-position tolerances account for the
same FP32 RoPE frequency-rounding differences as the target port.


### Context/tool qualification on 2026-11-01

All four new libraries compiled and the service advertises 16,384 / 4,096 tokens.
Both models answered arithmetic correctly. 12B generated a complete weather
call, accepted its matching result, copied a random proof word from that result,
and emitted structured tool-call SSE. After a device shutdown/reboot, the
service recovered automatically; a 1,220-token 12B prompt decoded correctly
with two of two proposals accepted. A real `pi-gemma4 --tools read` run read a
local temporary file and returned its random proof word, which was absent from
the user prompt. Six queue/budget/tool-adapter tests, seven shared tool tests,
and the assistant reference checks pass.

Sampled available system RAM stayed at or above 8,555 MiB before shutdown and
8,576 MiB during the brief post-reboot checks (about 8.4 GiB). These readings
include both resident model/assistant pairs. The near-4K prefill test was
interrupted when the board became unreachable; the near-16K request was never
run. Full-context peak RAM, sustained operation and long-context quality remain
unqualified. The power issue is deferred; the short checks and advertised
capacity do not guarantee 8 GiB available under every full-context workload.
See [the saved context/tool results](../../docs/benchmarks/xavier/context-tools-summary.json).

## Buffered JSON-object clients

Startup reserves Nemotron's 512-token prefill workspace on the serving event
loop before loading the embedding and decision workers. This avoids the late
1 GiB NvMap allocation failure observed with a mail-classification prompt when
specialists loaded first. Readiness is reached only after the warmup and all
configured workers finish loading.

Native tool headers accept whitespace after the tool marker and `call:`. For
example, `call: bash{...}` and `call:bash{...}` both map to the declared `bash`
tool. Unknown names and malformed arguments remain errors; the adapter never
executes tools or guesses a replacement name. This fixes a Pi/Nemotron failure
that previously reported a valid spaced header as an invalid tool name.

The experimental DFlash engine can commit tokens rejected by XGrammar and stop
its worker thread. The Python adapter therefore handles non-streaming chat
`response_format: {"type":"json_object"}` by generating without grammar
constraints, then parsing and validating the complete object before returning
it. Invalid/incomplete/non-object output returns HTTP 502, never a fabricated
result. The response header `X-Xavier-JSON-Mode: validated-output` identifies
this path. Clients should explicitly request the desired JSON in their prompt.
Schema-constrained JSON, streaming JSON, JSON with logprobs, and JSON through
the legacy completions endpoint are rejected with HTTP 400.

This supports clients such as agentic-mail using Nemotron for both drafting
and structured mail decisions. Set the language-model API root to the HTTPS
`/v1` endpoint, model `nemotron`, API key, and temperature 0. Leave the separate
System One decision API unset to use the application's language-model fallback.
The [d1 System One adapter](../../docs/deploy/xavier_d1_decisions.md) supports
the application's JSON state and `/systemone` protocol with a separately built
2K input bucket. Set the decision model to `d1-omni-2b` and keep `nemotron`
for drafting; inputs beyond the packed capacity are rejected without truncation.
