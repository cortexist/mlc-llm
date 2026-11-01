# Resident model HTTPS service on cerebellum

The deployed OpenAI-compatible API is **`https://cerebellum.lan/v1`**, with
model IDs **`nemotron`** and **`gemma4-12b`**. `/healthz` reports readiness. Model listing and
generation require `Authorization: Bearer <API key>`.

Both targets and their assistants stay resident. Select the model in each
chat request; switching does not unload/reload weights. One shared FIFO queue
serializes complete generations across both models, including streaming and
Nemotron tool calls. `X-Cerebellum-Queue-Wait-Ms` reports time spent waiting.
The existing Pi default remains Nemotron.
The existing `nemotron:e4b` client ID is also accepted as a Nemotron alias.

| API model | Decoder | Context | Tools | Response streaming |
|---|---|---:|---|---|
| `nemotron` | DFlash | 8,192 | Yes | Incremental chat; buffered tool turns |
| `gemma4-12b` | Official QAT MTP, two proposals | 1,024 | No | Buffered SSE with keepalives |

The 12B route uses the Heretic target and the original Unified QAT assistant.
It supports text-only `/v1/chat/completions`, greedy single-output generation,
and reserves two context slots for verification. `max_tokens` must fit the
remaining context; its default is the smaller of 128 or the available budget.
Oversized inputs and unsupported tools, stop strings, penalties, logprobs,
response constraints and debug settings return HTTP 400. Unknown model IDs
return HTTP 404. This HTTP adapter uses the bounded direct-VM prototype;
it does not add an `MLCEngine` MTP speculative mode. See
[the MTP build and validation notes](../../docs/deploy/xavier_gemma4_mtp.md).

## Access from clients

- On the LAN, the router resolves `cerebellum.lan` to `192.168.12.115`.
- On the existing Headscale tailnet, MagicDNS's extra record resolves the same
  name to `100.64.0.4`. Enable Tailscale DNS on the client. Cerebellum itself
  retains LAN DNS with `--accept-dns=false`.
- Clients must trust `nowhere:~/LocalAuthority/local-ca.crt`, or pass that file
  as their CA bundle. The existing CA permits `.lan`, LAN addresses and
  loopback; it excludes the tailnet suffix and Tailscale IP SANs. Always use
  `cerebellum.lan` as the HTTPS hostname over the tailnet.
- The API key is stored with mode 0600 at
  `nowhere:~/.config/cerebellum/api-key` and
  `cerebellum:~/.config/mlc-nemotron/api-key`. Copy it securely to authorized
  clients; keep it out of source control.

On a client with the root certificate and key file copied locally:

```bash
export CEREBELLUM_API_KEY="$(cat ~/.config/cerebellum/api-key)"
curl --cacert ./local-ca.crt https://cerebellum.lan/v1/models \
  -H "Authorization: Bearer $CEREBELLUM_API_KEY"
curl -N --cacert ./local-ca.crt https://cerebellum.lan/v1/chat/completions \
  -H "Authorization: Bearer $CEREBELLUM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron","messages":[{"role":"user","content":"Say hello."}],"max_tokens":128,"stream":true}'
```

For an SDK or application, set its base URL to `https://cerebellum.lan/v1`,
its model to `nemotron` or `gemma4-12b`, and its API key to the saved key. Configure its TLS
trust store with the local CA. An explicit tailnet connectivity test can use
`curl --resolve cerebellum.lan:443:100.64.0.4` with the same CA and credentials.

## Installed services

`mlc-nemotron.service` runs as `zero` from the validated experimental checkout
at `/home/zero/Workspaces/mlc-dflash-work`. It uses that checkout's Python
code, runtime, converted weights and compiled model libraries. The API listens
on `127.0.0.1:8000`. Nginx terminates TLS on port 443, permits LAN/loopback and
tailnet clients, and passes streaming responses without buffering. Public IPv6
clients are denied. No router port forwarding or Tailscale Funnel is configured.

The model configuration is the measured Nemotron target, F16 DFlash draft,
two proposals (`spec_draft_length=3`), `hybrid` prefill, one active sequence,
8K context and prefix caching disabled. Multiple HTTP clients can queue
requests. The service restarts on failure, with a bounded restart rate.
Nginx, MLC and tailscaled are enabled at boot.

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
sudo install -m 644 deploy/cerebellum/dual-model.conf \
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

### Pi on nowhere

`~/.pi/agent/models.json` registers `cerebellum/nemotron:e4b`, reads the API key
from the private credential file, and declares the 8192-token context and
1024-token output limit. `~/.pi/agent/bin/pi-cerebellum` selects it and loads
the system trust store through `NODE_USE_SYSTEM_CA=1` before Node starts:

```bash
pi-cerebellum
# Or select the provider directly:
NODE_USE_SYSTEM_CA=1 \
  pi --provider cerebellum --model nemotron:e4b --thinking off
```

On nowhere, the root is installed at
`/etc/ca-certificates/trust-source/anchors/local-ca.crt` and extracted by
`sudo update-ca-trust`. `/etc/environment` and
`/etc/profile.d/local-node-system-ca.sh` enable Node system-store trust for new
login sessions. Existing shells can run `export NODE_USE_SYSTEM_CA=1`. Some
applications maintain independent trust stores and need their own configuration.

Pi uses `openai-completions`, bearer authentication, `max_tokens`, system-role
messages and greedy sampling. Its per-model compaction settings reserve 1536
tokens and keep 2048 recent tokens. The default provider is `cerebellum` and the default client model is
`nemotron:e4b`. The `pi-llama-cpp` package remains installed with
`extensions: []`, suppressing its independent localhost:9931 startup probe;
re-enable it through `pi config` when using that integration. The ordinary Pi
launcher defaults `NODE_USE_SYSTEM_CA=1` for existing terminal environments.
Tools are enabled: the launcher has no `--no-tools` flag and the model's
`samplingParams` contains `temperature: 0, top_p: 1` without `tool_choice: none`.
Start a new Pi process after changing its model configuration.

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
Ordinary chat without tools retains MLC's existing streaming path. The 8K
context and greedy-only restriction still apply.

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
ssh cerebellum 'sudo systemctl status mlc-nemotron nginx tailscaled'
ssh cerebellum 'sudo journalctl -u mlc-nemotron -f'
ssh cerebellum 'sudo systemctl restart mlc-nemotron'
```

Templates are included beside this file. If moving the deployment, update the
checkout, venv and credential paths in the systemd unit. Keep compilation and
benchmarks serialized with the serving process; restore the service afterward.

## Certificates and DNS

The leaf is signed by `~/LocalAuthority/local-ca.crt` on nowhere. Its SANs are
`DNS:cerebellum.lan` and `IP:192.168.12.115`. The initial leaf expires on
2027-11-01. The CA private key stays on nowhere; only the leaf key is installed
on cerebellum, root-owned at `/etc/nginx/tls/cerebellum.key` (0600).

Renew on nowhere using the saved CSR and extensions; OpenSSL prompts for the
CA key passphrase:

```bash
cd ~/LocalAuthority
openssl x509 -req -in crts/cerebellum.csr \
  -CA local-ca.crt -CAkey local-ca.key -CAcreateserial \
  -out crts/cerebellum.crt -days 365 -sha256 -extfile exts/cerebellum.ext
scp crts/cerebellum.crt cerebellum:.config/mlc-nemotron/cerebellum.crt
ssh cerebellum 'sudo install -m 644 ~/.config/mlc-nemotron/cerebellum.crt /etc/nginx/tls/cerebellum.crt && sudo nginx -t && sudo systemctl reload nginx'
```

The tailnet record lives in
`convolution:/etc/headscale/extra-records.json`; Headscale watches that file.
Its config references the file through `dns.extra_records_path`. The preceding
configuration is backed up at
`/etc/headscale/config.yaml.before-cerebellum-https`. The node was enrolled
under user `zero` with a single-use key; enrollment key files were removed.

## Verification

Verified from nowhere: CA/hostname validation, LAN IP SAN validation,
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
under `nowhere:~/Workspaces/cerebellum-llm-bench/mlc-jp5/logs/dual-resident-20261101`;
see the [saved summary](../../docs/benchmarks/xavier/dual-resident-summary.json).
