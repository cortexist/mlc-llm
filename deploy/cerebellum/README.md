# Nemotron HTTPS service on cerebellum

The deployed OpenAI-compatible API is **`https://cerebellum.lan/v1`**, with
model ID **`nemotron`**. `/healthz` reports readiness. Model listing and
generation require `Authorization: Bearer <API key>`.

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
its model to `nemotron`, and its API key to the saved key. Configure its TLS
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
