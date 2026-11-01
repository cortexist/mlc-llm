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

`~/.pi/agent/models.json` registers `cerebellum/nemotron`, reads the API key
from the private credential file, and declares the 8192-token context and
1024-token output limit. `~/.pi/agent/bin/pi-cerebellum` selects it and loads
the local CA through `NODE_EXTRA_CA_CERTS` before Node starts:

```bash
pi-cerebellum
# Or select the provider directly:
NODE_EXTRA_CA_CERTS="$HOME/LocalAuthority/local-ca.crt" \
  pi --provider cerebellum --model nemotron --thinking off --no-tools
```

Pi uses `openai-completions`, bearer authentication, `max_tokens`, system-role
messages and greedy sampling. Its per-model compaction settings reserve 1536
tokens and keep 2048 recent tokens. Existing provider/default settings are
preserved. The current MLC protocol rejects assistant `tool_calls` messages,
so this launcher runs Pi in chat mode with `--no-tools`. Coding-agent tool
round-trips need additional backend support. A Pi print-mode request returned
`391` through this configuration with TLS verification enabled.

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
