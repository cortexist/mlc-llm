"""Compare repeated and alternating warm requests; no model-loading time is included."""

import argparse
import http.client
import json
import pathlib
import ssl
import statistics
import time
import urllib.parse

p = argparse.ArgumentParser()
p.add_argument("--endpoint", default="https://xavier.example.com")
p.add_argument(
    "--key-file", type=pathlib.Path, default=pathlib.Path.home() / ".config/xavier/api-key"
)
p.add_argument("--repeats", type=int, default=8)
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
u = urllib.parse.urlparse(a.endpoint)
conn = http.client.HTTPSConnection(
    u.hostname, u.port or 443, context=ssl.create_default_context(), timeout=60
)
headers = {
    "Authorization": "Bearer " + a.key_file.read_text().strip(),
    "Content-Type": "application/json",
}
rows = []
instances = set()
connections = set()


def request(method, path, payload=None):
    t = time.perf_counter_ns()
    conn.request(method, path, None if payload is None else json.dumps(payload), headers)
    connections.add(conn.sock.getsockname())
    r = conn.getresponse()
    raw = r.read()
    elapsed = (time.perf_counter_ns() - t) / 1e6
    h = dict(r.getheaders())
    d = json.loads(raw)
    assert r.status == 200, (r.status, d)
    return d, h, elapsed


def timing_header(headers, suffix):
    # Accept older deployments with a different vendor prefix as well.
    matches = [v for k, v in headers.items() if k.lower().endswith("-" + suffix)]
    assert len(matches) == 1, f"Missing or ambiguous timing header: {suffix}"
    return matches[0]


def run(model, phase):
    if model == "nemotron":
        d, h, elapsed = request(
            "POST",
            "/v1/chat/completions",
            dict(
                model="nemotron",
                messages=[
                    dict(role="user", content="What is 17 multiplied by 23? Give only the number.")
                ],
                temperature=0,
                max_tokens=16,
            ),
        )
        assert "391" in d["choices"][0]["message"]["content"]
    else:
        d, h, elapsed = request(
            "POST",
            "/v1/embeddings",
            dict(
                model="embeddinggemma-2",
                input="task: search result | query: how do I reset my password?",
            ),
        )
        assert len(d["data"][0]["embedding"]) == 768
        instances.add(timing_header(h, "embedding-instance"))
    row = dict(
        model=model, phase=phase, http_ms=elapsed, queue_ms=float(timing_header(h, "queue-wait-ms"))
    )
    if model != "nemotron":
        row.update(
            forward_ms=float(timing_header(h, "embedding-inference-ms")),
            worker_ms=float(timing_header(h, "embedding-total-ms")),
        )
    if phase != "warmup":
        rows.append(row)
    return row


before = request("GET", "/healthz")[0]
assert before["models"] == ["nemotron", "embeddinggemma-2"], before
for _ in range(2):
    for m in ["nemotron", "embeddinggemma-2"]:
        run(m, "warmup")
for order in [("nemotron", "embeddinggemma-2"), ("embeddinggemma-2", "nemotron")]:
    for m in order:
        for _ in range(a.repeats):
            run(m, "repeated")
    for _ in range(a.repeats):
        for m in order:
            run(m, "alternating")
after = request("GET", "/healthz")[0]
assert len(instances) == 1, instances
assert len(connections) == 1, connections
summary = {}
for m in ["nemotron", "embeddinggemma-2"]:
    summary[m] = {}
    for phase in ["repeated", "alternating"]:
        subset = [r for r in rows if r["model"] == m and r["phase"] == phase]
        s = {}
        for metric in ["http_ms", "forward_ms", "worker_ms", "queue_ms"]:
            vals = sorted(r[metric] for r in subset if metric in r)
            if vals:
                s[metric] = dict(
                    median=statistics.median(vals),
                    min=min(vals),
                    max=max(vals),
                    p10=vals[int(0.1 * (len(vals) - 1))],
                    p90=vals[int(0.9 * (len(vals) - 1))],
                )
        summary[m][phase] = dict(samples=len(subset), **s)
    summary[m]["alternating_minus_repeated_http_ms"] = (
        summary[m]["alternating"]["http_ms"]["median"] - summary[m]["repeated"]["http_ms"]["median"]
    )
report = dict(
    method="One persistent HTTPS connection; matched inputs; two warmups per model; repeated blocks and alternating pairs in both orders. Measures warm request penalty, not isolated CUDA context-switch or preemption latency.",
    summary=summary,
    embedding_instances=list(instances),
    connection_count=len(connections),
    health_before=before,
    health_after=after,
    rows=rows,
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
conn.close()
