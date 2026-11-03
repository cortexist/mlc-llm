"""Small authored evidence-gate trial and warm resident switching measurement.

This is a development smoke test, not a held-out quality benchmark.
"""

import argparse
import http.client
import json
import ssl
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

p = argparse.ArgumentParser()
p.add_argument("--endpoint", required=True)
p.add_argument("--key-file", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
u = urlparse(a.endpoint)
if u.scheme != "https":
    raise ValueError("Use the authenticated HTTPS endpoint")
conn = http.client.HTTPSConnection(
    u.hostname, u.port or 443, context=ssl.create_default_context(), timeout=120
)
headers = {
    "Authorization": "Bearer " + a.key_file.read_text().strip(),
    "Content-Type": "application/json",
}
instruction = "Does the evidence explicitly contain the information needed to answer the question?"
cases = [
    ("Which port does the API use?", "The API listens on port 8443.", True),
    ("Which port does the API use?", "The API uses HTTPS. Its port is not recorded.", False),
    (
        "How do I restart the service?",
        "Run systemctl restart example-api to restart the service.",
        True,
    ),
    ("How do I restart the service?", "The service is currently active.", False),
    (
        "What enables Node system CA trust?",
        "Set NODE_USE_SYSTEM_CA=1 to enable Node system CA trust.",
        True,
    ),
    (
        "What enables Node system CA trust?",
        "The root certificate is installed. Node settings are not documented.",
        False,
    ),
    ("How much RAM is required?", "This deployment requires at least 16 GiB of RAM.", True),
    ("How much RAM is required?", "This deployment uses CUDA 11.4.", False),
    ("Where is the model stored?", "The model weights are stored in /opt/models/example.", True),
    ("Where is the model stored?", "The weights were downloaded yesterday.", False),
    (
        "What is the maximum input length?",
        "Requests must contain at most 256 tokens including formatting.",
        True,
    ),
    (
        "What is the maximum input length?",
        "Oversized requests are rejected, but the limit is not recorded.",
        False,
    ),
]


def call(path, payload):
    start = time.perf_counter()
    conn.request("POST", path, json.dumps(payload), headers)
    response = conn.getresponse()
    body = json.loads(response.read())
    elapsed = 1000 * (time.perf_counter() - start)
    if response.status != 200:
        raise RuntimeError((response.status, body))
    return body, {k.lower(): v for k, v in response.getheaders()}, elapsed


def decision(state):
    return call(
        "/v1/decisions",
        dict(
            model="d1-omni-2b",
            state=state,
            questions={"enough": dict(type="noul", instructions=instruction)},
        ),
    )


state = f"Question: {cases[0][0]}\nEvidence: {cases[0][1]}"
decision(state)
repeated = [decision(state)[2] for _ in range(4)]
rows = []
instances = set()
for question, evidence, label in cases:
    state = f"Question: {question}\nEvidence: {evidence}"
    nemo, _, nemo_ms = call(
        "/v1/chat/completions",
        dict(
            model="nemotron",
            temperature=0,
            max_tokens=16,
            messages=[dict(role="user", content=f"{state}\n{instruction}\nAnswer only yes or no.")],
        ),
    )
    d1, h, d1_ms = decision(state)
    probability = d1["answers"]["enough"]["noul"]
    answer = nemo["choices"][0]["message"]["content"].strip().lower().rstrip(".! ")
    instances.add(h["x-xavier-decision-instance"])
    rows.append(
        dict(
            question=question,
            evidence=evidence,
            expected=label,
            d1_probability=probability,
            d1_correct=(probability >= 0.5) == label,
            nemo_answer=answer,
            nemo_correct=answer == ("yes" if label else "no"),
            d1_http_ms=d1_ms,
            nemo_http_ms=nemo_ms,
            forward_ms=float(h["x-xavier-decision-inference-ms"]),
            queue_ms=float(h["x-xavier-queue-wait-ms"]),
        )
    )
    print(json.dumps(rows[-1]), flush=True)
assert len(instances) == 1
summary = dict(
    samples=len(rows),
    d1_correct=sum(r["d1_correct"] for r in rows),
    nemo_correct=sum(r["nemo_correct"] for r in rows),
    d1_brier=sum((r["d1_probability"] - r["expected"]) ** 2 for r in rows) / len(rows),
    repeated_http_median_ms=statistics.median(repeated),
    alternating_http_median_ms=statistics.median(r["d1_http_ms"] for r in rows),
    nemo_http_median_ms=statistics.median(r["nemo_http_ms"] for r in rows),
    resident_instances=len(instances),
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(
    json.dumps(dict(summary=summary, repeated_http_ms=repeated, rows=rows), indent=2) + "\n"
)
print(json.dumps(summary), flush=True)
