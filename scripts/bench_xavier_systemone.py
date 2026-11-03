"""Synthetic mail protocol/quality trial. Does not read or change any mailbox."""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--endpoint", required=True, help="HTTPS API root ending in /v1")
p.add_argument("--key-file", required=True, type=Path)
p.add_argument("--cases", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
if not a.endpoint.startswith("https://"):
    raise ValueError("Use HTTPS with system CA trust")
key = a.key_file.read_text().strip()


def call(path, payload, authenticated=True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        a.endpoint.rstrip("/") + path, data=json.dumps(payload).encode(), headers=headers
    )
    start = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return (
            response.status,
            json.load(response),
            dict(response.headers),
            (time.perf_counter() - start) * 1000,
        )


cases = json.loads(a.cases.read_text())
rows = []
for case in cases:
    request = dict(model="d1-omni-2b", state=case["state"], questions=case["questions"])
    status, body, headers, elapsed = call("/systemone", request)
    assert status == 200, (case["name"], status, body)
    assert body["usage"]["output_tokens"] == 0
    assert body["model"] == "d1-omni-2b"
    assert set(body["answers"]) == set(request["questions"])
    observed, correct = {}, {}
    for name, expected in case["expected"].items():
        answer = body["answers"][name]
        assert answer["type"] == "noul"
        probability = answer["noul"]
        assert 0 <= probability <= 1
        upper = case["thresholds"][name]
        lower = case.get("negative_thresholds", {}).get(name, upper)
        verdict = True if probability >= upper else False if probability <= lower else None
        if verdict is None:
            verdict = case.get("fallback", {}).get(name)
        observed[name], correct[name] = verdict, verdict == expected
    timings = {k.lower(): v for k, v in headers.items() if k.lower().startswith("x-xavier-")}
    row = dict(
        name=case["name"],
        response=body,
        http_ms=elapsed,
        timings=timings,
        expected=case["expected"],
        observed=observed,
        correct=correct,
    )
    rows.append(row)
    print(json.dumps(row), flush=True)

# Alias compatibility, choice response, auth, unsupported input and no truncation.
request = dict(model="d1-omni-2b", state=cases[0]["state"], questions=cases[0]["questions"])
status, alias, _, _ = call("/decisions", request)
assert status == 200 and alias == rows[0]["response"]
assert call("/systemone", request, authenticated=False)[0] == 401
assert call("/systemone", {**request, "model": "not-a-model"})[0] == 404
assert call("/systemone", {**request, "state": "network " * 2100})[0] == 400
assert call("/systemone", {**request, "media": []})[0] == 422
assert (
    call(
        "/systemone",
        {
            **request,
            "questions": {
                "q": {"type": "noul", "instructions": "yes?", "criteria": {"true": "yes"}}
            },
        },
    )[0]
    == 400
)
status, choice, _, _ = call(
    "/systemone",
    dict(
        model="d1-omni-2b",
        state={"message": "The CUDA compiler is missing."},
        questions={
            "route": dict(
                type="choice",
                instructions="Choose the issue category.",
                criteria={"build": "Compilation or build tools", "network": "Connectivity or DNS"},
            )
        },
    ),
)
assert status == 200
answer = choice["answers"]["route"]
assert answer["choice"] == "build"
assert answer["confidence"] == max(answer["probabilities"].values())
summary = dict(
    cases=len(rows),
    all_labels_correct=sum(all(r["correct"].values()) for r in rows),
    labels=sum(len(r["correct"]) for r in rows),
    labels_correct=sum(sum(r["correct"].values()) for r in rows),
    http_median_ms=statistics.median(r["http_ms"] for r in rows),
    forward_median_ms=statistics.median(
        float(r["timings"]["x-xavier-decision-inference-ms"]) for r in rows
    ),
    min_input_tokens=min(r["response"]["usage"]["input_tokens"] for r in rows),
    max_input_tokens=max(r["response"]["usage"]["input_tokens"] for r in rows),
    protocol_checks="passed",
    limitation="Authored synthetic development cases, not held-out calibration",
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(summary=summary, rows=rows, choice=choice), indent=2) + "\n")
print(json.dumps(summary), flush=True)
