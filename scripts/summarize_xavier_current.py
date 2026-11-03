"""Summarize the current fixed-token Xavier matrix without publishing host paths."""

import argparse
import datetime
import hashlib
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("logs", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--date", help="Measurement date YYYY-MM-DD; otherwise read from log directory")
a = p.parse_args()
names = [
    "nemotron-mlc-none",
    "nemotron-mlc-dflash",
    "nemotron-llama-none",
    "nemotron-llama-dflash",
    "gemma4-mlc-mtp",
    "gemma4-llama-none",
    "gemma4-llama-mtp",
]
raw = {n: json.loads((a.logs / (n + ".json")).read_text()) for n in names}
rows = {n: d["runs"] for n, d in raw.items()}
date_match = re.search(r"current-matrix-(\d{8})", a.logs.name)
if a.date:
    measured_date = datetime.date.fromisoformat(a.date)
elif date_match:
    measured_date = datetime.datetime.strptime(date_match[1], "%Y%m%d").date()
else:
    p.error("--date is required when the log directory has no measurement date")
for name, value in raw.items():
    expected_context = 16384 if name.startswith("nemotron") else 4096
    assert value["args"]["context"] == expected_context
    assert value["args"]["tokens"] == 128
    assert value["args"]["warmups"] == 1 and value["args"]["repeats"] == 2
summary = dict(
    date=measured_date.isoformat(),
    device="Jetson AGX Xavier 32GB, JetPack 5.1.8, CUDA 11.4, sm_72, MAXN",
    protocol=dict(
        prompt_lengths=[161, 448, 910],
        output_tokens=128,
        ignore_eos=True,
        warmups_per_case_per_mode=1,
        measured_repeats=2,
        prefix_cache=False,
        prefill_chunk=512,
        cpu_threads=2,
        kv_dtype="float16",
        speculative_proposals=2,
        context=dict(nemotron=16384, gemma4=4096),
        prompt_format=(
            "Source HF chat template, enable_thinking=False; body tokens truncated/padded "
            "to exact length; identical input token IDs within each engine pair."
        ),
        prefill_column="Plain-mode prefill; averages of two measured requests.",
        decode_rate=(
            "Engine-reported decode throughput; native 12B controller uses "
            "(completion_tokens-1)/(end-first_token)."
        ),
        gemma4_mlc=(
            "Current direct-VM native MTP controller; CPU shared-state copies included in "
            "prefill and both plain/speculative decode."
        ),
        targets=(
            "Both engines use the same source checkpoint per row: NVIDIA Nemotron 3.5 "
            "Bolt E4B BF16, or Gemma4 Unified 12B Heretic BF16. MLC q4f16_1 versus "
            "llama.cpp Q4_K_M. Drafts F16."
        ),
    ),
    models={},
    runs={},
)


def selected(name, case, width=None):
    r = [
        x
        for x in rows[name]
        if x["case"] == case and (width is None or x.get("draft_tokens") == width)
    ]
    assert len(r) == 2, (name, case, width, len(r))
    assert {x["repeat"] for x in r} == {0, 1}
    assert all(x["completion_tokens"] == 128 and x["prompt_tokens"] == int(case) for x in r)
    return r


def mean(r, key):
    return statistics.mean(x[key] for x in r)


for model in ["nemotron", "gemma4"]:
    table = []
    for case in ["161", "448", "910"]:
        plain = selected(
            model + "-mlc-" + ("none" if model == "nemotron" else "mtp"),
            case,
            None if model == "nemotron" else 0,
        )
        spec = selected(
            model + "-mlc-" + ("dflash" if model == "nemotron" else "mtp"),
            case,
            None if model == "nemotron" else 2,
        )
        lp = selected(model + "-llama-none", case)
        ls = selected(model + "-llama-" + ("dflash" if model == "nemotron" else "mtp"), case)
        table.append(
            dict(
                prompt_tokens=int(case),
                prefill=mean(plain, "prefill_tok_s"),
                llama_prefill=mean(lp, "prefill_tok_s"),
                decode_plain=mean(plain, "decode_tok_s"),
                llama_decode_plain=mean(lp, "decode_tok_s"),
                decode_speculative=mean(spec, "decode_tok_s"),
                llama_decode_speculative=mean(ls, "decode_tok_s"),
            )
        )
    summary["models"][model] = dict(
        table=table,
        speculation="DFlash" if model == "nemotron" else "native MTP",
        prompt_sha256={
            r["case"]: r["sha256"]
            for r in json.loads((a.logs / (model + "-prompts.json")).read_text())
        },
    )
for name, rr in rows.items():
    cleaned = []
    for r in rr:
        keys = [
            "case",
            "repeat",
            "prompt_tokens",
            "completion_tokens",
            "prefill_tok_s",
            "decode_tok_s",
            "elapsed_s",
            "draft_tokens",
            "accepted",
            "proposed",
            "rounds",
            "matches_plain",
        ]
        out = {k: r[k] for k in keys if k in r}
        if "response" in r and "timings" in r["response"]:
            out["timings"] = r["response"]["timings"]
        output = r.get("tokens", r.get("text"))
        out["output_sha256"] = hashlib.sha256(
            json.dumps(output, ensure_ascii=False).encode()
        ).hexdigest()
        cleaned.append(out)
    summary["runs"][name] = cleaned
    if name == "gemma4-mlc-mtp":
        summary["models"]["gemma4"]["all_mlc_speculative_outputs_match_plain"] = all(
            r["matches_plain"] for r in rr if r["draft_tokens"] == 2
        )
    elif name.endswith("mlc-dflash"):
        plain = rows["nemotron-mlc-none"]
        summary["models"]["nemotron"]["all_mlc_speculative_outputs_match_plain"] = all(
            r["text"]
            == next(
                x["text"] for x in plain if (x["case"], x["repeat"]) == (r["case"], r["repeat"])
            )
            for r in rr
        )
for model, mode in [("nemotron", "dflash"), ("gemma4", "mtp")]:
    plain = rows[model + "-llama-none"]
    speculative = rows[model + "-llama-" + mode]
    matches = []
    for r in speculative:
        control = next(x for x in plain if (x["case"], x["repeat"]) == (r["case"], r["repeat"]))
        matches.append(
            dict(case=r["case"], repeat=r["repeat"], identical=r["text"] == control["text"])
        )
    summary["models"][model]["llama_plain_speculative_output_checks"] = matches
    summary["models"][model]["all_llama_speculative_outputs_match_plain"] = all(
        x["identical"] for x in matches
    )
summary["output_qualification"] = (
    "Some forced plain/speculative continuations differ. Output checks are recorded; "
    "throughput does not establish speculative equivalence or broad model quality."
)
if (a.logs / "artifact-sha256.txt").exists():
    summary["artifact_sha256"] = {
        Path(name).name: h
        for h, name in (
            line.split(maxsplit=1)
            for line in (a.logs / "artifact-sha256.txt").read_text().splitlines()
        )
    }
if (a.logs / "llama-revision.txt").exists():
    summary["llama_revision"] = (a.logs / "llama-revision.txt").read_text().strip()
summary["protocol"]["fixed_clocks_mhz"] = dict(gpu=1377, cpu=2265.6, emc=2133)
a.output.write_text(json.dumps(summary, indent=2) + "\n")
for name, data in summary["models"].items():
    print(name)
    for r in data["table"]:
        print(
            "| "
            + str(r["prompt_tokens"])
            + " | "
            + " | ".join(
                f"{r[k]:.2f}"
                for k in [
                    "prefill",
                    "llama_prefill",
                    "decode_plain",
                    "llama_decode_plain",
                    "decode_speculative",
                    "llama_decode_speculative",
                ]
            )
            + " |"
        )
