"""Summarize matched prefill experiments, including generation differences."""

import argparse
import json
from pathlib import Path
from statistics import mean

p = argparse.ArgumentParser()
p.add_argument("folder", type=Path)
p.add_argument("--output", required=True, type=Path)
args = p.parse_args()
variants = ["baseline", "bounds", "width", "hybrid"]
data = {v: json.loads((args.folder / f"prefill-{v}.json").read_text()) for v in variants}
base = {(r["case"], r["repeat"]): r for r in data["baseline"]["runs"]}
summary = {}
for variant, content in data.items():
    summary[variant] = {}
    for case in ["chat", "448", "1768"]:
        rows = [r for r in content["runs"] if r["case"] == case]
        assert len(rows) == 2
        assert all(r["usage"]["completion_tokens"] == 128 for r in rows)
        summary[variant][case] = {
            "prompt_tokens": rows[0]["usage"]["prompt_tokens"],
            "prefill_tok_s": mean(r["usage"]["extra"]["prefill_tokens_per_s"] for r in rows),
            "decode_tok_s": mean(r["usage"]["extra"]["decode_tokens_per_s"] for r in rows),
            "ttft_s": mean(r["ttft_s"] for r in rows),
            "wall_s": mean(r["elapsed_s"] for r in rows),
            "identical_output_to_baseline": [
                r["text"] == base[case, r["repeat"]]["text"] for r in rows
            ],
            "repeat_output_identical": rows[0]["text"] == rows[1]["text"],
        }
        summary[variant][case]["prefill_speedup"] = (
            summary[variant][case]["prefill_tok_s"] / summary["baseline"][case]["prefill_tok_s"]
        )
args.output.write_text(json.dumps(summary, indent=2) + "\n")
for variant, cases in summary.items():
    print(
        variant,
        " | ".join(
            f"{case}: pp {v['prefill_tok_s']:.2f}, tg {v['decode_tok_s']:.2f}, "
            f"TTFT {v['ttft_s']:.3f}s, wall {v['wall_s']:.3f}s, "
            f"match {v['identical_output_to_baseline']}"
            for case, v in cases.items()
        ),
    )
