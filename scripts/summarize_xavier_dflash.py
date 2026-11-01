"""Summarize per-request timing and speculative acceptance from saved runs."""

import json
import sys
from pathlib import Path


def phase_ms(before, after, key):
    old, new = before.get(key, {}), after.get(key, {})
    count = total = 0
    for label, value in new.items():
        if label.startswith("count{"):
            mean = label.replace("count{", "mean{", 1)
            count += value - old.get(label, 0)
            total += value * new[mean] - old.get(label, 0) * old.get(mean, 0)
    return 1000 * total / count if count else None


for filename in sys.argv[1:]:
    data = json.loads(Path(filename).read_text())
    for run in data["runs"]:
        before, after = run["metrics_before"], run["metrics_after"]
        usage = run["usage"]
        row = dict(
            file=str(filename),
            case=run["case"],
            repeat=run["repeat"],
            prompt_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            prefill_tok_s=usage["extra"].get("prefill_tokens_per_s"),
            decode_tok_s=usage["extra"].get("decode_tokens_per_s"),
            ttft_s=run["ttft_s"],
            elapsed_s=run["elapsed_s"],
        )
        if "token_limit" in run:
            row["token_limit"] = run["token_limit"]
        if after.get("spec_decode", {}).get("draft_count"):
            a, b = before.get("spec_decode", {}), after["spec_decode"]
            accepted = [
                v - a.get("accept_count", [0] * len(b["accept_count"]))[i]
                for i, v in enumerate(b["accept_count"])
            ]
            rounds = b["draft_count"][0] - a.get("draft_count", [0])[0]
            row.update(
                rounds=rounds,
                accepted_counts=accepted,
                outputs_per_round=sum(accepted) / rounds if rounds else None,
                draft_ms=phase_ms(before, after, "draft_time_by_batch_size"),
                verify_ms=phase_ms(before, after, "verify_time_by_batch_size"),
            )
        print(json.dumps(row))
