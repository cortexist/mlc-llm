"""Compare a captured real-checkpoint MLC assistant step with Transformers CPU."""

import argparse
import json

import numpy as np
import torch
from transformers import Gemma4UnifiedAssistantForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    arrays = np.load(args.capture)
    reference = Gemma4UnifiedAssistantForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).eval()

    def tensor(name):
        return torch.from_numpy(arrays[name])

    shared = {
        kind: tuple(tensor(prefix + suffix).permute(0, 2, 1, 3) for suffix in ("key", "value"))
        for kind, prefix in (("sliding_attention", "local_"), ("full_attention", "global_"))
    }
    with torch.no_grad():
        expected = reference(
            inputs_embeds=torch.cat([tensor("embedding"), tensor("previous_hidden")], dim=-1),
            position_ids=tensor("position")[None, :].long(),
            shared_kv_states=shared,
            use_cache=False,
        )
    report = {}
    for name, value in (("logits", expected.logits), ("hidden", expected.last_hidden_state)):
        actual = arrays[name].astype("float32").flatten()
        expected_array = value.float().numpy().flatten()
        report[name] = dict(
            max_abs_error=float(np.max(np.abs(actual - expected_array))),
            mean_abs_error=float(np.mean(np.abs(actual - expected_array))),
            cosine=float(
                np.dot(actual, expected_array)
                / (np.linalg.norm(actual) * np.linalg.norm(expected_array))
            ),
        )
        if name == "logits":
            report[name].update(
                mlc_argmax=int(actual.argmax()), hf_argmax=int(expected_array.argmax())
            )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    if any(value["cosine"] < 0.999 for value in report.values()):
        raise AssertionError("Assistant reference cosine fell below 0.999")


if __name__ == "__main__":
    main()
