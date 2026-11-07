"""Reproduce the short Nemotron head comparison with the GPU service stopped."""

import argparse
import json
import time
from pathlib import Path

from mlc_llm.op.xavier_prefill import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["mtp", "plain", "dflash"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prompts = json.loads(
        (root / "docs/benchmarks/xavier/nemotron-gemma-mtp-summary.json").read_text()
    )["prompt_token_ids"]
    install("hybrid-width", root / "dist/target-16k/mlc-chat-config.json")
    result = {
        "args": {"mode": args.mode, "tokens": args.tokens, "repeats": args.repeats},
        "runs": [],
    }

    def save(row):
        result["runs"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ["tokens", "accepted_by_round", "text"]}
            ),
            flush=True,
        )

    if args.mode == "mtp":
        from bench_xavier_e4b_mtp import Program, Trial

        trial = Trial(
            Program(
                root / "dist/nemotron-e4b-mtp-1k",
                root / "dist/libs/nemotron-e4b-mtp-target-1k-sm72.so",
            ),
            Program(
                Path.home() / "Models/gemma-4-E4B-IT-assistant-q0f16-MLC",
                root / "dist/libs/gemma4-e4b-mtp-assistant-topk-1k-sm72.so",
            ),
            root / "dist/libs/xavier-mtp-runtime.so",
        )
        for case in prompts:
            for repeat in range(-1, args.repeats):
                for width in [0, 1, 2]:
                    row = trial.run(case["tokens"], args.tokens, width, ignore_eos=True)
                    row.update(
                        case=case["case"], repeat=repeat, warmup=repeat < 0, draft_tokens=width
                    )
                    save(row)
    else:
        from mlc_llm import MLCEngine
        from mlc_llm.serve.config import EngineConfig

        draft = args.mode == "dflash"
        engine = MLCEngine(
            str(root / "dist/target-16k"),
            model_lib=str(root / "dist/libs/target-dflash-vector-dequant-16k-sm72.so"),
            device="cuda",
            mode="interactive",
            engine_config=EngineConfig(
                prefix_cache_mode="disable",
                prefill_chunk_size=512,
                max_num_sequence=1,
                max_single_sequence_length=1024,
                max_total_sequence_length=1024,
                speculative_mode="dflash" if draft else "disable",
                spec_draft_length=3 if draft else 0,
                additional_models=[
                    (
                        str(root / "dist/draft-16k"),
                        str(root / "dist/libs/draft-cached-tc-16k-sm72.so"),
                    )
                ]
                if draft
                else [],
            ),
        )
        try:
            for case in prompts:
                for repeat in range(-1, args.repeats):
                    start = time.perf_counter()
                    response = engine.completions.create(
                        prompt=case["tokens"],
                        temperature=0,
                        max_tokens=args.tokens,
                        extra_body={"debug_config": {"ignore_eos": True}},
                    ).model_dump()
                    usage = response["usage"]
                    save(
                        dict(
                            case=case["case"],
                            repeat=repeat,
                            warmup=repeat < 0,
                            completion_tokens=usage["completion_tokens"],
                            decode_tok_s=usage["extra"]["decode_tokens_per_s"],
                            elapsed_s=time.perf_counter() - start,
                            text=response["choices"][0]["text"],
                        )
                    )
        finally:
            engine.terminate()


if __name__ == "__main__":
    main()
