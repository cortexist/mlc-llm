"""Sequential Xavier DFlash prefix-cache regression and latency probe."""

import argparse
import json
import time
from pathlib import Path

from pydantic import ValidationError

from mlc_llm import MLCEngine
from mlc_llm.op.xavier_prefill import install
from mlc_llm.serve.config import EngineConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", choices=["disable", "radix"], required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Uncached JSON whose outputs must match")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if isinstance(prompts, dict):
        prompts = prompts["prompts"]
    base = next(p["tokens"] for p in prompts if len(p["tokens"]) == 910)
    other = next(p["tokens"] for p in prompts if len(p["tokens"]) == 161)
    install("hybrid-width", Path("dist/target-16k/mlc-chat-config.json"))
    engine = MLCEngine(
        "dist/target-16k",
        device="cuda",
        mode="interactive",
        model_lib="dist/libs/target-dflash-vector-dequant-16k-sm72.so",
        engine_config=EngineConfig(
            additional_models=[("dist/draft-16k", "dist/libs/draft-cached-tc-16k-sm72.so")],
            max_num_sequence=1,
            max_total_sequence_length=16384,
            max_single_sequence_length=16384,
            prefill_chunk_size=512,
            speculative_mode="dflash",
            spec_draft_length=3,
            prefix_cache_mode=args.cache,
            prefix_cache_max_num_recycling_seqs=1,
        ),
    )
    rows = []

    def run(name, tokens):
        start = time.perf_counter()
        result = engine.completions.create(
            prompt=tokens,
            temperature=0,
            max_tokens=24,
            extra_body={"debug_config": {"ignore_eos": True}},
        ).model_dump()
        assert result["usage"]["completion_tokens"] == 24, result
        row = dict(case=name, elapsed_s=time.perf_counter() - start, response=result)
        rows.append(row)
        args.output.write_text(json.dumps(dict(cache=args.cache, rows=rows), indent=2) + "\n")
        print(
            json.dumps(dict(case=name, elapsed_s=row["elapsed_s"], usage=result["usage"])),
            flush=True,
        )

    try:
        run("cold", base)
        run("repeat", base)
        run("append", base + other[-32:])
        run("shorter_fork", base[:448])
        run("unrelated", other)
        run("after_eviction", base)
        # Aborted generations must not leave draft features attached to recycled IDs.
        stream = engine.completions.create(
            prompt=base,
            temperature=0,
            max_tokens=128,
            stream=True,
            request_id="prefix-cancel",
        )
        next(stream)
        engine.abort("prefix-cancel")
        # The sync API shares a callback queue: drain cancellation before the next
        # request. Its OpenAI completion schema currently rejects finish_reason=abort.
        try:
            for _ in stream:
                pass
        except ValidationError as exc:
            assert all(
                e.get("input") == "abort" and "finish_reason" in e["loc"] for e in exc.errors()
            ), exc
        finally:
            stream.close()
        run("after_abort", base)
        engine.reset()
        run("after_reset", base)
        by_case = {row["case"]: row for row in rows}
        assert by_case["after_reset"]["response"]["usage"]["extra"]["prefill_tokens"] == len(base)
        if args.cache == "radix":
            assert by_case["repeat"]["response"]["usage"]["extra"]["prefill_tokens"] == 1
            assert by_case["append"]["response"]["usage"]["extra"]["prefill_tokens"] == 32
            assert by_case["shorter_fork"]["response"]["usage"]["extra"]["prefill_tokens"] == 1
        if args.reference:
            reference = json.loads(args.reference.read_text())["rows"]
            assert [row["case"] for row in reference] == [row["case"] for row in rows]
            for expected, actual in zip(reference, rows):
                assert (
                    expected["response"]["choices"][0]["text"]
                    == actual["response"]["choices"][0]["text"]
                ), actual["case"]
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
