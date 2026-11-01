"""Controlled plain/DFlash comparisons using the existing brain1 prompt corpus."""

import argparse
import ast
import ctypes
import json
import time
from pathlib import Path

import mlc_llm
from mlc_llm import MLCEngine
from mlc_llm.base import _LIB_PATH
from mlc_llm.serve.config import EngineConfig

parser = argparse.ArgumentParser()
parser.add_argument("--draft-tokens", type=int, default=2, help="Zero disables speculation")
parser.add_argument("--target-lib", default="dist/libs/target-dflash-graph-b3-sm72.so")
parser.add_argument("--draft-lib", default="dist/libs/draft-cached-tc-sm72.so")
parser.add_argument("--tokens", type=int, default=128)
parser.add_argument(
    "--limits", help="Comma-separated token limits for boundary checks in one engine"
)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--trace", action="store_true", help="CUPTI diagnostic, not a timing run")
parser.add_argument("--cases", default="chat,448,1768")
parser.add_argument("--allow-eos", action="store_true")
parser.add_argument("--output", required=True)
parser.add_argument(
    "--prefill-variant",
    choices=["baseline", "bounds", "width", "wmma", "hybrid"],
    default="baseline",
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
assert Path(mlc_llm.__file__).resolve().is_relative_to(root / "python")
assert Path(_LIB_PATH).resolve().is_relative_to(root / "build"), _LIB_PATH
constants = {}
corpus = root.parent / "mlc-jp5/mlc_bench.py"
for node in ast.parse(corpus.read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ("DESIGN_NOTE", "FILLER"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
prompts = {
    "chat": constants["DESIGN_NOTE"],
    "448": "Read this and reply with one word.\n" + constants["FILLER"] * 13,
    "1768": "Read this and reply with one word.\n" + constants["FILLER"] * 53,
    "short": "What is 17 multiplied by 23? Give only the number.",
}
config = EngineConfig(
    prefix_cache_mode="disable",
    prefill_chunk_size=512,
    max_num_sequence=1,
    max_single_sequence_length=8192,
    max_total_sequence_length=8192,
    speculative_mode="dflash" if args.draft_tokens else "disable",
    spec_draft_length=args.draft_tokens + 1 if args.draft_tokens else 0,
    additional_models=[(str(root / "dist/draft"), str(root / args.draft_lib))]
    if args.draft_tokens
    else [],
)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
results = dict(args=vars(args), runtime=_LIB_PATH, runs=[])
trace = None
if args.trace:
    trace = ctypes.CDLL(str(root.parent / "mlc-jp5/cupti_summary.so"))
    trace.trace_start.restype = ctypes.c_int
    trace.trace_stop.argtypes = [ctypes.c_char_p]
    trace.trace_stop.restype = ctypes.c_int
    trace.trace_finish.restype = None
    assert trace.trace_start() == 0
print("Loading experimental runtime:", _LIB_PATH, flush=True)
prefill_receipt = None
if args.prefill_variant != "baseline":
    from mlc_llm.op.xavier_prefill import install

    prefill_receipt = install(args.prefill_variant)
engine = MLCEngine(
    str(root / "dist/target"),
    model_lib=str(root / args.target_lib),
    device="cuda",
    mode="interactive",
    engine_config=config,
)
if prefill_receipt is not None:
    assert prefill_receipt["target_caches"] == 1, prefill_receipt


def run(prompt, label=None, token_limit=None):
    token_limit = args.tokens if token_limit is None else token_limit
    before = engine.metrics().metrics
    if trace and label:
        assert trace.trace_start() == 0
    start = time.perf_counter()
    first = None
    chunks = []
    usage = None
    for chunk in engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=token_limit,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"debug_config": {"ignore_eos": not args.allow_eos}},
    ):
        if chunk.usage is not None:
            usage = chunk.usage.model_dump()
        for choice in chunk.choices:
            if choice.delta.content:
                if first is None:
                    first = time.perf_counter()
                    if trace and label:
                        assert (
                            trace.trace_stop(str(out.with_name(label + ".prefill.csv")).encode())
                            == 0
                        )
                        assert trace.trace_start() == 0
                chunks.append(choice.delta.content)
    end = time.perf_counter()
    if trace and label:
        assert trace.trace_stop(str(out.with_name(label + ".decode.csv")).encode()) == 0
    assert usage is not None
    assert usage["completion_tokens"] <= token_limit, usage
    if not args.allow_eos:
        assert usage["completion_tokens"] == token_limit, usage
        assert usage["extra"]["decode_tokens"] == token_limit - 1, usage
    return dict(
        text="".join(chunks),
        usage=usage,
        elapsed_s=end - start,
        ttft_s=(first or end) - start,
        wall_decode_tok_s=(usage["completion_tokens"] - 1) / (end - first) if first else 0,
        metrics_before=before,
        metrics_after=engine.metrics().metrics,
    )


try:
    for case in args.cases.split(","):
        for limit in map(int, args.limits.split(",")) if args.limits else [args.tokens]:
            for _ in range(args.warmups):
                run(prompts[case], token_limit=limit)
            for repeat in range(args.repeats):
                result = dict(
                    case=case,
                    repeat=repeat,
                    token_limit=limit,
                    **run(prompts[case], f"{out.stem}.{case}.{limit}.{repeat}", limit),
                )
                results["runs"].append(result)
                out.write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ["metrics_before", "metrics_after"]
                        }
                    ),
                    flush=True,
                )
finally:
    if trace:
        trace.trace_finish()
    engine.terminate()
