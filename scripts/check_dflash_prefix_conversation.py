"""Replay exported OpenAI tool-chat payloads to check long DFlash prefix reuse.

Payloads may contain private instructions or documents; keep fixtures outside
public repositories. Run each cache mode sequentially, with the service stopped.
"""

import argparse
import json
import time
from pathlib import Path

from gemma4_tools import normalize_messages
from transformers import AutoTokenizer

from mlc_llm import MLCEngine
from mlc_llm.op.xavier_prefill import install
from mlc_llm.serve.config import EngineConfig

parser = argparse.ArgumentParser()
parser.add_argument("--payloads", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference", type=Path)
parser.add_argument("--cache", choices=["disable", "radix"], required=True)
parser.add_argument("--kernels", choices=["baseline", "hybrid-width"], default="hybrid-width")
parser.add_argument("--block", type=int, choices=[1, 3], default=3)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(
    Path.home() / "Models/nemotron-3.5-Bolt-E4B-BF16", local_files_only=True
)
payloads = json.loads(args.payloads.read_text())
assert isinstance(payloads, list) and len(payloads) >= 2
prompts = [
    tokenizer.apply_chat_template(
        normalize_messages(x["messages"]),
        tools=x["tools"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    for x in payloads
]
print("Lengths", list(map(len, prompts)), flush=True)
if args.kernels != "baseline":
    install(args.kernels, Path("dist/target-16k/mlc-chat-config.json"))
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
        spec_draft_length=args.block,
        prefix_cache_mode=args.cache,
        prefix_cache_max_num_recycling_seqs=1,
    ),
)
rows = []
try:
    for i, tokens in enumerate(prompts):
        start = time.perf_counter()
        response = engine.completions.create(
            prompt=tokens, temperature=0, max_tokens=8 if i == 0 else 160
        ).model_dump()
        row = {"elapsed_s": time.perf_counter() - start, "response": response}
        rows.append(row)
        print(json.dumps(row), flush=True)
finally:
    engine.terminate()
args.output.write_text(json.dumps(rows, indent=2) + "\n")
if args.reference:
    reference = json.loads(args.reference.read_text())
    assert len(reference) == len(rows)
    for expected, actual in zip(reference, rows):
        assert (
            expected["response"]["usage"]["prompt_tokens"]
            == actual["response"]["usage"]["prompt_tokens"]
        )
        assert (
            expected["response"]["choices"][0]["text"] == actual["response"]["choices"][0]["text"]
        )
