"""GPU regression for a protected service prefix and a two-entry conversation cache."""

import argparse
import json
import re
import time
from pathlib import Path

from pydantic import ValidationError
from transformers import AutoTokenizer
from xavier_service_prompt import prefix_tokens, prepend_policy

from mlc_llm import MLCEngine
from mlc_llm.op.xavier_prefill import install
from mlc_llm.serve.config import EngineConfig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(
    Path.home() / "Models/nemotron-3.5-Bolt-E4B-BF16", local_files_only=True
)
base = prefix_tokens(tokenizer)
install("hybrid-width", Path("dist/target-16k/mlc-chat-config.json"))
engine = MLCEngine(
    "dist/target-16k",
    device="cuda",
    mode="interactive",
    model_lib="dist/libs/target-dflash-vector-dequant-16k-sm72.so",
    engine_config=EngineConfig(
        additional_models=[("dist/draft-16k", "dist/libs/draft-cached-tc-16k-sm72.so")],
        max_num_sequence=1,
        max_total_sequence_length=32768,
        max_single_sequence_length=16384,
        prefill_chunk_size=512,
        speculative_mode="dflash",
        spec_draft_length=3,
        prefix_cache_mode="radix",
        prefix_cache_max_num_recycling_seqs=2,
    ),
)
rows = []


def prompt(client):
    messages = prepend_policy(
        [
            {"role": "system", "content": f"Client {client}. Return exactly the requested code."},
            {"role": "user", "content": f"Return only this code: ORBIT-{client}-4827"},
        ]
    )
    tokens = tokenizer.apply_chat_template(
        messages,
        tools=[],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    assert tokens[: len(base)] == base
    return tokens


def run(name, tokens, protected=True, count=32):
    start = time.perf_counter()
    r = engine.completions.create(
        prompt=tokens,
        max_tokens=count,
        temperature=0,
        extra_body={"debug_config": {"ignore_eos": not protected}},
    ).model_dump()
    usage = r["usage"]
    cached = usage["prompt_tokens"] - usage["extra"]["prefill_tokens"]
    text = r["choices"][0]["text"]
    if protected:
        assert cached >= len(base), (name, cached, len(base))
        expected = re.search(
            r"Return only this code: (ORBIT-[^<\n]+)", tokenizer.decode(tokens)
        ).group(1)
        assert expected in text, (name, expected, text)
    row = dict(case=name, seconds=time.perf_counter() - start, cached_tokens=cached, response=r)
    rows.append(row)
    args.output.write_text(json.dumps(dict(base_tokens=len(base), rows=rows), indent=2) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "response"}), flush=True)
    return row


try:
    engine.completions.create(
        prompt=base,
        max_tokens=1,
        temperature=0,
        extra_body={"debug_config": {"pinned_system_prompt": True}},
    )
    a = prompt("A")
    b = prompt("B")
    run("client_a", a)
    run("client_b", b)
    reused = run("client_a_again", a)
    assert reused["cached_tokens"] == len(a) - 1
    run("client_c", prompt("C"))
    recent = run("client_a_after_c", a)
    assert recent["cached_tokens"] == len(a) - 1
    run("client_d", prompt("D"))
    run("client_e", prompt("E"))
    run("client_a_after_eviction", a)
    for token in (100, 101, 102):
        run(f"raw_{token}", [token] * 700, protected=False, count=8)
    after_raw = run("after_raw_eviction", b)
    assert after_raw["cached_tokens"] == len(base)
    stream = engine.completions.create(
        prompt=prompt("cancel"),
        max_tokens=128,
        temperature=0,
        stream=True,
        request_id="protected-cancel",
    )
    next(stream)
    engine.abort("protected-cancel")
    try:
        for _ in stream:
            pass
    except ValidationError as exc:
        assert all(e.get("input") == "abort" and "finish_reason" in e["loc"] for e in exc.errors())
    finally:
        stream.close()
    run("after_cancel", a)
finally:
    engine.terminate()
