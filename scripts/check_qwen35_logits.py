"""Teacher-forced logits of a compiled Qwen3.5 library, driven through the Relax VM.

A reference run decodes greedily and records its tokens and logits. A comparison run feeds
the reference tokens to another library and reports how far its logits are from the
reference. `--repeat` reruns the same sequence in one process and compares bit for bit.
`--verify 2,3,4` feeds the same tokens several at a time through `batch_verify` and compares
with the token-by-token logits of the same library; `--hold 4:2` does so through
`verify_hold`, committing two of every four rows with `commit_rows` and feeding the others
again, as speculative decoding does after a rejected proposal. With several `--case` names
the case is inserted before the suffix of `--out` and `--force`; forcing another model's
tokens on a draft model measures how often the draft predicts them.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm_ffi
import tvm_tirx
from mtp_program import release_file_cache
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime.vm import VirtualMachine
from tvm_ffi import Shape
from xavier_bench_prompts import CODE, DESIGN_NOTE, FILLER, PLAN, PROSE

from mlc_llm.conversation_template import ConvTemplateRegistry
from mlc_llm.protocol.mlc_chat_config import MLCChatConfig
from mlc_llm.serve import engine_utils
from mlc_llm.support.auto_device import detect_device
from mlc_llm.tokenizers import Tokenizer

PROMPTS = {
    "chat": DESIGN_NOTE,
    "448": "Read this and reply with one word.\n" + FILLER * 13,
    "1768": "Read this and reply with one word.\n" + FILLER * 53,
    "short": "What is 17 multiplied by 23? Give only the number.",
    "code": CODE,
    "prose": PROSE,
    "plan": PLAN,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Converted model directory")
parser.add_argument("--lib", required=True)
parser.add_argument("--case", default="chat", help="One or several of " + ",".join(PROMPTS))
parser.add_argument("--tokens", type=int, default=48, help="Decode steps after the prefill")
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--chunk", type=int, default=512)
parser.add_argument("--context", type=int, default=4096)
parser.add_argument("--force", help="Reference .npz whose tokens are fed instead of the argmax")
parser.add_argument("--verify", default="", help="Row counts for batch_verify, e.g. 2,3,4")
parser.add_argument("--hold", default="", help="rows:kept for verify_hold, e.g. 4:4,4:2,3:1")
parser.add_argument("--out", required=True)
args = parser.parse_args()

model = Path(args.model)
device = detect_device("cuda")
start = time.perf_counter()
executable = tvm_ffi.load_module(args.lib)
metadata = json.loads(VirtualMachine(executable, tvm_ffi.device("cpu"))["_metadata"]())
mod = relax.VirtualMachine(executable, device).module
loaded, _ = tvmjs.load_tensor_cache(str(model), device)
params = [loaded[p["name"]] for p in metadata["params"]]
del loaded
release_file_cache(model)
print(f"loaded in {time.perf_counter() - start:.1f} s", flush=True)

text = (model / "mlc-chat-config.json").read_text(encoding="utf-8")
tokenizer = Tokenizer(str(model))


def encode(case):
    """Token ids of the case as one user turn in the model's conversation template."""
    template = MLCChatConfig.model_validate_json(text).conv_template
    conversation = (
        ConvTemplateRegistry.get_conv_template(template) if isinstance(template, str) else template
    )
    conversation.messages.append(("user", PROMPTS[case]))
    conversation.messages.append(("assistant", None))
    messages = conversation.as_prompt(json.loads(text))
    pieces = engine_utils.process_prompts(messages, tokenizer.encode)
    assert all(isinstance(piece, list) for piece in pieces)
    ids = [token for piece in pieces for token in piece]
    return (conversation.system_prefix_token_ids or []) + ids


add_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
remove_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_remove_sequence")
begin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
end_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
popn = tvm_ffi.get_global_func("vm.builtin.kv_state_popn")
reshape = tvm_ffi.get_global_func("vm.builtin.reshape")
cache = mod["create_tir_paged_kv_cache"](
    Shape([1]), Shape([args.context]), Shape([args.chunk]), Shape([16]), Shape([0])
)
state = mod["create_rnn_state"](Shape([1]), Shape([1]))


def embed(tokens):
    ids = tvm_tirx.runtime.tensor(np.array(tokens, "int32"), device=device)
    out = mod["embed"](ids, params)
    return reshape(out, Shape([1, len(tokens), out.shape[1]]))


def forward(name, tokens):
    begin_forward(cache, Shape([0]), Shape([len(tokens)]))
    begin_forward(state, Shape([0]), Shape([len(tokens)]))
    if name == "batch_prefill":
        last = tvm_tirx.runtime.tensor(np.array([len(tokens) - 1], "int32"), device=device)
        result = mod[name](embed(tokens), last, cache, state, params)
    else:
        result = mod[name](embed(tokens), cache, state, params)
    end_forward(cache)
    end_forward(state)
    logits = result[0]
    logits = logits.numpy() if hasattr(logits, "numpy") else logits.asnumpy()
    return logits.reshape(-1 if name == "batch_prefill" else len(tokens), logits.shape[-1])


def run(forced):
    """Logits after the prompt and after each fed token; tokens are `forced` or the argmax."""
    add_sequence(cache, 0)
    add_sequence(state, 0)
    rows, tokens, seconds = [], [], []
    begin = time.perf_counter()
    for offset in range(0, len(prompt), args.chunk):
        logits = forward("batch_prefill", prompt[offset : offset + args.chunk])
    seconds.append(time.perf_counter() - begin)
    rows.append(logits[0])
    for step in range(args.tokens):
        tokens.append(int(forced[step]) if forced is not None else int(rows[-1].argmax()))
        begin = time.perf_counter()
        rows.append(forward("batch_decode", [tokens[-1]])[0])
        seconds.append(time.perf_counter() - begin)
    remove_sequence(cache, 0)
    remove_sequence(state, 0)
    return np.stack(rows), np.array(tokens, "int32"), np.array(seconds)


def run_verify(tokens, group):
    """Logits after the prompt and after each of `tokens`, fed `group` at a time."""
    add_sequence(cache, 0)
    add_sequence(state, 0)
    for offset in range(0, len(prompt), args.chunk):
        logits = forward("batch_prefill", prompt[offset : offset + args.chunk])
    rows, seconds = [logits[0]], []
    for offset in range(0, len(tokens), group):
        begin = time.perf_counter()
        rows.extend(forward("batch_verify", [int(t) for t in tokens[offset : offset + group]]))
        seconds.append(time.perf_counter() - begin)
    remove_sequence(cache, 0)
    remove_sequence(state, 0)
    return np.stack(rows), np.array(seconds)


def run_hold(tokens, rows, kept):
    """As `run_verify` through `verify_hold`: of every `rows` tokens fed, `kept` are committed."""
    add_sequence(cache, 0)
    add_sequence(state, 0)
    for offset in range(0, len(prompt), args.chunk):
        logits = forward("batch_prefill", prompt[offset : offset + args.chunk])
    out, seconds, offset = [logits[0]], [], 0
    while offset < len(tokens):
        part = [int(t) for t in tokens[offset : offset + rows]]
        keep = min(kept, len(part))
        begin = time.perf_counter()
        begin_forward(cache, Shape([0]), Shape([len(part)]))
        begin_forward(state, Shape([0]), Shape([0]))  # the state advances in commit_rows
        result = mod["verify_hold"](embed(part), cache, state, params)
        end_forward(cache)
        end_forward(state)
        logits = result[0].numpy().reshape(len(part), -1)
        if len(part) > keep:
            popn(cache, 0, len(part) - keep)
        begin_forward(state, Shape([0]), Shape([keep]))
        mod["commit_rows"](result[3], Shape([keep]), state)
        end_forward(state)
        device.sync()
        seconds.append(time.perf_counter() - begin)
        out.extend(logits[:keep])
        offset += keep
    remove_sequence(cache, 0)
    remove_sequence(state, 0)
    return np.stack(out), np.array(seconds)


def compare(logits, reference):
    """Per-step distance of `logits` from `reference` (both steps x vocabulary)."""
    order = np.argsort(reference, axis=1)
    top, second = order[:, -1], order[:, -2]
    steps = np.arange(len(reference))
    margin = reference[steps, top] - reference[steps, second]
    same = logits.argmax(axis=1) == top
    difference = np.abs(logits - reference).max(axis=1)
    cosine = (logits * reference).sum(axis=1) / (
        np.linalg.norm(logits, axis=1) * np.linalg.norm(reference, axis=1)
    )
    flipped = [
        dict(step=int(i), margin=float(margin[i]), difference=float(difference[i]))
        for i in np.flatnonzero(~same)
    ]
    return dict(
        steps=len(reference),
        same_argmax=int(same.sum()),
        bit_identical=bool(np.array_equal(logits, reference)),
        first_differing_step=int(np.flatnonzero(difference > 0)[0]) if difference.any() else None,
        max_abs_difference=float(difference.max()),
        mean_abs_difference=float(np.abs(logits - reference).mean()),
        min_cosine=float(cosine.min()),
        smallest_reference_margin=float(margin.min()),
        flipped=flipped,
    )


cases = args.case.split(",")


def named(path, case):
    path = Path(path)
    return path if len(cases) == 1 else path.with_name(f"{path.stem}.{case}{path.suffix}")


for case in cases:
    prompt = encode(case)
    reference = np.load(named(args.force, case)) if args.force else None
    forced = reference["tokens"][: args.tokens] if reference is not None else None
    first_logits, first_tokens, first_seconds = run(forced)
    report = dict(
        args=vars(args),
        case=case,
        prompt_tokens=len(prompt),
        tokens=first_tokens.tolist(),
        prefill_s=float(first_seconds[0]),
        prefill_tok_s=len(prompt) / float(first_seconds[0]),
        decode_tok_s=float(args.tokens / first_seconds[1:].sum()) if args.tokens else None,
        decode_ms_median=float(np.median(first_seconds[1:]) * 1e3) if args.tokens else None,
        text=tokenizer.decode(first_tokens.tolist()),
    )
    if reference is not None:
        assert list(reference["prompt"]) == prompt
        report["against_reference"] = compare(first_logits, reference["logits"][: args.tokens + 1])
    report["repeats"] = []
    for _ in range(args.repeat - 1):
        logits, _, seconds = run(first_tokens)
        again = compare(logits, first_logits)
        again["prefill_s"] = float(seconds[0])
        report["repeats"].append(again)
    report["verify"] = []
    for group in [int(x) for x in args.verify.split(",") if x]:
        logits, seconds = run_verify(first_tokens, group)
        full = seconds[: len(first_tokens) // group]  # calls that carry `group` rows
        grouped = compare(logits, first_logits)
        grouped.update(rows=group, calls=len(full), ms_per_call=float(np.median(full) * 1e3))
        report["verify"].append(grouped)
    report["hold"] = []
    for rows, kept in [map(int, x.split(":")) for x in args.hold.split(",") if x]:
        logits, seconds = run_hold(first_tokens, rows, kept)
        held = compare(logits, first_logits)
        held.update(rows=rows, kept=kept, calls=len(seconds))
        held["ms_per_call"] = float(np.median(seconds) * 1e3)
        report["hold"].append(held)
    out = named(args.out, case)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=first_logits, tokens=first_tokens, prompt=np.array(prompt, "int32"))
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
