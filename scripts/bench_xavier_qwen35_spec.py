"""Greedy speculative decoding of a Qwen3.5 target with a small Qwen3.5 draft (Relax VM).

Every round the draft proposes `--drafts` tokens. The target scores the last accepted token
and the proposals in one `verify_hold` step, keeps the proposals up to its first disagreement
and adds its own next token. `commit_rows` then advances the target's recurrent state by the
kept rows only; the draft rolls back through its state history. Each prompt is also decoded
token by token with the target alone, and the two token sequences are compared.
"""

import argparse
import json
import time
from collections import Counter
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
    "code": CODE,
    "prose": PROSE,
    "plan": PLAN,
    "448": "Read this and reply with one word.\n" + FILLER * 13,
    "1768": "Read this and reply with one word.\n" + FILLER * 53,
    "short": "What is 17 multiplied by 23? Give only the number.",
}

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, help="Converted target model directory")
parser.add_argument("--target-lib", required=True)
parser.add_argument("--draft", required=True, help="Converted draft model directory")
parser.add_argument("--draft-lib", required=True)
parser.add_argument("--cases", default="chat,code,prose,plan")
parser.add_argument("--tokens", type=int, default=128, help="Token limit of an answer")
parser.add_argument("--drafts", type=int, default=3, help="Proposals per round, at most 3")
parser.add_argument("--chunk", type=int, default=512)
parser.add_argument("--context", type=int, default=4096)
parser.add_argument("--output", required=True)
args = parser.parse_args()
assert 1 <= args.drafts <= 3, "the small-batch kernel verifies at most four rows"

device = detect_device("cuda")
add_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
remove_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_remove_sequence")
begin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
end_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
popn = tvm_ffi.get_global_func("vm.builtin.kv_state_popn")
reshape = tvm_ffi.get_global_func("vm.builtin.reshape")


class Program:
    """One compiled model with an attention cache and a recurrent state for one sequence."""

    def __init__(self, model, lib, history):
        executable = tvm_ffi.load_module(lib)
        metadata = json.loads(VirtualMachine(executable, tvm_ffi.device("cpu"))["_metadata"]())
        self.mod = relax.VirtualMachine(executable, device).module
        loaded, _ = tvmjs.load_tensor_cache(str(model), device)
        self.params = [loaded[p["name"]] for p in metadata["params"]]
        release_file_cache(model)
        self.cache = self.mod["create_tir_paged_kv_cache"](
            Shape([1]), Shape([args.context]), Shape([args.chunk]), Shape([16]), Shape([0])
        )
        # Single-token steps are kept in `history` slots, which is what `pop` goes back through.
        self.state = self.mod["create_rnn_state"](Shape([1]), Shape([history]))

    def start(self):
        add_sequence(self.cache, 0)
        add_sequence(self.state, 0)

    def stop(self):
        remove_sequence(self.cache, 0)
        remove_sequence(self.state, 0)

    def forward(self, name, tokens, *extra, advance=True):
        ids = tvm_tirx.runtime.tensor(np.array(tokens, "int32"), device=device)
        embeds = self.mod["embed"](ids, self.params)
        embeds = reshape(embeds, Shape([1, len(tokens), embeds.shape[1]]))
        begin_forward(self.cache, Shape([0]), Shape([len(tokens)]))
        begin_forward(self.state, Shape([0]), Shape([len(tokens) if advance else 0]))
        result = self.mod[name](embeds, *extra, self.cache, self.state, self.params)
        end_forward(self.cache)
        end_forward(self.state)
        return result

    def prefill(self, prompt):
        """The token after the prompt."""
        for offset in range(0, len(prompt), args.chunk):
            part = prompt[offset : offset + args.chunk]
            last = tvm_tirx.runtime.tensor(np.array([len(part) - 1], "int32"), device=device)
            result = self.forward("batch_prefill", part, last)
        return int(result[0].numpy().argmax())

    def decode(self, token):
        """The token after `token`."""
        return int(self.forward("batch_decode", [token])[0].numpy().argmax())

    def verify(self, tokens):
        """The token after each of `tokens`, and what `commit` needs; the state stays."""
        result = self.forward("verify_hold", tokens, advance=False)
        return result[0].numpy().reshape(len(tokens), -1).argmax(axis=1), result[3]

    def commit(self, held, rows, drop):
        """Keep the first `rows` rows of the last `verify` and drop the other `drop`."""
        if drop:
            popn(self.cache, 0, drop)
        begin_forward(self.state, Shape([0]), Shape([rows]))
        self.mod["commit_rows"](held, Shape([rows]), self.state)
        end_forward(self.state)
        device.sync()

    def pop(self, count):
        popn(self.cache, 0, count)
        popn(self.state, 0, count)


def now():
    return time.perf_counter()


def common(a, b):
    """Length of the common prefix."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def finish(tokens):
    """Cut at the token limit and after the first stop token."""
    tokens = tokens[: args.tokens]
    for i, token in enumerate(tokens):
        if token in stop:
            return tokens[: i + 1], True
    return tokens, len(tokens) >= args.tokens


def plain(prompt):
    target.start()
    begin = now()
    tokens = [target.prefill(prompt)]
    first = now() - begin
    begin = now()
    while not finish(tokens)[1]:
        tokens.append(target.decode(tokens[-1]))
    seconds = now() - begin
    target.stop()
    return dict(
        tokens=tokens,
        first_token_s=first,
        decode_s=seconds,
        decode_tok_s=(len(tokens) - 1) / seconds if len(tokens) > 1 else None,
    )


def speculative(prompt):
    target.start()
    draft.start()
    begin = now()
    tokens = [target.prefill(prompt)]
    first = now() - begin
    begin = now()
    draft.prefill(prompt)
    device.sync()
    draft_prefill = now() - begin
    clock, accepted_counts, waiting = Counter(), [], None
    while not finish(tokens)[1]:
        start = now()
        # A proposal the target accepted last round but the draft has not read yet.
        if waiting is not None:
            draft.decode(waiting)
        proposals = []
        for _ in range(args.drafts):
            proposals.append(draft.decode(proposals[-1] if proposals else tokens[-1]))
        drafted = now()
        choice, held = target.verify([tokens[-1], *proposals])
        accepted = 0
        while accepted < args.drafts and proposals[accepted] == choice[accepted]:
            accepted += 1
        verified = now()
        target.commit(held, accepted + 1, args.drafts - accepted)
        # The draft has read all proposals but the last.
        waiting = proposals[-1] if accepted == args.drafts else None
        if accepted < args.drafts - 1:
            draft.pop(args.drafts - 1 - accepted)
        tokens.extend(int(x) for x in choice[: accepted + 1])
        accepted_counts.append(accepted)
        clock.update(draft=drafted - start, verify=verified - drafted, commit=now() - verified)
    seconds = now() - begin
    target.stop()
    draft.stop()
    tokens = finish(tokens)[0]
    rounds = len(accepted_counts)
    return dict(
        tokens=tokens,
        first_token_s=first,
        draft_prefill_s=draft_prefill,
        decode_s=seconds,
        decode_tok_s=(len(tokens) - 1) / seconds if len(tokens) > 1 else None,
        rounds=rounds,
        accepted=dict(sorted(Counter(accepted_counts).items())),
        proposals_accepted=sum(accepted_counts) / (rounds * args.drafts) if rounds else None,
        ms_per_round={k: v / rounds * 1e3 for k, v in clock.items()} if rounds else None,
    )


text = (Path(args.target) / "mlc-chat-config.json").read_text(encoding="utf-8")
tokenizer = Tokenizer(args.target)
stop = set(json.loads(text)["conv_template"]["stop_token_ids"])


def encode(case):
    """Token ids of the case as one user turn in the target's conversation template."""
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


begin = now()
target = Program(args.target, args.target_lib, history=1)
draft = Program(args.draft, args.draft_lib, history=args.drafts + 1)
print(f"loaded in {now() - begin:.1f} s", flush=True)
# Kernels and memory pools are set up by the first calls.
speculative(encode("short"))
plain(encode("short"))

results = dict(args=vars(args), runs=[])
for case in args.cases.split(","):
    prompt = encode(case)
    reference, trial = plain(prompt), speculative(prompt)
    same = trial["tokens"] == reference["tokens"]
    rates = trial["decode_tok_s"], reference["decode_tok_s"]
    run = dict(
        case=case,
        prompt_tokens=len(prompt),
        answer_tokens=len(trial["tokens"]),
        same_tokens=same,
        first_difference=None if same else common(trial["tokens"], reference["tokens"]),
        speedup=rates[0] / rates[1] if all(rates) else None,
        plain={k: v for k, v in reference.items() if k != "tokens"},
        speculative={k: v for k, v in trial.items() if k != "tokens"},
        text=tokenizer.decode(trial["tokens"]),
        plain_text=None if same else tokenizer.decode(reference["tokens"]),
    )
    results["runs"].append(run)
    print(json.dumps(run), flush=True)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2) + "\n")
