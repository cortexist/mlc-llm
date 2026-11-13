"""Greedy speculative decoding of a Qwen3.5 target with a small Qwen3.5 draft, measured.

Each prompt is decoded by the target alone and with the draft (scripts/qwen35_spec.py), and
the two token sequences are compared. With `--follow-up` the conversation then goes on by
one user turn, once continued from what the models hold and once read from the start
(with `--head` the head's conversation does the same).

With `--head` (a target built with MLC_QWEN35_MTP=1) each prompt is also decoded with the
proposals of the target's own multi-token prediction head, the same weights loaded once.
Every run reports where its time went: reading the prompt, proposing, verifying, the rest.
"""

import argparse
import json
import time
from pathlib import Path

from qwen35_spec import Decoder, Program
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
    "long": FILLER * 100 + "\nSummarize the text above in five sentences.",
    "short": "What is 17 multiplied by 23? Give only the number.",
}
FOLLOW_UP = "Now give the main point of your answer in one sentence."

parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True, help="Converted target model directory")
parser.add_argument("--target-lib", required=True)
parser.add_argument("--draft", required=True, help="Converted draft model directory")
parser.add_argument("--draft-lib", required=True)
parser.add_argument("--cases", default="chat,code,prose,plan")
parser.add_argument("--tokens", type=int, default=128, help="Token limit of an answer")
parser.add_argument("--drafts", type=int, default=3, help="Proposals per round, at most 3")
parser.add_argument("--follow-up", action="store_true", help="Add a second user turn")
parser.add_argument("--head", action="store_true", help="Also propose with the MTP head")
parser.add_argument("--no-plain", action="store_true", help="Leave the target alone out")
parser.add_argument(
    "--attention", choices=["baseline", "paged"], default="baseline", help="As the service"
)
parser.add_argument("--output", required=True)
args = parser.parse_args()

text = (Path(args.target) / "mlc-chat-config.json").read_text(encoding="utf-8")
tokenizer = Tokenizer(args.target)
stops = set(json.loads(text)["conv_template"]["stop_token_ids"])


def encode(*turns):
    """Token ids of user and assistant turns in the target's conversation template."""
    template = MLCChatConfig.model_validate_json(text).conv_template
    conversation = (
        ConvTemplateRegistry.get_conv_template(template) if isinstance(template, str) else template
    )
    for index, turn in enumerate(turns):
        conversation.messages.append(("assistant" if index % 2 else "user", turn))
    conversation.messages.append(("assistant", None))
    messages = conversation.as_prompt(json.loads(text))
    pieces = engine_utils.process_prompts(messages, tokenizer.encode)
    assert all(isinstance(piece, list) for piece in pieces)
    ids = [token for piece in pieces for token in piece]
    return (conversation.system_prefix_token_ids or []) + ids


def common(a, b):
    """Length of the common prefix."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def compare(trial, reference):
    if reference is None:
        return {}
    same = trial["tokens"] == reference["tokens"]
    rates = trial["decode_tokens_per_s"], reference["decode_tokens_per_s"]
    return dict(
        same_tokens=same,
        first_difference=None if same else common(trial["tokens"], reference["tokens"]),
        speedup=rates[0] / rates[1] if all(rates) else None,
    )


def numbers(result):
    return {k: v for k, v in result.items() if k != "tokens"}


def answer(result):
    tokens = result["tokens"]
    return tokenizer.decode(tokens[:-1] if result["stopped"] else tokens)


device = detect_device("cuda")
receipt = None
if args.attention == "paged":
    from mlc_llm.op.xavier_prefill import install_qwen35

    receipt = install_qwen35()
target = Program(args.target, args.target_lib, device, history=1)
draft = Program(args.draft, args.draft_lib, device, history=args.drafts + 1)
assert receipt is None or receipt["caches"] == 2, receipt
plain, speculative = Decoder(target), Decoder(target, draft, args.drafts)
decoders = [plain, speculative]
head = None
if args.head:
    # The same program in its other mode; a conversation is in one mode from its start.
    target.mtp = True
    head = Decoder(target, None, args.drafts)
    target.mtp = False
    decoders.append(head)

spent = {}


def timed(program, name, label):
    """Seconds spent in a method of a program, under `label`."""
    method = getattr(program, name)

    def call(*args_, **more):
        begin = time.perf_counter()
        try:
            return method(*args_, **more)
        finally:
            spent[label] = spent.get(label, 0.0) + time.perf_counter() - begin

    setattr(program, name, call)


# A step ends with its logits on the host, so each time is that of the step itself; the
# commit of a round is not awaited and ends within the next step.
timed(target, "prefill", "target_read_s")
timed(target, "propose", "propose_s")
timed(target, "verify", "verify_s")
timed(target, "commit", "commit_s")
timed(target, "decode", "target_step_s")
timed(draft, "prefill", "draft_read_s")
timed(draft, "decode", "propose_s")
timed(draft, "pop", "draft_pop_s")


def follow_up(decoder, case, held):
    """The next turn after the answer `held`, which the models of `decoder` still hold."""
    prompt = encode(PROMPTS[case], answer(held), FOLLOW_UP)
    continued = run(decoder, prompt, fresh=False)
    anew = run(decoder, prompt)
    return dict(
        prompt_tokens=len(prompt),
        **compare(continued, anew),
        continued=numbers(continued),
        anew=numbers(anew),
        text=answer(continued),
    )


def run(decoder, prompt, fresh=True):
    if fresh:
        # All decoders use the target: none may find another's conversation in it.
        for other in decoders:
            other.forget()
        target.mtp = decoder is head
    spent.clear()
    result = decoder.generate(prompt, args.tokens, stops)
    result.update({label: round(seconds, 4) for label, seconds in spent.items()})
    if result["rounds"]:
        for label in ("propose_s", "verify_s", "commit_s"):
            result[label.replace("_s", "_ms_per_round")] = round(
                1e3 * spent.get(label, 0.0) / result["rounds"], 2
            )
    return result


# Kernels and memory pools are set up by the first calls.
for decoder in decoders[::-1]:
    run(decoder, encode(PROMPTS["short"]))

results = dict(args=vars(args), runs=[])
for case in args.cases.split(","):
    prompt = encode(PROMPTS[case])
    reference = None if args.no_plain else run(plain, prompt)
    trial = run(speculative, prompt)
    report = dict(
        case=case,
        prompt_tokens=len(prompt),
        answer_tokens=len(trial["tokens"]),
        **compare(trial, reference),
        plain=reference and numbers(reference),
        speculative=numbers(trial),
        text=answer(trial),
        plain_text=None
        if reference is None or trial["tokens"] == reference["tokens"]
        else answer(reference),
    )
    if args.follow_up:
        report["follow_up"] = follow_up(speculative, case, trial)
    if head is not None:
        other = run(head, prompt)
        rates = other["decode_tokens_per_s"], trial["decode_tokens_per_s"]
        report["head"] = dict(
            **numbers(other),
            **{f"{name}_to_plain": value for name, value in compare(other, reference).items()},
            same_tokens_as_draft=other["tokens"] == trial["tokens"],
            first_difference_to_draft=None
            if other["tokens"] == trial["tokens"]
            else common(other["tokens"], trial["tokens"]),
            speedup_to_draft=rates[0] / rates[1] if all(rates) else None,
            text=None if other["tokens"] == trial["tokens"] else answer(other),
        )
        if args.follow_up:
            report["head_follow_up"] = follow_up(head, case, other)
    results["runs"].append(report)
    print(json.dumps(report), flush=True)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2) + "\n")
