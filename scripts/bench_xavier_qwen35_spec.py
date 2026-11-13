"""Greedy speculative decoding of a Qwen3.5 target with a small Qwen3.5 draft, measured.

Each prompt is decoded by the target alone and with the draft (scripts/qwen35_spec.py), and
the two token sequences are compared. With `--follow-up` the conversation then goes on by
one user turn, once continued from what the models hold and once read from the start.
"""

import argparse
import json
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
target = Program(args.target, args.target_lib, device, history=1)
draft = Program(args.draft, args.draft_lib, device, history=args.drafts + 1)
plain, speculative = Decoder(target), Decoder(target, draft, args.drafts)


def run(decoder, prompt, fresh=True):
    if fresh:
        # Both decoders use the target: neither may find the other's conversation in it.
        plain.forget()
        speculative.forget()
    return decoder.generate(prompt, args.tokens, stops)


# Kernels and memory pools are set up by the first calls.
run(speculative, encode(PROMPTS["short"]))
run(plain, encode(PROMPTS["short"]))

results = dict(args=vars(args), runs=[])
for case in args.cases.split(","):
    prompt = encode(PROMPTS[case])
    reference, trial = run(plain, prompt), run(speculative, prompt)
    report = dict(
        case=case,
        prompt_tokens=len(prompt),
        answer_tokens=len(trial["tokens"]),
        **compare(trial, reference),
        plain=numbers(reference),
        speculative=numbers(trial),
        text=answer(trial),
        plain_text=None if trial["tokens"] == reference["tokens"] else answer(reference),
    )
    if args.follow_up:
        # The models hold the conversation of `trial`; the same turn is then read anew.
        prompt = encode(PROMPTS[case], answer(trial), FOLLOW_UP)
        continued = run(speculative, prompt, fresh=False)
        anew = run(speculative, prompt)
        report["follow_up"] = dict(
            prompt_tokens=len(prompt),
            **compare(continued, anew),
            continued=numbers(continued),
            anew=numbers(anew),
            text=answer(continued),
        )
    results["runs"].append(report)
    print(json.dumps(report), flush=True)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2) + "\n")
