"""Direct-VM, greedy Unified 12B MTP benchmark (one sequence, compiled context limit).

Shared K/V and hidden states stay on the GPU (bench_xavier_e4b_mtp.Trial); only
the selected token IDs are copied back. Not the MLCEngine serving path.
"""

import argparse
import json
from pathlib import Path

import bench_xavier_e4b_mtp
import numpy as np
import tvm_ffi
from mtp_program import Program  # The resident services import it from here.
from xavier_bench_prompts import DESIGN_NOTE, FILLER

import mlc_llm
from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.serve import engine_utils
from mlc_llm.tokenizers import Tokenizer


class Trial(bench_xavier_e4b_mtp.Trial):
    """The Unified 12B shapes: eight K/V heads in local layers, one in global ones."""

    def __init__(self, target, assistant, runtime_library, capture=None):
        super().__init__(target, assistant, runtime_library, hidden_size=3840, kv_heads=(8, 1))
        self.capture = capture

    def draft(self, token, previous, assistant=None, states=None):
        if self.capture:
            # One actual step with its inputs, for comparison with the reference.
            committed = int(self.position.numpy()[0])
            embedding = self.target.call("embed", self.tensor(np.array([token], dtype="int32")))
            embedding = self.reshape(embedding, tvm_ffi.Shape(self.hidden.shape))
            logits, hidden = self.assistant.call(
                "mtp_step", embedding, previous, *self.states, self.position, self.position
            )
            names = ("local_key", "local_value", "global_key", "global_value")
            np.savez(
                self.capture,
                embedding=embedding.numpy(),
                previous_hidden=previous.numpy(),
                **{name: s.numpy()[:, :committed] for name, s in zip(names, self.states)},
                position=np.array([committed], dtype="int32"),
                logits=logits.numpy(),
                hidden=hidden.numpy(),
            )
            self.capture = None
        return super().draft(token, previous, assistant, states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="dist/gemma4-12b-mtp-target")
    parser.add_argument("--target-lib", default="dist/libs/gemma4-12b-mtp-target-sm72.so")
    parser.add_argument("--assistant", required=True)
    parser.add_argument("--assistant-lib", default="dist/libs/gemma4-12b-mtp-assistant-sm72.so")
    parser.add_argument("--runtime", default="dist/libs/xavier-mtp-runtime.so")
    parser.add_argument("--attention", choices=["baseline", "paged"], default="baseline")
    parser.add_argument(
        "--draft-tokens",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="three need a target built with MLC_MTP_VERIFY_LEN=4",
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--cases", default="short,chat")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--capture", help="Save one actual assistant step for HF comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads((Path(args.target) / "mlc-chat-config.json").read_text())
    tokenizer = Tokenizer(args.target)
    prompts = {
        "short": "What is 17 multiplied by 23? Give only the number.",
        "chat": DESIGN_NOTE,
        "448": "Read this and reply with one word.\n" + FILLER * 13,
    }

    def encode(text):
        conv = Conversation.model_validate(config["conv_template"])
        conv.messages = [("user", text), ("assistant", None)]
        parts = engine_utils.process_prompts(conv.as_prompt(config), tokenizer.encode)
        ids = [i for part in parts for i in part]
        return (conv.system_prefix_token_ids or []) + ids

    if args.attention == "paged":
        from mlc_llm.op.xavier_prefill import install

        install("paged", Path(args.target) / "mlc-chat-config.json")
    print("Loading target and official MTP assistant", flush=True)
    target = Program(args.target, args.target_lib)
    assistant = Program(args.assistant, args.assistant_lib)
    trial = Trial(target, assistant, args.runtime, args.capture)
    results = dict(
        args=vars(args),
        prototype="direct VM, state on the GPU",
        mlc_source=mlc_llm.__file__,
        runs=[],
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(args.warmups):
        for width in (0, args.draft_tokens):
            trial.run(encode(prompts["short"]), 4, width, ignore_eos=True)
    for case in args.cases.split(","):
        for repeat in range(args.repeats):
            baseline = None
            for width in (0, args.draft_tokens):
                result = trial.run(encode(prompts[case]), args.tokens, width, args.ignore_eos)
                result.update(
                    case=case,
                    repeat=repeat,
                    draft_tokens=width,
                    text=tokenizer.decode(result["tokens"]),
                )
                if baseline is None:
                    baseline = result["tokens"]
                else:
                    result["matches_plain"] = result["tokens"] == baseline
                results["runs"].append(result)
                out.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)
                if result.get("matches_plain") is False:
                    raise AssertionError("Speculative output differs from target-only output")


if __name__ == "__main__":
    main()
