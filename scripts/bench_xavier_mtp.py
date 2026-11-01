"""Direct-VM, greedy Unified 12B MTP prototype (one sequence, <=1024 tokens).

Timings include Python control and CPU copies of shared KV/hidden states. This
is a correctness/acceptance experiment, not the MLCEngine serving path.
"""

import argparse
import json
import time
from pathlib import Path

import mlc_llm  # Register runtime functions before loading compiled programs.
import numpy as np
import tvm_ffi
import tvm_tirx
from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.serve import engine_utils
from mlc_llm.tokenizers import Tokenizer
from xavier_bench_prompts import DESIGN_NOTE, FILLER


class Program:
    def __init__(self, model, library):
        self.module = tvm_ffi.load_module(str(library))
        self.vm = self.module["vm_load_executable"]()
        self.vm["vm_initialization"](2, 0, 1, 0)
        metadata = json.loads(self.vm["_metadata"]())
        get = tvm_ffi.get_global_func
        get("vm.builtin.tensor_cache.load")(str(model), 2, 0)
        self.params = get("vm.builtin.param_array_from_cache_by_name")(
            [p["name"] for p in metadata["params"]]
        )
        get("vm.builtin.tensor_cache.clear")()

    def call(self, name, *args):
        return self.vm[name](*args, self.params)


class Trial:
    def __init__(self, target, assistant, capture=None):
        self.target, self.assistant = target, assistant
        self.device = tvm_tirx.cuda(0)
        self.capture = capture
        self.captured = False
        self.builtin = lambda name: tvm_ffi.get_global_func("vm.builtin.kv_state_" + name)
        self.cache = target.vm["create_tir_paged_kv_cache"](
            *[tvm_ffi.Shape([n]) for n in (1, 1024, 512, 16, 0)]
        )

    def tensor(self, array):
        return tvm_tirx.runtime.tensor(np.ascontiguousarray(array), self.device)

    def forward(self, ids, prefill=False):
        self.builtin("begin_forward")(self.cache, tvm_ffi.Shape([0]), tvm_ffi.Shape([len(ids)]))
        result = self.target.call(
            "mtp_prefill" if prefill else "mtp_verify",
            self.tensor(np.array([ids], dtype="int32")),
            self.cache,
        )
        self.builtin("end_forward")(self.cache)
        return [x.numpy() for x in result[:6]]

    def run(self, prompt, count, drafts, ignore_eos=False):
        if not 0 <= drafts <= 2 or count < 1:
            raise ValueError("Use 0..2 draft tokens and a positive output limit")
        if len(prompt) + count + drafts > 1024:
            raise ValueError("Prototype context must fit 1024 tokens including verification")
        self.builtin("clear")(self.cache)
        self.builtin("add_sequence")(self.cache, 0)
        self.device.sync()
        start = time.perf_counter()
        logits, hidden, *states = self.forward(prompt, prefill=True)
        frontier = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        hidden = hidden[:, -1:, :]
        output = [frontier]
        first = time.perf_counter()
        committed = len(prompt)
        accepted = proposed = rounds = 0
        accepted_by_round = []
        stop = {1, 50, 106}
        while len(output) < count and (ignore_eos or output[-1] not in stop):
            proposals = []
            last = frontier
            previous = hidden
            # Reserve one output slot for the target's bonus token.
            width = min(drafts, count - len(output) - 1)
            shared = (
                [
                    self.tensor(np.pad(a, ((0, 0), (0, 1024 - a.shape[1]), (0, 0), (0, 0))))
                    for a in states
                ]
                if width
                else []
            )
            position = self.tensor(np.array([committed], dtype="int32"))
            context_length = self.tensor(np.array([states[0].shape[1]], dtype="int32"))
            for _ in range(width):
                embedding = (
                    self.target.call("embed", self.tensor(np.array([last], dtype="int32")))
                    .numpy()
                    .reshape(1, 1, -1)
                )
                inputs = [embedding, previous, *states]
                result = self.assistant.call(
                    "mtp_step",
                    self.tensor(embedding),
                    self.tensor(previous),
                    *shared,
                    position,
                    context_length,
                )
                draft_logits, previous = (a.numpy() for a in result)
                if self.capture and not self.captured:
                    np.savez(
                        self.capture,
                        **dict(
                            zip(
                                (
                                    "embedding",
                                    "previous_hidden",
                                    "local_key",
                                    "local_value",
                                    "global_key",
                                    "global_value",
                                ),
                                inputs,
                            )
                        ),
                        position=np.array([committed], dtype="int32"),
                        logits=draft_logits,
                        hidden=previous,
                    )
                    self.captured = True
                last = int(draft_logits.argmax())
                proposals.append(last)
            logits, verify_hidden, *delta = self.forward([frontier, *proposals])
            predicted = logits[0].argmax(axis=-1).tolist()
            match = 0
            for candidate, actual in zip(proposals, predicted):
                if candidate != actual:
                    break
                match += 1
            keep = 1 + match
            rejected = len(proposals) - match
            if rejected:
                self.builtin("popn")(self.cache, 0, rejected)
            states = [np.concatenate([a, b[:, :keep]], axis=1) for a, b in zip(states, delta)]
            hidden = verify_hidden[:, match : match + 1, :]
            frontier = int(predicted[match])
            emitted = proposals[:match] + [frontier]
            for token in emitted:
                output.append(token)
                if not ignore_eos and token in stop:
                    break
            committed += keep
            rounds += 1
            accepted += match
            proposed += len(proposals)
            accepted_by_round.append(match)
        self.device.sync()
        end = time.perf_counter()
        return dict(
            prompt_tokens=len(prompt),
            tokens=output,
            completion_tokens=len(output),
            ttft_s=first - start,
            elapsed_s=end - start,
            decode_s=end - first,
            decode_tok_s=(len(output) - 1) / (end - first),
            rounds=rounds,
            proposed=proposed,
            accepted=accepted,
            accepted_by_round=accepted_by_round,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="dist/gemma4-12b-mtp-target")
    parser.add_argument("--target-lib", default="dist/libs/gemma4-12b-mtp-target-sm72.so")
    parser.add_argument("--assistant", required=True)
    parser.add_argument("--assistant-lib", default="dist/libs/gemma4-12b-mtp-assistant-sm72.so")
    parser.add_argument("--draft-tokens", type=int, choices=(1, 2), default=2)
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

    print("Loading target and official MTP assistant", flush=True)
    target = Program(args.target, args.target_lib)
    assistant = Program(args.assistant, args.assistant_lib)
    trial = Trial(target, assistant, args.capture)
    results = dict(
        args=vars(args),
        prototype="direct VM with CPU state copies",
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
