"""Greedy decoding of a Qwen3.5 target, alone or with a small Qwen3.5 draft (Relax VM).

With a draft, every round the draft proposes a few tokens. The target scores the last
decoded token and the proposals in one `verify_hold` step, keeps the proposals up to its
first disagreement and adds its own next token. `commit_rows` then advances the target's
recurrent state by the kept rows only; the draft goes back through its state history.

One conversation stays in the models between requests. The recurrent state cannot go back,
so a prompt is continued from one of two points: where the last answer ended, or where the
last prompt ended (a copy of the recurrent state; the attention cache drops what follows).
"""

import json
import time

import numpy as np
import tvm_ffi
import tvm_tirx
from mtp_program import release_file_cache
from tvm import relax
from tvm.contrib import tvmjs
from tvm.runtime.vm import VirtualMachine
from tvm_ffi import Shape

LIVE, ANCHOR = 0, 1  # sequences of a recurrent state; the attention cache has LIVE only


def builtin(name):
    return tvm_ffi.get_global_func("vm.builtin." + name)


class Program:
    """One compiled model with an attention cache and a recurrent state."""

    def __init__(self, model, lib, device, history):
        executable = tvm_ffi.load_module(str(lib))
        metadata = json.loads(VirtualMachine(executable, tvm_ffi.device("cpu"))["_metadata"]())
        self.context = metadata["context_window_size"]
        self.chunk = metadata["prefill_chunk_size"]
        self.device = device
        self.mod = relax.VirtualMachine(executable, device).module
        loaded, _ = tvmjs.load_tensor_cache(str(model), device)
        self.params = [loaded[p["name"]] for p in metadata["params"]]
        release_file_cache(model)
        self.cache = self.mod["create_tir_paged_kv_cache"](
            Shape([1]), Shape([self.context]), Shape([self.chunk]), Shape([16]), Shape([0])
        )
        # Single-token steps are kept in `history` slots, which is what `pop` goes back through.
        self.state = self.mod["create_rnn_state"](Shape([2]), Shape([history]))
        self.length = None  # tokens in the attention cache; None without a sequence
        self.anchor = None  # the same when the anchor was taken

    def clear(self):
        if self.length is not None:
            builtin("kv_state_remove_sequence")(self.cache, LIVE)
            builtin("kv_state_remove_sequence")(self.state, LIVE)
        if self.anchor is not None:
            builtin("kv_state_remove_sequence")(self.state, ANCHOR)
        self.length = self.anchor = None

    def start(self):
        self.clear()
        builtin("kv_state_add_sequence")(self.cache, LIVE)
        builtin("kv_state_add_sequence")(self.state, LIVE)
        self.length = 0

    def mark(self):
        """Keep the present recurrent state as the anchor."""
        if self.anchor is not None:
            builtin("kv_state_remove_sequence")(self.state, ANCHOR)
        builtin("kv_state_fork_sequence")(self.state, LIVE, ANCHOR, -1)
        self.anchor = self.length

    def recall(self):
        """Go back to the anchor."""
        if self.length > self.anchor:
            builtin("kv_state_popn")(self.cache, LIVE, self.length - self.anchor)
        builtin("kv_state_remove_sequence")(self.state, LIVE)
        builtin("kv_state_fork_sequence")(self.state, ANCHOR, LIVE, -1)
        self.length = self.anchor

    def forward(self, name, tokens, *extra, advance=True):
        ids = tvm_tirx.runtime.tensor(np.array(tokens, "int32"), device=self.device)
        embeds = self.mod["embed"](ids, self.params)
        embeds = builtin("reshape")(embeds, Shape([1, len(tokens), embeds.shape[1]]))
        builtin("kv_state_begin_forward")(self.cache, Shape([LIVE]), Shape([len(tokens)]))
        builtin("kv_state_begin_forward")(
            self.state, Shape([LIVE]), Shape([len(tokens) if advance else 0])
        )
        result = self.mod[name](embeds, *extra, self.cache, self.state, self.params)
        builtin("kv_state_end_forward")(self.cache)
        builtin("kv_state_end_forward")(self.state)
        self.length += len(tokens)
        return result

    def prefill(self, tokens):
        """The token after `tokens`."""
        for offset in range(0, len(tokens), self.chunk):
            part = tokens[offset : offset + self.chunk]
            last = tvm_tirx.runtime.tensor(np.array([len(part) - 1], "int32"), device=self.device)
            result = self.forward("batch_prefill", part, last)
        return int(result[0].numpy().argmax())

    def decode(self, token):
        """The token after `token`."""
        return int(self.forward("batch_decode", [token])[0].numpy().argmax())

    def verify(self, tokens):
        """The token after each of `tokens`, and what `commit` needs; the state stays."""
        result = self.forward("verify_hold", tokens, advance=False)
        return result[0].numpy().reshape(len(tokens), -1).argmax(axis=1).tolist(), result[3]

    def commit(self, held, rows, drop):
        """Keep the first `rows` rows of the last `verify` and drop the other `drop`."""
        self.pop(drop, state=False)
        builtin("kv_state_begin_forward")(self.state, Shape([LIVE]), Shape([rows]))
        self.mod["commit_rows"](held, Shape([rows]), self.state)
        builtin("kv_state_end_forward")(self.state)

    def pop(self, count, state=True):
        if count:
            builtin("kv_state_popn")(self.cache, LIVE, count)
            if state:
                builtin("kv_state_popn")(self.state, LIVE, count)
            self.length -= count


class Decoder:
    """A target, a draft or none, and the conversation they hold."""

    def __init__(self, target, draft=None, proposals=3):
        assert 1 <= proposals <= 3, "the small-batch kernel verifies at most four rows"
        self.target, self.draft = target, draft
        self.proposals = proposals if draft is not None else 0
        self.context = target.context
        # (tokens read, token after them) where the last answer and the last prompt ended.
        self.live = self.anchor = None

    def programs(self):
        return [self.target] if self.draft is None else [self.target, self.draft]

    def forget(self):
        self.live = self.anchor = None
        for program in self.programs():
            program.clear()

    def resume(self, prompt):
        """Where `prompt` continues from: None (the start), "live" or "anchor"."""
        for name in ("live", "anchor"):
            held = getattr(self, name)
            if held is not None and prompt[: len(held[0])] == held[0]:
                return name
        return None

    def read(self, program, prompt, origin):
        """Bring `program` to the end of `prompt`; the token after it."""
        if origin is None:
            program.start()
            reused, after = 0, None
        else:
            if origin == "anchor":
                program.recall()
            reused, after = len(getattr(self, origin)[0]), getattr(self, origin)[1]
        if reused < len(prompt):
            after = program.prefill(prompt[reused:])
        program.mark()
        return after

    def generate(self, prompt, limit, stops, on_token=lambda token: True, cancelled=lambda: False):
        """At most `limit` tokens after `prompt`, ending with the first token of `stops`.

        `on_token` sees every token but a stop token as soon as the target has decided it,
        and returns whether to go on. Rows under verification need `proposals` free places
        in the context.
        """
        if not prompt or limit < 1 or len(prompt) + limit + self.proposals > self.context:
            raise ValueError("Prompt and answer do not fit the context")
        try:
            return self.run(list(prompt), limit, stops, on_token, cancelled)
        except BaseException:
            # The models may be anywhere between two tokens: nothing is kept.
            self.forget()
            raise

    def run(self, prompt, limit, stops, on_token, cancelled):
        begin = time.perf_counter()
        origin = self.resume(prompt)
        reused = len(getattr(self, origin)[0]) if origin else 0
        output = [self.read(self.target, prompt, origin)]
        first = time.perf_counter()

        def more():
            token = output[-1]
            return token not in stops and on_token(token) and len(output) < limit

        go = more()
        if self.draft is not None:
            self.read(self.draft, prompt, origin)
        self.live = None
        self.anchor = (prompt, output[0])
        rounds = accepted = proposed = 0
        while go and not cancelled():
            # No more proposals than the answer has places left after the next token.
            width = min(self.proposals, limit - len(output) - 1)
            if width < 1:
                output.append(self.target.decode(output[-1]))
                if self.draft is not None:
                    self.draft.decode(output[-2])
                go = more()
                continue
            proposals = []
            for _ in range(width):
                proposals.append(self.draft.decode(proposals[-1] if proposals else output[-1]))
            choice, held = self.target.verify([output[-1], *proposals])
            match = 0
            while match < width and proposals[match] == choice[match]:
                match += 1
            # Rows kept: the last token and the proposals that end up before the new last one.
            rows = 0
            for token in [*proposals[:match], choice[match]]:
                output.append(token)
                rows += 1
                go = more()
                if not go:
                    break
            self.target.commit(held, rows, width + 1 - rows)
            # The draft has read the last token and all proposals but the last.
            if rows > width:
                self.draft.decode(proposals[-1])
            else:
                self.draft.pop(width - rows)
            rounds += 1
            accepted += match
            proposed += width
        self.target.device.sync()
        end = time.perf_counter()
        self.live = (prompt + output[:-1], output[-1])
        return dict(
            tokens=output,
            stopped=output[-1] in stops,
            resumed=origin,
            cached_tokens=reused,
            prefill_tokens=len(prompt) - reused,
            ttft_s=first - begin,
            elapsed_s=end - begin,
            decode_tokens_per_s=(len(output) - 1) / (end - first) if len(output) > 1 else None,
            rounds=rounds,
            accepted=accepted,
            proposed=proposed,
        )
