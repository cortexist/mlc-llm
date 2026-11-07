"""Single-worker MTP generation with a pinned prefix and two reusable conversations."""

import time
from collections import OrderedDict

import numpy as np
import tvm_ffi
from bench_xavier_e4b_mtp import Trial


def common_prefix(left, right, limit):
    for i in range(min(len(left), len(right), limit)):
        if left[i] != right[i]:
            return i
    return min(len(left), len(right), limit)


class CachedTrial(Trial):
    def __init__(
        self, target, assistant, short_assistant, runtime_library, budget=32768, assistants=None
    ):
        super().__init__(target, assistant, runtime_library, max_sequences=4, kv_budget=budget)
        self.short_assistant = short_assistant
        self.short_states = [
            self.tensor(np.zeros((1, 1024, 2, d), dtype="float16")) for d in [256, 256, 512, 512]
        ]
        self.buckets = {
            1024: (short_assistant, self.short_states),
            self.context_length: (assistant, self.states),
        }
        for size, program in (assistants or {}).items():
            self.buckets[size] = (
                program,
                [
                    self.tensor(np.zeros((1, size, 2, d), dtype="float16"))
                    for d in [256, 256, 512, 512]
                ],
            )
        self.budget = budget
        self.entries = OrderedDict()
        self.pinned = None
        self.next_sid = 2
        self.sid = None
        self.committed = 0
        self.replace_sid = None

    def forward(self, ids, prefill=False):
        self.builtin("begin_forward")(
            self.cache, tvm_ffi.Shape([self.sid]), tvm_ffi.Shape([len(ids)])
        )
        try:
            return self.target.call(
                "mtp_prefill" if prefill else "mtp_verify",
                self.tensor(np.array([ids], dtype="int32")),
                self.cache,
            )[:6]
        finally:
            self.builtin("end_forward")(self.cache)

    def append(self, delta, committed, count):
        super().append(delta, committed, count)
        for size, (_, states) in self.buckets.items():
            if size == self.context_length:
                continue
            short_count = min(count, size - committed)
            if short_count > 0:
                for dest, source in zip(states, delta):
                    row_bytes = 4 * dest.shape[-1]
                    self.copy(
                        dest, source, row_bytes * short_count, dest_offset=committed * row_bytes
                    )

    def snapshot(self, tokens, pinned=False):
        n = len(tokens)
        states = []
        for source in self.states:
            dest = self.tensor(np.zeros((1, n, 2, source.shape[-1]), dtype="float16"))
            self.copy(dest, source, n * 4 * source.shape[-1])
            states.append(dest)
        record = dict(sid=self.sid, tokens=list(tokens), states=states)
        if pinned:
            self.pinned = record
        else:
            if self.replace_sid in self.entries:
                self.evict(self.replace_sid)
            while len(self.entries) >= 2:
                self.evict(next(iter(self.entries)))
            self.entries[self.sid] = record
        self.sid = None

    def evict(self, sid):
        self.builtin("remove_sequence")(self.cache, sid)
        del self.entries[sid]

    def prepare(self, prompt, count):
        required = ((len(prompt) + count + 2 + 15) // 16) * 16
        pinned_size = ((len(self.pinned["tokens"]) + 15) // 16) * 16 if self.pinned else 0
        # Forks are independent linear copies in the Xavier runtime. Account for
        # the whole active request, not just its matched prefix, before forking.
        while (
            self.entries
            and required
            + pinned_size
            + sum(((len(e["tokens"]) + 15) // 16) * 16 for e in self.entries.values())
            > self.budget
        ):
            self.evict(next(iter(self.entries)))
        choices = ([self.pinned] if self.pinned else []) + list(self.entries.values())
        chosen = max(
            choices, key=lambda e: common_prefix(e["tokens"], prompt, len(prompt) - 1), default=None
        )
        reuse = common_prefix(chosen["tokens"], prompt, len(prompt) - 1) if chosen else 0
        # A shared policy/header does not make two different chats one session.
        # Replace a cached version only when one complete history is a prefix
        # of the other; retain other conversations and cancelled-request sources.
        self.replace_sid = (
            chosen["sid"]
            if chosen is not None
            and chosen is not self.pinned
            and reuse == min(len(chosen["tokens"]), len(prompt) - 1)
            else None
        )
        self.sid = self.next_sid
        self.next_sid += 1
        if reuse:
            self.builtin("fork_sequence")(self.cache, chosen["sid"], self.sid, reuse)
            self.append(chosen["states"], 0, reuse)
        else:
            self.builtin("add_sequence")(self.cache, self.sid)
        self.committed = reuse
        for offset in range(reuse, len(prompt), 512):
            chunk = prompt[offset : offset + 512]
            logits, hidden, *delta = self.forward(chunk, True)
            self.append(delta, offset, len(chunk))
            self.committed += len(chunk)
        self.copy(self.hidden, hidden, 5120, source_offset=(len(chunk) - 1) * 5120)
        return self.argmax(logits)[0], reuse

    def pin(self, prompt):
        if self.pinned is not None:
            raise RuntimeError("Server prefix is already pinned")
        self.sid = 1
        self.builtin("add_sequence")(self.cache, self.sid)
        for offset in range(0, len(prompt), 512):
            chunk = prompt[offset : offset + 512]
            _, _, *delta = self.forward(chunk, True)
            self.append(delta, offset, len(chunk))
        self.snapshot(prompt, pinned=True)
        self.device.sync()

    def generate(
        self, prompt, count, on_token, cancelled, cache_result=True, ignore_eos=False, drafts=2
    ):
        if not prompt or count < 1 or len(prompt) + count + 2 > self.context_length:
            raise ValueError(
                "Prompt/output exceed the 16K context, including two verification slots"
            )
        output = []
        accepted = proposed = rounds = 0
        reused = 0
        start = time.perf_counter()
        first = start
        try:
            frontier, reused = self.prepare(prompt, count)
            first = time.perf_counter()
            stopped = False

            def emit(token):
                nonlocal stopped
                output.append(token)
                stopped = (not ignore_eos and token in {1, 50, 106}) or not on_token(token)
                return stopped or cancelled.is_set() or len(output) >= count

            done = emit(frontier)
            while not done:
                width = min(drafts, count - len(output) - 1)
                proposals = []
                previous = self.hidden
                last = frontier
                self.position.copyfrom(np.array([self.committed], dtype="int32"))
                assistant, states = self.buckets[
                    min(size for size in self.buckets if size >= self.committed)
                ]
                for _ in range(width):
                    embedding = self.target.call(
                        "embed", self.tensor(np.array([last], dtype="int32"))
                    )
                    embedding = self.reshape(embedding, tvm_ffi.Shape([1, 1, 2560]))
                    logits, previous, ids = assistant.call(
                        "mtp_step", embedding, previous, *states, self.position, self.position
                    )
                    last = self.argmax(logits, ids)[0]
                    proposals.append(last)
                logits, hidden, *delta = self.forward([frontier, *proposals])
                predicted = self.argmax(logits)
                match = 0
                for candidate, actual in zip(proposals, predicted):
                    if candidate != actual:
                        break
                    match += 1
                keep = match + 1
                if width > match:
                    self.builtin("popn")(self.cache, self.sid, width - match)
                self.append(delta, self.committed, keep)
                self.copy(self.hidden, hidden, 5120, source_offset=match * 5120)
                self.committed += keep
                frontier = predicted[match]
                accepted += match
                proposed += width
                rounds += 1
                for token in [*proposals[:match], frontier]:
                    if emit(token):
                        done = True
                        break
            end = time.perf_counter()
            # Early EOS/stop strings can terminate inside an accepted block.
            # Store only the consumed history, excluding its unverified frontier.
            retain = min(self.committed, len(prompt) + len(output) - 1)
            if self.committed > retain:
                self.builtin("popn")(self.cache, self.sid, self.committed - retain)
            if cache_result and not cancelled.is_set():
                self.snapshot((prompt + output)[:retain])
            self.device.sync()
            return dict(
                tokens=output,
                prompt_tokens=len(prompt),
                completion_tokens=len(output),
                cached_tokens=reused,
                prefill_tokens=len(prompt) - reused,
                ttft_s=first - start,
                elapsed_s=end - start,
                decode_tokens_per_s=(len(output) - 1) / max(end - first, 1e-9),
                mtp_accepted=accepted,
                mtp_proposed=proposed,
                mtp_rounds=rounds,
                stopped=stopped,
                cancelled=cancelled.is_set(),
            )
        finally:
            if self.sid is not None:
                self.builtin("remove_sequence")(self.cache, self.sid)
                self.sid = None
            self.device.sync()
