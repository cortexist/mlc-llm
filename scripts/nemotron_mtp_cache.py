"""Single-worker MTP generation with a pinned prefix and two reusable conversations.

A scratch request (a one-off, such as a routing decision) has a place of its own: the
next one starts from it, and it never puts a conversation out.
"""

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
    # Only the last prompt token is predicted from. With this set, the others run
    # the layers that store K/V only (mtp_prefill_kv), and the last one a
    # one-row verification step.
    kv_only_prefill = False
    scratch = None  # the last scratch request, held like an entry

    def __init__(
        self,
        target,
        assistant,
        short_assistant,
        runtime_library,
        budget=32768,
        assistants=None,
        **shapes,
    ):
        # Sequences: the pinned prefix, two conversations, a scratch request, the active one.
        super().__init__(
            target, assistant, runtime_library, max_sequences=5, kv_budget=budget, **shapes
        )
        # Assistants compiled for shorter contexts; none when the assistant
        # attends the valid context only (MLC_E4B_MTP_ATTN=1).
        self.buckets = {self.context_length: (assistant, self.states)}
        shorter = {1024: short_assistant} if short_assistant else {}
        for size, program in {**shorter, **(assistants or {})}.items():
            self.buckets[size] = (
                program,
                [
                    self.tensor(np.zeros((1, size, *state.shape[2:]), dtype="float16"))
                    for state in self.states
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

    def fill(self, ids):
        self.builtin("begin_forward")(
            self.cache, tvm_ffi.Shape([self.sid]), tvm_ffi.Shape([len(ids)])
        )
        try:
            return self.target.call(
                "mtp_prefill_kv", self.tensor(np.array([ids], dtype="int32")), self.cache
            )[1:5]
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
                    row = self.row_bytes(dest)
                    self.copy(dest, source, row * short_count, dest_offset=committed * row)

    def snapshot(self, tokens, pinned=False, scratch=False):
        n = len(tokens)
        states = []
        for source in self.states:
            dest = self.tensor(np.zeros((1, n, *source.shape[2:]), dtype="float16"))
            self.copy(dest, source, n * self.row_bytes(source))
            states.append(dest)
        record = dict(sid=self.sid, tokens=list(tokens), states=states)
        if pinned:
            self.pinned = record
        elif scratch:
            self.drop_scratch()
            self.scratch = record
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

    def drop_scratch(self):
        if self.scratch is not None:
            self.builtin("remove_sequence")(self.cache, self.scratch["sid"])
            self.scratch = None

    def stored(self):
        return [*self.entries.values(), *([self.scratch] if self.scratch else [])]

    def prepare(self, prompt, count):
        required = ((len(prompt) + count + 2 + 15) // 16) * 16
        pinned_size = ((len(self.pinned["tokens"]) + 15) // 16) * 16 if self.pinned else 0
        # Forks are independent linear copies in the Xavier runtime. Account for
        # the whole active request, not just its matched prefix, before forking.
        while (
            self.stored()
            and required
            + pinned_size
            + sum(((len(e["tokens"]) + 15) // 16) * 16 for e in self.stored())
            > self.budget
        ):
            if self.scratch:
                self.drop_scratch()
            else:
                self.evict(next(iter(self.entries)))
        choices = ([self.pinned] if self.pinned else []) + self.stored()
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
            and chosen["sid"] in self.entries
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
        if self.kv_only_prefill:
            for offset in range(reuse, len(prompt) - 1, 512):
                chunk = prompt[offset : min(offset + 512, len(prompt) - 1)]
                self.append(self.fill(chunk), offset, len(chunk))
                self.committed += len(chunk)
            logits, hidden, *delta = self.forward(prompt[-1:])
            self.append(delta, self.committed, 1)
            self.committed += 1
            self.keep(hidden, 0)
            return self.argmax(logits)[0], reuse
        for offset in range(reuse, len(prompt), 512):
            chunk = prompt[offset : offset + 512]
            logits, hidden, *delta = self.forward(chunk, True)
            self.append(delta, offset, len(chunk))
            self.committed += len(chunk)
        self.keep(hidden, len(chunk) - 1)
        return self.argmax(logits)[0], reuse

    def pin(self, prompt):
        if self.pinned is not None:
            raise RuntimeError("Server prefix is already pinned")
        self.sid = 1
        self.builtin("add_sequence")(self.cache, self.sid)
        for offset in range(0, len(prompt), 512):
            chunk = prompt[offset : offset + 512]
            delta = self.fill(chunk) if self.kv_only_prefill else self.forward(chunk, True)[2:]
            self.append(delta, offset, len(chunk))
        self.snapshot(prompt, pinned=True)
        self.device.sync()

    def generate(
        self,
        prompt,
        count,
        on_token,
        cancelled,
        cache_result=True,
        ignore_eos=False,
        drafts=2,
        scratch=False,
    ):
        if not prompt or count < 1 or len(prompt) + count + 2 > self.context_length:
            raise ValueError("Prompt/output exceed the context, including two verification slots")
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
                    last, previous = self.draft(last, previous, assistant, states)
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
                self.keep(hidden, match)
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
                self.snapshot((prompt + output)[:retain], scratch=scratch)
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
