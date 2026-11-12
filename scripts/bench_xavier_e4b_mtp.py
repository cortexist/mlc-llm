"""Greedy MTP controller retaining shared KV and hidden states on CUDA.

This remains a direct-VM benchmark, not an MLCEngine serving mode. Compile
xavier_mtp_runtime.cu with nvcc -O3 -shared -Xcompiler -fPIC -arch=sm_72.
Only the selected token IDs cross back to CPU each round. The defaults are
the E4B shapes; bench_xavier_mtp.py sets those of the Unified 12B.
"""

import ctypes
import time

import numpy as np
import tvm_ffi
import tvm_tirx
from mtp_program import Program as Program


class Trial:
    def __init__(
        self,
        target,
        assistant,
        runtime_library,
        max_sequences=1,
        kv_budget=None,
        hidden_size=2560,
        kv_heads=(2, 2),
    ):
        """kv_heads: K/V heads the assistant shares from local and from global layers."""
        self.target, self.assistant = target, assistant
        self.context_length = min(target.context_length, assistant.context_length)
        self.device = tvm_tirx.cuda(0)
        self.builtin = lambda name: tvm_ffi.get_global_func("vm.builtin.kv_state_" + name)
        self.cache = target.vm["create_tir_paged_kv_cache"](
            *[
                tvm_ffi.Shape([n])
                for n in (max_sequences, kv_budget or self.context_length, 512, 16, 0)
            ]
        )
        self.reshape = tvm_ffi.get_global_func("vm.builtin.reshape")
        self.cuda = ctypes.CDLL("libcudart.so")
        self.cuda.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.runtime = ctypes.CDLL(str(runtime_library))
        self.runtime.mtp_argmax.argtypes = (
            [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        )
        self.partial = self.tensor(np.zeros((4, 256), dtype="float32"))
        self.indices = self.tensor(np.zeros((4, 256), dtype="int32"))
        self.predictions = {n: self.tensor(np.zeros(n, dtype="int32")) for n in [1, 2, 3, 4]}
        local, shared = kv_heads
        self.states = [
            self.tensor(np.zeros((1, self.context_length, heads, d), dtype="float16"))
            for heads, d in [(local, 256), (local, 256), (shared, 512), (shared, 512)]
        ]
        self.hidden = self.tensor(np.zeros((1, 1, hidden_size), dtype="float16"))
        self.position = self.tensor(np.zeros(1, dtype="int32"))

    def tensor(self, value):
        return tvm_tirx.runtime.tensor(np.ascontiguousarray(value), self.device)

    def copy(self, dest, source, count, dest_offset=0, source_offset=0):
        status = self.cuda.cudaMemcpyAsync(
            dest.data_ptr() + dest_offset,
            source.data_ptr() + source_offset,
            count,
            3,
            tvm_ffi.get_raw_stream(self.device),
        )
        if status:
            raise RuntimeError(f"CUDA device copy failed: {status}")

    def argmax(self, logits, ids=None):
        rows = int(np.prod(logits.shape[:-1]))
        out = self.predictions[rows]
        status = self.runtime.mtp_argmax(
            logits.data_ptr(),
            self.partial.data_ptr(),
            self.indices.data_ptr(),
            ids.data_ptr() if ids is not None else None,
            out.data_ptr(),
            rows,
            logits.shape[-1],
            tvm_ffi.get_raw_stream(self.device),
        )
        if status:
            raise RuntimeError(f"CUDA argmax failed: {status}")
        return out.numpy().tolist()

    def forward(self, ids, prefill=False):
        self.builtin("begin_forward")(self.cache, tvm_ffi.Shape([0]), tvm_ffi.Shape([len(ids)]))
        result = self.target.call(
            "mtp_prefill" if prefill else "mtp_verify",
            self.tensor(np.array([ids], dtype="int32")),
            self.cache,
        )
        self.builtin("end_forward")(self.cache)
        return result[:6]

    @staticmethod
    def row_bytes(state):
        """Bytes of one token in a shared K/V state [1, tokens, heads, width]."""
        return state.shape[2] * state.shape[3] * 2

    def append(self, delta, committed, count):
        for dest, source in zip(self.states, delta):
            size = self.row_bytes(dest)
            self.copy(dest, source, size * count, dest_offset=committed * size)

    def keep(self, hidden, row):
        """Retain the hidden state of one row for the next draft."""
        size = self.hidden.shape[-1] * 2
        self.copy(self.hidden, hidden, size, source_offset=row * size)

    def draft(self, token, previous, assistant=None, states=None):
        """One assistant step after `token`: the proposed token and the next hidden state.

        Set self.position first; it is fixed for the round.
        """
        embedding = self.target.call("embed", self.tensor(np.array([token], dtype="int32")))
        embedding = self.reshape(embedding, tvm_ffi.Shape(self.hidden.shape))
        # An assistant with a centroid-filtered head also returns the token IDs of its logits.
        logits, previous, *ids = (assistant or self.assistant).call(
            "mtp_step", embedding, previous, *(states or self.states), self.position, self.position
        )
        return self.argmax(logits, *ids)[0], previous

    def run(self, prompt, count, drafts, ignore_eos=False):
        if drafts not in [0, 1, 2, 3] or count < 1 or not prompt:
            raise ValueError("Expected nonempty prompt, positive output count and 0..3 drafts")
        if len(prompt) + count + drafts > self.context_length:
            raise ValueError("Request plus verification exceeds compiled context")
        self.builtin("clear")(self.cache)
        self.builtin("add_sequence")(self.cache, 0)
        self.device.sync()
        start = time.perf_counter()
        for offset in range(0, len(prompt), 512):
            chunk = prompt[offset : offset + 512]
            logits, hidden, *delta = self.forward(chunk, True)
            self.append(delta, offset, len(chunk))
        self.keep(hidden, len(chunk) - 1)
        frontier = self.argmax(logits)[0]
        first = time.perf_counter()
        output = [frontier]
        committed = len(prompt)
        rounds = proposed = accepted = 0
        accepted_by_round = []
        stop = {1, 50, 106}
        while len(output) < count and (ignore_eos or output[-1] not in stop):
            width = min(drafts, count - len(output) - 1)
            proposals = []
            previous = self.hidden
            last = frontier
            self.position.copyfrom(np.array([committed], dtype="int32"))
            for _ in range(width):
                last, previous = self.draft(last, previous)
                proposals.append(last)
            logits, hidden, *delta = self.forward([frontier, *proposals])
            predicted = self.argmax(logits)
            match = 0
            for candidate, actual in zip(proposals, predicted):
                if candidate != actual:
                    break
                match += 1
            keep = match + 1
            if width - match:
                self.builtin("popn")(self.cache, 0, width - match)
            self.append(delta, committed, keep)
            self.keep(hidden, match)
            frontier = predicted[match]
            for token in [*proposals[:match], frontier]:
                output.append(token)
                if not ignore_eos and token in stop:
                    break
            committed += keep
            rounds += 1
            accepted += match
            proposed += width
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
