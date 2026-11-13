"""CPU checks for MTP request boundaries and draining the shared GPU queue."""

import asyncio
import sys
import threading
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from nemotron_mtp_cache import CachedTrial, common_prefix
from nemotron_mtp_service import NemotronMTP
from xavier_serve_queue import GenerationQueue, SharedGenerationQueue

from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.protocol.openai_api_protocol import CompletionRequest


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.engine = NemotronMTP.__new__(NemotronMTP)

    def test_verification_reserve_and_invalid_ids(self):
        good = CompletionRequest(prompt=[2] * 16380, max_tokens=2, temperature=0)
        self.assertEqual(self.engine._prepare(good, False)[1], 2)
        for request in [
            good.model_copy(update={"max_tokens": 3}),
            CompletionRequest(prompt=[]),
            CompletionRequest(prompt=[262144]),
            CompletionRequest(prompt=[-1]),
            CompletionRequest(prompt=[2], max_tokens=0),
        ]:
            with self.assertRaises(BadRequestError):
                self.engine._prepare(request, False)

    def test_verification_reserve_follows_the_proposals(self):
        self.engine.draft_tokens = 3
        good = CompletionRequest(prompt=[2] * 16380, max_tokens=1, temperature=0)
        self.assertEqual(self.engine._prepare(good, False)[1], 1)
        with self.assertRaisesRegex(BadRequestError, "3 verification slots"):
            self.engine._prepare(good.model_copy(update={"max_tokens": 2}), False)

    def test_unsupported_generation_controls_fail_explicitly(self):
        for kw in [
            {"temperature": 1},
            {"n": 2},
            {"frequency_penalty": 1},
            {"logit_bias": {12: 2}},
            {"logprobs": 3},
            {"echo": True},
        ]:
            with self.subTest(kw=kw), self.assertRaises(BadRequestError):
                self.engine._prepare(CompletionRequest(prompt=[2], **kw), False)

    def test_prefix_requires_matching_tokens_and_one_prefill_token(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 3], 2), 2)
        self.assertEqual(common_prefix([1, 2, 3], [1, 4, 3], 2), 1)
        self.assertEqual(common_prefix([1], [2], 1), 0)

    def test_shared_header_retains_both_conversations_but_extension_replaces_old_version(self):
        for prompt, expected in [([2, 100, 201, 9], [10, 12]), ([2, 100, 200, 4, 5, 6], [11, 12])]:
            trial = CachedTrial.__new__(CachedTrial)
            trial.pinned = {"sid": 1, "tokens": [2], "states": []}
            trial.entries = OrderedDict(
                [(10, {"sid": 10, "tokens": [2, 100, 200, 4, 5], "states": []})]
            )
            if prompt[-1] == 6:
                trial.entries[11] = {"sid": 11, "tokens": [2, 100, 201, 9], "states": []}
            trial.budget = 32768
            trial.next_sid = 12
            trial.cache = None
            trial.builtin = lambda name: lambda *args: None
            trial.append = lambda *args: None
            trial.forward = lambda *args: (None, None)
            trial.keep = lambda *args: None
            trial.argmax = lambda *args: [7]
            trial.hidden = None
            trial.states = []
            trial.prepare(prompt, 4)
            trial.snapshot(prompt)
            self.assertEqual(list(trial.entries), expected)
            self.assertEqual(trial.pinned["sid"], 1)

    def test_scratch_request_starts_from_the_last_one_and_displaces_no_conversation(self):
        removed = []
        trial = CachedTrial.__new__(CachedTrial)
        trial.pinned = {"sid": 1, "tokens": [2], "states": []}
        trial.entries = OrderedDict([(10, {"sid": 10, "tokens": [2, 100, 4, 5], "states": []})])
        trial.budget, trial.next_sid, trial.cache, trial.hidden = 32768, 12, None, None
        trial.builtin = lambda name: lambda *args: (
            removed.append(args[1]) if name == "remove_sequence" else None
        )
        trial.append = lambda *args: None
        trial.forward = lambda *args: (None, None)
        trial.keep = lambda *args: None
        trial.argmax = lambda *args: [7]
        trial.states = []

        def run(prompt, scratch=False):
            reuse = trial.prepare(prompt, 4)[1]
            trial.snapshot(prompt, scratch=scratch)
            return reuse

        # Two scratch requests in a row: the second starts from the first and takes
        # its place; the conversation stays.
        self.assertEqual(run([2, 300, 301, 8], scratch=True), 1)
        self.assertEqual(run([2, 300, 301, 9], scratch=True), 3)
        self.assertEqual((list(trial.entries), trial.scratch["sid"], removed), ([10], 13, [12]))
        # The conversation goes on from its own entry, and a second one is kept beside it.
        self.assertEqual(run([2, 100, 4, 5, 6, 7]), 4)
        self.assertEqual(run([2, 100, 9, 9]), 2)
        self.assertEqual(
            (list(trial.entries), trial.scratch["sid"], removed), ([14, 15], 13, [12, 10])
        )
        # With both conversations kept, a scratch request still starts from the last one.
        self.assertEqual(run([2, 300, 301, 7], scratch=True), 3)
        self.assertEqual(
            (list(trial.entries), trial.scratch["sid"], removed[-1]), ([14, 15], 16, 13)
        )
        # A third conversation puts out the oldest conversation, not the scratch request.
        self.assertEqual(run([2, 200, 1, 1]), 1)
        self.assertEqual(
            (list(trial.entries), trial.scratch["sid"], removed[-1]), ([15, 17], 16, 14)
        )
        # Short of K/V slots, the scratch request goes first.
        trial.budget = 64
        self.assertEqual(run([2, 200, 1, 1, 3]), 4)
        self.assertEqual((list(trial.entries), trial.scratch), ([15, 18], None))

    def test_kv_only_prefill_runs_the_whole_model_only_where_a_token_is_predicted(self):
        prompt = [2] * 1200
        for mode, kv_only, whole in [
            (False, [], [(512, True), (512, True), (176, True)]),
            (True, [512, 512, 175], [(1, False)]),
        ]:
            trial = CachedTrial.__new__(CachedTrial)
            trial.kv_only_prefill = mode
            trial.pinned, trial.entries = None, OrderedDict()
            trial.budget, trial.next_sid, trial.cache, trial.hidden = 32768, 12, None, None
            trial.builtin = lambda name: lambda *args: None
            trial.append = lambda *args: None
            trial.keep = lambda *args: None
            trial.argmax = lambda *args: [7]
            calls = ([], [])
            trial.fill = lambda ids, calls=calls: calls[0].append(len(ids))
            trial.forward = lambda ids, prefill=False, calls=calls: (
                calls[1].append((len(ids), prefill)),
                None,
            )
            with self.subTest(mode=mode):
                self.assertEqual(trial.prepare(prompt, 4), (7, 0))
                self.assertEqual(calls, (kv_only, whole))
                self.assertEqual(trial.committed, len(prompt))


class Drain(unittest.IsolatedAsyncioTestCase):
    async def test_response_cancellation_cannot_release_gpu_before_worker_exits(self):
        engine = NemotronMTP.__new__(NemotronMTP)
        pool = ThreadPoolExecutor(max_workers=1)
        started, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()

        def worker():
            started.set()
            release.wait(5)

        future = loop.run_in_executor(pool, worker)
        engine.active = (cancelled, future)
        queue = GenerationQueue()
        queue.drain = engine.drain
        response_started = asyncio.Event()
        second_started = asyncio.Event()

        async def backend(scope, receive, send):
            if scope.get("second"):
                second_started.set()
            else:
                response_started.set()
                await asyncio.Event().wait()

        app = SharedGenerationQueue(backend, queue, "key")
        scope = dict(
            type="http",
            method="POST",
            path="/v1/chat/completions",
            headers=[(b"authorization", b"Bearer key")],
        )
        first = asyncio.create_task(app(scope, None, None))
        try:
            await response_started.wait()
            first.cancel()
            second = asyncio.create_task(app(dict(scope, second=True), None, None))
            await asyncio.sleep(0.02)
            self.assertTrue(cancelled.is_set())
            self.assertFalse(second_started.is_set())
            self.assertFalse(first.done())
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)
            self.assertTrue(second_started.is_set())
            self.assertFalse(queue.lock.locked())
        finally:
            release.set()
            pool.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
