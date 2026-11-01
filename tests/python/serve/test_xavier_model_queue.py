"""CPU checks for cross-model serialization and uncancellable CUDA work."""

import asyncio
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from gemma4_mtp_service import MTPChat
from xavier_serve_queue import GenerationQueue, SharedGenerationQueue


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_lifetime_serializes_both_models_and_auth_skips_queue(self):
        queue = GenerationQueue()
        started, release = asyncio.Event(), asyncio.Event()
        sequence, responses = [], []

        async def backend(scope, receive, send):
            model = scope["model"]
            sequence.append((model, "start"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if model == "nemotron":
                started.set()
                await release.wait()
            await send({"type": "http.response.body", "body": b"done"})
            sequence.append((model, "end"))

        app = SharedGenerationQueue(backend, queue, "test-key")

        async def call(model, key="Bearer test-key"):
            async def send(event):
                responses.append(event)

            await app(
                dict(
                    type="http",
                    method="POST",
                    path="/v1/chat/completions",
                    model=model,
                    headers=[(b"authorization", key.encode())],
                ),
                None,
                send,
            )

        first = asyncio.create_task(call("nemotron"))
        await started.wait()
        second = asyncio.create_task(call("gemma4-12b"))
        await asyncio.sleep(0.01)
        self.assertEqual(queue.waiting, 1)
        await asyncio.wait_for(call("invalid", "bad-key"), 1)
        self.assertEqual(responses[-2]["status"], 401)
        self.assertEqual(sequence, [("nemotron", "start")])
        release.set()
        await asyncio.gather(first, second)
        self.assertEqual(
            sequence,
            [
                ("nemotron", "start"),
                ("nemotron", "end"),
                ("gemma4-12b", "start"),
                ("gemma4-12b", "end"),
            ],
        )
        self.assertEqual(queue.completed, 2)

    async def test_cancelled_thread_work_retains_gpu_lock_until_finished(self):
        service = MTPChat.__new__(MTPChat)
        service.pool = ThreadPoolExecutor(max_workers=1)
        started, release = threading.Event(), threading.Event()
        queue = GenerationQueue()
        next_started = asyncio.Event()

        def gpu_work():
            started.set()
            release.wait(timeout=3)

        async def first_request():
            async with queue.lock:
                await service.work(gpu_work)

        async def next_request():
            async with queue.lock:
                next_started.set()

        first = asyncio.create_task(first_request())
        try:
            await asyncio.wait_for(asyncio.to_thread(started.wait), 2)
            first.cancel()
            second = asyncio.create_task(next_request())
            await asyncio.sleep(0.02)
            self.assertFalse(next_started.is_set())
            self.assertFalse(first.done())
            release.set()
            result = await asyncio.gather(first, second, return_exceptions=True)
            self.assertIsInstance(result[0], asyncio.CancelledError)
            self.assertTrue(next_started.is_set())
        finally:
            release.set()
            service.pool.shutdown(wait=True)

    async def test_cancelled_waiter_does_not_leak_queue_or_release_current_lock(self):
        queue = GenerationQueue()
        await queue.lock.acquire()
        app = SharedGenerationQueue(None, queue, "key")
        waiter = asyncio.create_task(
            app(
                dict(
                    type="http",
                    method="POST",
                    path="/v1/chat/completions",
                    headers=[(b"authorization", b"Bearer key")],
                ),
                None,
                None,
            )
        )
        await asyncio.sleep(0.01)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        self.assertEqual(queue.waiting, 0)
        self.assertTrue(queue.lock.locked())
        queue.lock.release()


class BudgetTest(unittest.TestCase):
    def test_verification_slots_are_reserved_before_generation(self):
        service = MTPChat.__new__(MTPChat)
        service.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [2] * 1020)
        request = SimpleNamespace(
            temperature=0,
            n=1,
            tools=None,
            tool_choice=None,
            logprobs=False,
            response_format=None,
            logit_bias=None,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            debug_config=None,
            max_tokens=3,
            messages=[SimpleNamespace(role="user", content="hello", tool_calls=None)],
        )
        with self.assertRaisesRegex(ValueError, "2 verification slots"):
            service.prepare(request)
        request.max_tokens = 2
        self.assertEqual(service.prepare(request)[1], 2)


if __name__ == "__main__":
    unittest.main()
