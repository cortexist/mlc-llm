"""Serialize GPU generations, including the entire streamed response lifetime."""

import asyncio
import secrets
import time

import anyio


class GenerationQueue:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.waiting = 0
        self.active_model = None
        self.completed = 0
        self.drain = None

    def status(self):
        return dict(active_model=self.active_model, waiting=self.waiting, completed=self.completed)


class SharedGenerationQueue:
    def __init__(self, app, queue, api_key):
        self.app, self.queue = app, queue
        self.authorization = ("Bearer " + api_key).encode()

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path")
            not in (
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/decisions",
                "/v1/systemone",
            )
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if not secrets.compare_digest(headers.get(b"authorization", b""), self.authorization):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            return await send(
                {"type": "http.response.body", "body": b'{"error":{"message":"Invalid API key"}}'}
            )
        queued = time.monotonic()
        self.queue.waiting += 1
        try:
            await self.queue.lock.acquire()
        finally:
            self.queue.waiting -= 1
        wait_ms = str(round((time.monotonic() - queued) * 1000)).encode()

        async def respond(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-xavier-queue-wait-ms", wait_ms),
                ]
            await send(message)

        try:
            await self.app(scope, receive, respond)
        finally:
            try:
                if self.queue.drain is not None:
                    with anyio.CancelScope(shield=True):
                        await self.queue.drain()
            finally:
                self.queue.active_model = None
                self.queue.completed += 1
                self.queue.lock.release()
