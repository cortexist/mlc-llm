"""OpenAI engine adapter for resident Nemotron with Gemma E4B MTP drafts.

Libraries are named under dist/libs. By default: the target, and assistants
compiled for 1K to 16K contexts. Options, all for libraries built to match:

MLC_NEMOTRON_MTP_TARGET        target library
MLC_NEMOTRON_MTP_KV_PREFILL    1: K/V-only prefill, the target has mtp_prefill_kv (see CachedTrial)
MLC_NEMOTRON_MTP_ASSISTANT     one assistant that attends the valid context only
MLC_NEMOTRON_MTP_DRAFTS        proposals of a round (2); 3 needs a target built with
                               MLC_MTP_VERIFY_LEN=4
"""

import asyncio
import contextvars
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing
from pathlib import Path

import anyio

from mlc_llm.protocol.conversation_protocol import Conversation
from mlc_llm.protocol.error_protocol import BadRequestError
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionUsage,
)
from mlc_llm.serve.engine_utils import process_prompts
from mlc_llm.tokenizers import StopStrHandler, TextStreamer, Tokenizer

# Set by the server for a one-off request (X-Xavier-Scratch: 1), which the cache keeps
# apart from the two conversations.
scratch_request = contextvars.ContextVar("scratch_request", default=False)


class NemotronMTP:
    context_length = 16384
    draft_tokens = int(os.environ.get("MLC_NEMOTRON_MTP_DRAFTS", "2"))

    def __init__(self, root):
        self.root = Path(root)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron-mtp")
        self.active = None
        try:
            self.pool.submit(self._load).result()
        except BaseException:
            self.pool.shutdown(wait=True)
            raise

    def _load(self):
        from mtp_program import Program
        from nemotron_mtp_cache import CachedTrial

        libs, option = self.root / "dist/libs", os.environ.get
        target = Program(
            self.root / "dist/nemotron-e4b-mtp-16k",
            libs / option("MLC_NEMOTRON_MTP_TARGET", "nemotron-e4b-mtp-target-16k-sm72.so"),
        )
        single = option("MLC_NEMOTRON_MTP_ASSISTANT")
        assistant = Program(
            self.root / "dist/gemma4-e4b-mtp-assistant-16k",
            libs / (single or "gemma4-e4b-mtp-assistant-topk-16k-sm72.so"),
        )
        short, buckets = None, {}
        if not single:
            short = Program(
                Path.home() / "Models/gemma-4-E4B-IT-assistant-q0f16-MLC",
                libs / "gemma4-e4b-mtp-assistant-topk-1k-sm72.so",
                share_params_from=assistant,
            )
            buckets = {
                size: Program(
                    self.root / f"dist/gemma4-e4b-mtp-assistant-{size // 1024}k",
                    libs / f"gemma4-e4b-mtp-assistant-topk-{size // 1024}k-sm72.so",
                    share_params_from=assistant,
                )
                for size in [2048, 4096, 8192]
            }
        self.trial = CachedTrial(
            target,
            assistant,
            short,
            libs / "xavier-mtp-runtime.so",
            int(option("MLC_NEMOTRON_KV_BUDGET", "32768")),
            buckets,
        )
        kv_only = option("MLC_NEMOTRON_MTP_KV_PREFILL", "0")
        if kv_only not in ("0", "1"):
            raise ValueError("MLC_NEMOTRON_MTP_KV_PREFILL must be 0 or 1")
        self.trial.kv_only_prefill = kv_only == "1"
        if self.trial.context_length != self.context_length:
            raise RuntimeError("Nemotron MTP libraries must support 16K context")
        self.tokenizer = Tokenizer(str(self.root / "dist/target-16k"))
        self.config = json.loads((self.root / "dist/target-16k/mlc-chat-config.json").read_text())

    async def work(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(self.pool, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                await asyncio.shield(future)
            raise

    def _warm(self, prefix):
        # Long enough for one verification step of every row.
        self.trial.generate(
            [100] * 512,
            self.draft_tokens + 2,
            lambda _: True,
            threading.Event(),
            cache_result=False,
            drafts=self.draft_tokens,
        )
        if prefix:
            self.trial.pin(prefix)

    async def warm_and_pin(self, prefix):
        await self.work(self._warm, prefix)

    def _prepare(self, request, chat):
        if request.temperature not in (None, 0) or request.n != 1:
            raise BadRequestError("Nemotron MTP supports temperature=0 and n=1")
        if (
            request.logprobs
            or request.logit_bias
            or request.frequency_penalty
            or request.presence_penalty
            or request.response_format is not None
            or getattr(request, "best_of", 1) != 1
            or getattr(request, "echo", False)
            or getattr(request, "suffix", None)
        ):
            raise BadRequestError(
                "MTP does not support penalties, logit bias, logprobs, echo, suffix "
                "or grammar constraints"
            )
        stop = request.stop or []
        if isinstance(stop, str):
            stop = [stop]
        if chat:
            request.check_message_validity()
            conv = Conversation.model_validate(self.config["conv_template"])
            for message in request.messages:
                content = message.content
                if isinstance(content, list):
                    if any(x.get("type") != "text" for x in content):
                        raise BadRequestError("Nemotron MTP supports text messages only")
                    content = "\n".join(x["text"] for x in content)
                if message.role == "system":
                    conv.system_message = content or ""
                else:
                    conv.messages.append((message.role, content))
            conv.messages.append(("assistant", None))
            parts = process_prompts(conv.as_prompt(self.config), self.tokenizer.encode)
            prompt = (conv.system_prefix_token_ids or []) + [t for part in parts for t in part]
            stop = [*conv.stop_str, *stop]
        else:
            prompt = (
                self.tokenizer.encode(request.prompt)
                if isinstance(request.prompt, str)
                else list(request.prompt)
            )
        if not prompt or any(t < 0 or t >= 262144 for t in prompt):
            raise BadRequestError("A nonempty prompt with valid token IDs is required")
        remaining = self.context_length - len(prompt) - self.draft_tokens
        wanted = 1 if request.max_tokens is None else request.max_tokens
        if wanted < 1:
            raise BadRequestError("max_tokens must be at least 1")
        if wanted > remaining:
            # Clients that read this wording shorten the conversation and ask again.
            raise BadRequestError(
                f"This model's maximum context length is {self.context_length} tokens. However, "
                f"you requested {len(prompt) + wanted + self.draft_tokens} tokens ({len(prompt)} "
                f"in the messages, {wanted} in the completion, {self.draft_tokens} verification "
                "slots). Please reduce the length of the messages or completion."
            )
        count = min(512, remaining) if request.max_tokens is None else wanted
        return prompt, count, stop

    async def _events(self, request, chat):
        prompt, count, stops = await self.work(self._prepare, request, chat)
        scratch = scratch_request.get()
        loop = asyncio.get_running_loop()
        events = asyncio.Queue()
        cancelled = threading.Event()

        def send(kind, value):
            loop.call_soon_threadsafe(events.put_nowait, (kind, value))

        def producer():
            try:
                streamer = TextStreamer(self.tokenizer)
                stopper = StopStrHandler(stops, self.tokenizer)

                def token(value):
                    if cancelled.is_set():
                        return False
                    text = streamer.put(stopper.put(value))
                    if text:
                        send("text", text)
                    return not stopper.stop_triggered

                result = self.trial.generate(
                    prompt, count, token, cancelled, drafts=self.draft_tokens, scratch=scratch
                )
                tail = streamer.put(stopper.finish()) + streamer.finish()
                if tail:
                    send("text", tail)
                finish = "stop" if result["stopped"] or stopper.stop_triggered else "length"
                send("finish", finish)
                usage = CompletionUsage(
                    prompt_tokens=len(prompt),
                    completion_tokens=result["completion_tokens"],
                    total_tokens=len(prompt) + result["completion_tokens"],
                    extra={
                        k: v
                        for k, v in result.items()
                        if k not in ["tokens", "stopped", "cancelled"]
                    },
                )
                send("usage", usage)
                print(
                    json.dumps({"event": "nemotron_mtp_request", **usage.model_dump()}), flush=True
                )
            except BaseException as exc:
                send("error", exc)
            finally:
                send("end", None)

        future = loop.run_in_executor(self.pool, producer)
        self.active = (cancelled, future)
        try:
            while True:
                kind, value = await events.get()
                if kind == "error":
                    raise value
                if kind == "end":
                    break
                yield kind, value
        finally:
            cancelled.set()
            # Hold the shared FIFO queue until outstanding GPU work has drained.
            # A cancelled SSE connection must never overlap the next specialist.
            with anyio.CancelScope(shield=True):
                await asyncio.shield(future)
            if self.active is not None and self.active[1] is future:
                self.active = None

    async def _handle_completion(
        self, request, request_id, request_final_usage_include_extra=False
    ):
        async with aclosing(self._events(request, False)) as events:
            async for kind, value in events:
                if kind == "usage":
                    if request.stream and not (
                        request.stream_options and request.stream_options.include_usage
                    ):
                        continue
                    if not request_final_usage_include_extra:
                        value.extra = None
                choices = (
                    []
                    if kind == "usage"
                    else [
                        CompletionResponseChoice(
                            text=value if kind == "text" else "",
                            finish_reason=value if kind == "finish" else None,
                        )
                    ]
                )
                yield CompletionResponse(
                    id=request_id,
                    model=request.model,
                    choices=choices,
                    usage=value if kind == "usage" else None,
                )

    async def _handle_chat_completion(
        self, request, request_id, request_final_usage_include_extra=False
    ):
        async with aclosing(self._events(request, True)) as events:
            async for kind, value in events:
                if kind == "usage":
                    if request.stream and not (
                        request.stream_options and request.stream_options.include_usage
                    ):
                        continue
                    if not request_final_usage_include_extra:
                        value.extra = None
                choices = (
                    []
                    if kind == "usage"
                    else [
                        ChatCompletionStreamResponseChoice(
                            delta=ChatCompletionMessage(
                                role="assistant", content=value if kind == "text" else ""
                            ),
                            finish_reason=value if kind == "finish" else None,
                        )
                    ]
                )
                yield ChatCompletionStreamResponse(
                    id=request_id,
                    model=request.model,
                    system_fingerprint="xavier-nemotron-gemma-mtp2",
                    choices=choices,
                    usage=value if kind == "usage" else None,
                )

    async def drain(self):
        if self.active is not None:
            cancelled, future = self.active
            cancelled.set()
            with anyio.CancelScope(shield=True):
                await asyncio.shield(future)
            if self.active is not None and self.active[1] is future:
                self.active = None

    def terminate(self):
        self.pool.shutdown(wait=True)

    def status(self):
        return dict(
            mode="gemma_mtp",
            draft_tokens=self.draft_tokens,
            context_length=self.context_length,
            cached_conversations=len(self.trial.entries),
            prefix_tokens=len(self.trial.pinned["tokens"]) if self.trial.pinned else 0,
        )
