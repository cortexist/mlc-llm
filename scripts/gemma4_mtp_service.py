"""Buffered OpenAI chat/SSE adapter for the bounded Unified MTP prototype."""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import fastapi
from gemma4_tools import ToolFormatError, parse_output
from jinja2 import TemplateError


class MTPChat:
    model_id = "gemma4-12b"
    context_length = 1024

    def __init__(self, root):
        self.root = root
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma4-mtp")
        # Construct and use the VM programs on the same CUDA host thread.
        try:
            self.pool.submit(self._load).result()
        except Exception:
            self.pool.shutdown(wait=True)
            raise

    def _load(self):
        from bench_xavier_mtp import Program, Trial
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.root / "dist/gemma4-12b-mtp-target", local_files_only=True
        )
        # MLC conversion copies tokenizer tables, but not chat_template.jinja.
        self.tokenizer.chat_template = (
            Path.home() / "Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-heretic/chat_template.jinja"
        ).read_text()
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 17 multiplied by 23? Give only the number."}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        target = Program(
            self.root / "dist/gemma4-12b-mtp-target",
            self.root / "dist/libs/gemma4-12b-mtp-target-sm72.so",
        )
        assistant = Program(
            Path.home() / "Models/gemma-4-12B-IT-QAT-Q4_0-unquantized-assistant-q0f16-MLC",
            self.root / "dist/libs/gemma4-12b-mtp-assistant-sm72.so",
        )
        self.trial = Trial(target, assistant)
        result = self.trial.run(prompt, 16, 2)
        text = self.tokenizer.decode(result["tokens"])
        if "391" not in text:
            raise RuntimeError("Resident MTP arithmetic warmup failed")
        print(
            json.dumps(
                {
                    "event": "mtp_loaded",
                    "model": self.model_id,
                    "context_length": self.context_length,
                    "accepted": result["accepted"],
                    "proposed": result["proposed"],
                }
            ),
            flush=True,
        )

    def prepare(self, request):
        if request.temperature not in (None, 0) or request.n != 1:
            raise ValueError("gemma4-12b supports temperature=0 and n=1")
        if (
            request.tools
            or request.tool_choice not in (None, "none", "auto")
            or any(m.role == "tool" or m.tool_calls for m in request.messages)
        ):
            raise ValueError("Tools are currently supported by nemotron only")
        if (
            request.logprobs
            or request.response_format is not None
            or request.logit_bias
            or request.frequency_penalty
            or request.presence_penalty
            or request.stop
            or request.debug_config is not None
        ):
            raise ValueError(
                "gemma4-12b does not support penalties, stops, logprobs, "
                "logit bias, response_format or debug_config"
            )
        if not request.messages:
            raise ValueError("gemma4-12b requires text system/user/assistant messages")
        messages = []
        for message in request.messages:
            content = message.content
            if isinstance(content, list):
                if any(
                    p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content
                ):
                    raise ValueError("gemma4-12b supports text content blocks only")
                content = "\n".join(p["text"] for p in content)
            if message.role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("gemma4-12b requires text system/user/assistant messages")
            messages.append(dict(role=message.role, content=content))
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        remaining = self.context_length - len(prompt) - 2
        limit = min(128, remaining) if request.max_tokens is None else request.max_tokens
        if limit < 1 or limit > remaining:
            raise ValueError(
                f"gemma4-12b context is 1024 tokens; prompt uses {len(prompt)}, "
                f"max_tokens must be 1..{max(0, remaining)} (2 verification slots)"
            )
        return prompt, limit

    def generate(self, prompt, limit):
        result = self.trial.run(prompt, limit, 2)
        ids = result["tokens"]
        stopped = ids[-1] in {1, 50, 106}
        raw = self.tokenizer.decode(ids[:-1] if stopped else ids)
        try:
            content, _ = parse_output(raw, set())
        except ToolFormatError:
            if not stopped and "<|channel>" in raw and "<channel|>" not in raw:
                content = ""
            else:
                raise
        usage = dict(
            prompt_tokens=len(prompt),
            completion_tokens=len(ids),
            total_tokens=len(prompt) + len(ids),
            extra={
                "decode_tokens_per_s": result["decode_tok_s"],
                "ttft_s": result["ttft_s"],
                "elapsed_s": result["elapsed_s"],
                "mtp_accepted": result["accepted"],
                "mtp_proposed": result["proposed"],
            },
        )
        print(json.dumps({"event": "mtp_request", "model": self.model_id, **usage}), flush=True)
        return content, "stop" if stopped else "length", usage

    async def work(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(self.pool, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A disconnected streaming client must not release the shared GPU
            # queue while its synchronous CUDA worker is still generating.
            with anyio.CancelScope(shield=True):
                await asyncio.shield(future)
            raise

    async def handle(self, request):
        try:
            prompt, limit = await self.work(self.prepare, request)
        except (ValueError, TypeError, TemplateError) as exc:
            raise fastapi.HTTPException(400, str(exc)) from exc
        request_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

        def response(choices, stream=False, usage=None):
            payload = dict(
                id=request_id,
                object="chat.completion.chunk" if stream else "chat.completion",
                created=created,
                model=self.model_id,
                choices=choices,
            )
            if usage is not None:
                payload["usage"] = usage
            return payload

        if not request.stream:
            content, finish, usage = await self.work(self.generate, prompt, limit)
            return response(
                [
                    dict(
                        index=0,
                        message=dict(role="assistant", content=content),
                        finish_reason=finish,
                    )
                ],
                usage=usage,
            )

        def event(delta=None, finish=None, usage=None):
            choices = (
                []
                if usage is not None
                else [dict(index=0, delta=delta or {}, finish_reason=finish)]
            )
            return "data: " + json.dumps(response(choices, True, usage)) + "\n\n"

        async def stream():
            task = asyncio.create_task(self.work(self.generate, prompt, limit))
            try:
                yield event(dict(role="assistant", content=""))
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=5)
                    if not done:
                        yield ": generating\n\n"
                content, finish, usage = task.result()
                yield event(dict(content=content))
                yield event(finish=finish)
                if request.stream_options and request.stream_options.include_usage:
                    yield event(usage=usage)
            except Exception as exc:
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "mtp_generation_error"}})
                    + "\n\n"
                )
            finally:
                task.cancel()
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(task, return_exceptions=True)
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(stream(), media_type="text/event-stream")
