"""OpenAI chat adapter for a resident Qwen3.5 target with a small Qwen3.5 draft.

Under dist/, as scripts/build_xavier_qwen35.sh leaves them:

  qwen35-target/  target weights and tokenizer, and the model's chat_template.jinja
  qwen35-draft/   draft weights
  libs/qwen35-target-sm72.so, libs/qwen35-draft-sm72.so

MLC_QWEN35_MODEL_ID  name of the model in the API (qwen3.8-27b)
MLC_QWEN35_DRAFTS    proposals of a round, 1 to 3; 0 serves the target alone (3)

Greedy, text only, without thinking. Plain chat streams as it is decoded. With tools the
answer is complete before anything is sent, so a partial call never reaches a client.
"""

import asyncio
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import fastapi
from gemma4_tools import ToolFormatError, normalize_messages
from jinja2 import TemplateError

CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
# A value ends at the last closing tag before the next parameter or the end of the call.
PARAMETER = re.compile(r"\s*<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>\s*(?=<parameter=|$)", re.S)
MARKERS = ("<tool_call>", "</tool_call>", "<function=", "</function>")
WARMUP = "What is 17 multiplied by 23? Give only the number."


def parse_output(text, tools):
    """Ordinary content, and the complete calls of declared tools as OpenAI tool calls."""
    schemas = {
        tool["function"]["name"]: (tool["function"].get("parameters") or {}).get("properties") or {}
        for tool in tools
    }
    calls = []

    def call(match):
        name, body, arguments, position = match[1].strip(), match[2], {}, 0
        if name not in schemas:
            raise ToolFormatError(f"Model emitted an undeclared tool name: {name[:80]!r}")
        while body[position:].strip():
            parameter = PARAMETER.match(body, position)
            if parameter is None or parameter[1] in arguments:
                raise ToolFormatError("Model emitted malformed tool parameters")
            key, value = parameter[1], parameter[2]
            # The template writes strings as they are and everything else as JSON.
            if schemas[name].get(key, {}).get("type") != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    pass
            arguments[key] = value
            position = parameter.end()
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
        return ""

    content = CALL.sub(call, text).strip()
    if any(marker in content for marker in MARKERS):
        raise ToolFormatError("Incomplete tool call")
    return content, calls


class Qwen35Chat:
    model_id = os.environ.get("MLC_QWEN35_MODEL_ID", "qwen3.8-27b")
    drafts = int(os.environ.get("MLC_QWEN35_DRAFTS", "3"))

    def __init__(self, root):
        self.root = Path(root)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen35")
        # Construct and use the VM programs on the same CUDA host thread.
        try:
            self.pool.submit(self.load).result()
        except BaseException:
            self.pool.shutdown(wait=True)
            raise

    def load(self):
        from qwen35_spec import Decoder, Program
        from transformers import AutoTokenizer

        from mlc_llm.support.auto_device import detect_device
        from mlc_llm.tokenizers import Tokenizer

        dist = self.root / "dist"
        model = dist / "qwen35-target"
        self.template = AutoTokenizer.from_pretrained(model, local_files_only=True)
        # MLC conversion copies the tokenizer tables, but not chat_template.jinja.
        self.template.chat_template = (model / "chat_template.jinja").read_text()
        self.tokenizer = Tokenizer(str(model))
        config = json.loads((model / "mlc-chat-config.json").read_text())
        self.stops = set(config["conv_template"]["stop_token_ids"])
        device = detect_device("cuda")
        target = Program(model, dist / "libs/qwen35-target-sm72.so", device, history=1)
        draft = None
        if self.drafts:
            draft = Program(
                dist / "qwen35-draft",
                dist / "libs/qwen35-draft-sm72.so",
                device,
                history=self.drafts + 1,
            )
        self.decoder = Decoder(target, draft, self.drafts or 1)
        self.context_length = self.decoder.context
        prompt = self.render([{"role": "user", "content": WARMUP}], [])
        result = self.decoder.generate(prompt, 16, self.stops)
        if "391" not in self.tokenizer.decode(result["tokens"]):
            raise RuntimeError("Resident Qwen3.5 arithmetic warmup failed")
        # Both models take the workspace of a full prefill chunk now, before the other
        # workers load: Xavier NvMap can reject a large late allocation.
        self.decoder.forget()
        self.decoder.generate((prompt * target.chunk)[: target.chunk], 2, ())
        self.decoder.forget()
        print(
            json.dumps(
                {
                    "event": "qwen35_loaded",
                    "model": self.model_id,
                    "context_length": self.context_length,
                    "drafts": self.drafts,
                    "accepted": result["accepted"],
                    "proposed": result["proposed"],
                }
            ),
            flush=True,
        )

    def render(self, messages, tools):
        return self.template.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

    def prepare(self, request):
        name = self.model_id
        if request.temperature not in (None, 0) or request.n != 1:
            raise ValueError(f"{name} supports temperature=0 and n=1")
        if request.tool_choice not in (None, "none", "auto"):
            raise ValueError(f"{name} supports tool_choice auto or none")
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
                f"{name} does not support penalties, stops, logprobs, logit bias, "
                "response_format or debug_config"
            )
        if not request.messages:
            raise ValueError(f"{name} requires text system/user/assistant messages")
        messages = normalize_messages([m.model_dump(exclude_none=True) for m in request.messages])
        if any(m["role"] not in ("system", "user", "assistant", "tool") for m in messages):
            raise ValueError("Unsupported message role")
        tools = [t.model_dump(exclude_none=True) for t in request.tools or []]
        if request.tool_choice == "none":
            tools = []
        prompt = self.render(messages, tools)
        remaining = self.context_length - len(prompt) - self.drafts
        limit = min(512, remaining) if request.max_tokens is None else request.max_tokens
        if limit < 1 or limit > remaining:
            raise ValueError(
                f"{name} context is {self.context_length} tokens; prompt uses {len(prompt)}, "
                f"max_tokens must be 1..{max(0, remaining)} ({self.drafts} verification slots)"
            )
        return prompt, limit, tools

    def streamer(self):
        from mlc_llm.tokenizers import TextStreamer

        return TextStreamer(self.tokenizer)

    def generate(self, prompt, limit, tools, emit, cancelled):
        """The assistant message, why it ended and the usage; `emit` gets text as it comes."""
        streamer, pieces = self.streamer(), []

        def on_token(token):
            pieces.append(streamer.put([token]))
            if emit is not None and pieces[-1]:
                emit(pieces[-1])
            return True

        result = self.decoder.generate(prompt, limit, self.stops, on_token, cancelled.is_set)
        pieces.append(streamer.finish())
        if emit is not None and pieces[-1]:
            emit(pieces[-1])
        content, calls = "".join(pieces), []
        if tools:
            content, calls = parse_output(content, tools)
        completion = len(result["tokens"])
        usage = dict(
            prompt_tokens=len(prompt),
            completion_tokens=completion,
            total_tokens=len(prompt) + completion,
            prompt_tokens_details={"cached_tokens": result["cached_tokens"]},
            extra={k: v for k, v in result.items() if k not in ("tokens", "stopped")},
        )
        print(json.dumps({"event": "qwen35_request", "model": self.model_id, **usage}), flush=True)
        message = dict(role="assistant", content=content or None)
        if calls:
            message["tool_calls"] = calls
        return (
            message,
            "tool_calls" if calls else ("stop" if result["stopped"] else "length"),
            usage,
        )

    async def work(self, function, *args, cancelled=None):
        future = asyncio.get_running_loop().run_in_executor(self.pool, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A disconnected client must not release the shared GPU queue while its
            # synchronous CUDA worker is still generating: it ends with the round it is in.
            if cancelled is not None:
                cancelled.set()
            with anyio.CancelScope(shield=True):
                await asyncio.wait({future})
            raise

    async def answer(self, prompt, limit, tools, emit=None):
        cancelled = threading.Event()
        return await self.work(
            self.generate, prompt, limit, tools, emit, cancelled, cancelled=cancelled
        )

    async def handle(self, request):
        try:
            prompt, limit, tools = await self.work(self.prepare, request)
        except (ValueError, TypeError, KeyError, TemplateError) as exc:
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
            try:
                message, finish, usage = await self.answer(prompt, limit, tools)
            except ToolFormatError as exc:
                raise fastapi.HTTPException(502, str(exc)) from exc
            return response([dict(index=0, message=message, finish_reason=finish)], usage=usage)

        def event(delta=None, finish=None, usage=None):
            choices = (
                []
                if usage is not None
                else [dict(index=0, delta=delta or {}, finish_reason=finish)]
            )
            return "data: " + json.dumps(response(choices, True, usage)) + "\n\n"

        async def stream():
            loop, texts = asyncio.get_running_loop(), asyncio.Queue()

            def emit(text):
                loop.call_soon_threadsafe(texts.put_nowait, text)

            task = asyncio.create_task(self.answer(prompt, limit, tools, None if tools else emit))
            task.add_done_callback(lambda _: texts.put_nowait(None))
            try:
                yield event(dict(role="assistant", content=""))
                while True:
                    try:
                        text = await asyncio.wait_for(texts.get(), timeout=5)
                    except asyncio.TimeoutError:
                        yield ": generating\n\n"
                        continue
                    if text is None:
                        break
                    yield event(dict(content=text))
                message, finish, usage = task.result()
                if tools:
                    delta = {}
                    if message["content"]:
                        delta["content"] = message["content"]
                    if "tool_calls" in message:
                        delta["tool_calls"] = [
                            {"index": i, **call} for i, call in enumerate(message["tool_calls"])
                        ]
                    yield event(delta)
                yield event(finish=finish)
                if request.stream_options and request.stream_options.include_usage:
                    yield event(usage=usage)
            except Exception as exc:
                kind = (
                    "tool_format_error" if isinstance(exc, ToolFormatError) else "generation_error"
                )
                yield "data: " + json.dumps({"error": {"message": str(exc), "type": kind}}) + "\n\n"
            finally:
                task.cancel()
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(task, return_exceptions=True)
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(stream(), media_type="text/event-stream")
