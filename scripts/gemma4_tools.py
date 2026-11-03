"""Native Gemma4 tool calls for the Xavier service (no tool execution here).

Formatting uses the served NVIDIA checkpoint's official chat_template.jinja.
The small parser accepts Gemma4's marker-quoted values without evaluating code.
Tool generations are buffered before emitting OpenAI deltas, so partial or
malformed calls never become executable client tool requests.
"""

import asyncio
import json
import math
import re
import time
import uuid

QUOTE = '<|"|>'
CALL_START = "<|tool_call>"
CALL_END = "<tool_call|>"


class ToolFormatError(ValueError):
    """Invalid native tool syntax or history."""


class ValueParser:
    """Parse JSON-like values with bare keys and Gemma4 quoted strings."""

    def __init__(self, source):
        self.source = source
        self.pos = 0

    def whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self.pos += 1

    def take(self, token):
        self.whitespace()
        if self.source.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def value(self, depth=0):
        if depth > 64:
            raise ToolFormatError("Tool arguments exceed maximum nesting")
        self.whitespace()
        if self.take(QUOTE):
            end = self.source.find(QUOTE, self.pos)
            if end < 0:
                raise ToolFormatError("Unterminated tool string")
            result = self.source[self.pos : end]
            self.pos = end + len(QUOTE)
            return result
        if self.take("{"):
            result = {}
            if self.take("}"):
                return result
            while True:
                self.whitespace()
                if self.source.startswith(QUOTE, self.pos):
                    key = self.value(depth + 1)
                else:
                    match = re.match(r"[^\s:{}\[\],]+", self.source[self.pos :])
                    if match is None:
                        raise ToolFormatError("Invalid tool argument key")
                    key = match[0]
                    self.pos += len(key)
                if key in result or not self.take(":"):
                    raise ToolFormatError("Duplicate key or missing colon")
                result[key] = self.value(depth + 1)
                if self.take("}"):
                    return result
                if not self.take(","):
                    raise ToolFormatError("Expected comma in tool object")
        if self.take("["):
            result = []
            if self.take("]"):
                return result
            while True:
                result.append(self.value(depth + 1))
                if self.take("]"):
                    return result
                if not self.take(","):
                    raise ToolFormatError("Expected comma in tool array")
        match = re.match(
            r"(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            self.source[self.pos :],
        )
        if match is None:
            raise ToolFormatError("Invalid tool argument value")
        self.pos += len(match[0])
        value = json.loads(match[0])
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolFormatError("Non-finite tool argument number")
        return value


def parse_output(text, allowed_names):
    """Return ordinary content and complete, declared OpenAI tool calls."""
    calls, content = [], []
    while text:
        marker = re.search(r"<\|(?:tool_call|channel)>", text)
        if marker is None:
            content.append(text)
            break
        content.append(text[: marker.start()])
        text = text[marker.end() :]
        if marker[0] == "<|channel>":
            end = text.find("<channel|>")
            if end < 0:
                raise ToolFormatError("Incomplete native thinking channel")
            text = text[end + len("<channel|>") :]
            continue
        # Some valid generations put whitespace after the marker or colon.
        # Normalize syntax only; never guess or remap undeclared tool names.
        match = re.match(r"\s*call:\s*([A-Za-z_][\w.-]*)", text)
        if match is None:
            raise ToolFormatError("Model emitted an invalid native tool-call header")
        if match[1] not in allowed_names:
            raise ToolFormatError(f"Model emitted an undeclared tool name: {match[1][:80]!r}")
        parser = ValueParser(text[match.end() :])
        arguments = parser.value()
        if not isinstance(arguments, dict) or not parser.take(CALL_END):
            raise ToolFormatError("Incomplete tool call or non-object arguments")
        text = parser.source[parser.pos :]
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": match[1],
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    content = "".join(content).strip()
    if any(
        marker in content
        for marker in ("<|tool", "<tool|>", "<tool_call|>", "<|channel>", "<channel|>")
    ):
        raise ToolFormatError("Incomplete native tool/channel output")
    return content, calls


def normalize_messages(messages):
    """Decode OpenAI argument strings and require complete, ordered tool results."""
    messages = [dict(message) for message in messages]
    pending, seen = set(), set()
    for message in messages:
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ToolFormatError("Tool result does not match a pending call")
            pending.remove(call_id)
        else:
            if pending:
                raise ToolFormatError("Tool results must immediately follow assistant calls")
            calls = message.get("tool_calls") or []
            if calls and message["role"] != "assistant":
                raise ToolFormatError("Only assistant messages can contain tool calls")
            for call in calls:
                call_id = call["id"]
                if call_id in seen:
                    raise ToolFormatError("Duplicate tool call ID")
                seen.add(call_id)
                pending.add(call_id)
                arguments = call["function"].get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ToolFormatError("Tool arguments must be an object")
                call["function"]["arguments"] = arguments
        if isinstance(message.get("content"), list):
            if any(part.get("type") != "text" for part in message["content"]):
                raise ToolFormatError("This service supports text tool conversations only")
            message["content"] = "\n".join(part["text"] for part in message["content"])
    if pending:
        raise ToolFormatError("Missing tool results")
    return messages


class Gemma4ToolAdapter:
    """Translate chat requests onto the existing raw completion engine."""

    def __init__(self, engine, tokenizer_path):
        from transformers import AutoTokenizer

        self.engine = engine
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    async def handle(self, request, raw_request):
        import fastapi
        from jinja2 import TemplateError

        from mlc_llm.protocol.openai_api_protocol import CompletionRequest

        try:
            if request.logprobs or request.response_format is not None:
                raise ToolFormatError("Tool adapter does not support logprobs/response_format")
            if request.tool_choice not in (None, "auto", "none"):
                raise ToolFormatError("Tool adapter supports tool_choice auto or none")
            messages = normalize_messages(
                [m.model_dump(exclude_none=True) for m in request.messages]
            )
            tools = [t.model_dump(exclude_none=True) for t in request.tools or []]
            if request.tool_choice == "none":
                tools = []
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            fields = (
                "model",
                "frequency_penalty",
                "presence_penalty",
                "logit_bias",
                "max_tokens",
                "n",
                "seed",
                "stop",
                "temperature",
                "top_p",
            )
            completion = CompletionRequest(
                prompt=prompt, **{f: getattr(request, f) for f in fields}
            )
        except (ValueError, TypeError, KeyError, TemplateError) as exc:
            raise fastapi.HTTPException(400, str(exc)) from exc

        request_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        async def collect():
            text, finish, usage = "", "stop", None
            generator = self.engine._handle_completion(
                completion, request_id, request_final_usage_include_extra=True
            )
            try:
                async for response in generator:
                    if not request.stream and await raw_request.is_disconnected():
                        raise asyncio.CancelledError()
                    if response.usage is not None:
                        usage = response.usage.model_dump(exclude={"extra"})
                        metrics = getattr(response.usage, "extra", None) or {}
                        if "prefill_tokens" in metrics:
                            usage["prompt_tokens_details"] = {
                                "cached_tokens": max(
                                    0, usage["prompt_tokens"] - metrics["prefill_tokens"]
                                )
                            }
                    for choice in response.choices:
                        text += choice.text
                        finish = choice.finish_reason or finish
            finally:
                await generator.aclose()
            content, calls = parse_output(text, {tool["function"]["name"] for tool in tools})
            message = {"role": "assistant", "content": content or None}
            if calls:
                message["tool_calls"] = calls
                finish = "tool_calls"
            return message, finish, usage

        if not request.stream:
            try:
                message, finish, usage = await collect()
            except ToolFormatError as exc:
                raise fastapi.HTTPException(502, str(exc)) from exc
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage,
            }

        def event(delta=None, finish=None, usage=None):
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": []
                if usage is not None
                else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
            }
            if usage is not None:
                payload["usage"] = usage
            return "data: " + json.dumps(payload) + "\n\n"

        async def stream():
            task = asyncio.create_task(collect())
            try:
                yield event({"role": "assistant", "content": ""})
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=5)
                    if not done:
                        yield ": generating\n\n"
                message, finish, usage = task.result()
                delta = {}
                if message["content"]:
                    delta["content"] = message["content"]
                if "tool_calls" in message:
                    delta["tool_calls"] = [
                        {"index": i, **call} for i, call in enumerate(message["tool_calls"])
                    ]
                yield event(delta)
                yield event(finish=finish)
                if request.stream_options and request.stream_options.include_usage and usage:
                    yield event(usage=usage)
            except ToolFormatError as exc:
                yield (
                    "data: "
                    + json.dumps({"error": {"message": str(exc), "type": "tool_format_error"}})
                    + "\n\n"
                )
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            yield "data: [DONE]\n\n"

        return fastapi.responses.StreamingResponse(stream(), media_type="text/event-stream")
