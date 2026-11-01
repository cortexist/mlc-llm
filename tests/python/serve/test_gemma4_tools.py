"""CPU-only regression tests for the native Gemma4/OpenAI tool boundary.

Run directly with Python; no GPU runtime or pytest installation is needed.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[3] / "scripts/gemma4_tools.py"
SPEC = importlib.util.spec_from_file_location("gemma4_tools", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

try:
    from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest
except ImportError:
    ChatCompletionRequest = None


class Gemma4ToolsTest(unittest.TestCase):
    def test_nested_arguments_and_multiple_calls(self):
        text = (
            'Checking. <|tool_call>call:read{path:<|"|>/tmp/a\\b\n.txt<|"|>,'
            'opts:{<|"|>enabled<|"|>:true,values:[null,-1,2.5e2]}}<tool_call|>'
            '<|tool_call>call:read{path:<|"|>/tmp/b<|"|>}<tool_call|>'
        )
        content, calls = MODULE.parse_output(text, {"read"})
        self.assertEqual(content, "Checking.")
        self.assertEqual(len(calls), 2)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(
            args, {"path": "/tmp/a\\b\n.txt", "opts": {"enabled": True, "values": [None, -1, 250]}}
        )
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_incomplete_unknown_and_unsafe_calls_rejected(self):
        for text in (
            '<|tool_call>call:read{path:<|"|>x<|"|>}',
            '<|tool_call>call:bash{command:<|"|>x<|"|>}<tool_call|>',
            '<|tool_call>call:read{path:__import__("os")}<tool_call|>',
            "<|tool_call>call:read{path:null,path:false}<tool_call|>",
            "<|tool_call>call:read[]<tool_call|>",
            "<|tool_call>call:read{value:1e999}<tool_call|>",
            "<|channel>thought\nunfinished",
        ):
            with self.subTest(text=text), self.assertRaises(MODULE.ToolFormatError):
                MODULE.parse_output(text, {"read"})

    def test_normal_answer_and_thinking(self):
        self.assertEqual(MODULE.parse_output("391", {"read"}), ("391", []))
        self.assertEqual(
            MODULE.parse_output("<|channel>thought\nhidden<channel|>Done.", {"read"}), ("Done.", [])
        )

    def test_native_markers_inside_arguments_preserved(self):
        value = "<|channel>thought\nexample<channel|><|tool_call>example"
        text = (
            '<|channel>thought\nplan<channel|><|tool_call>call:write{content:<|"|>'
            + value
            + '<|"|>}<tool_call|>'
        )
        content, calls = MODULE.parse_output(text, {"write"})
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["content"], value)

    def history(self):
        return [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"/tmp/a"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_a",
                "content": [{"type": "text", "text": "actual file contents"}],
            },
        ]

    def test_openai_history_normalization(self):
        messages = MODULE.normalize_messages(self.history())
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], {"path": "/tmp/a"})
        self.assertEqual(messages[2]["content"], "actual file contents")

    def test_orphan_missing_and_duplicate_results_rejected(self):
        history = self.history()
        for messages in (
            history[:-1],
            history[2:],
            history + history[2:],
            [*history[:2], {"role": "user", "content": "Skip results"}],
        ):
            with self.subTest(messages=messages), self.assertRaises(MODULE.ToolFormatError):
                MODULE.normalize_messages(messages)


@unittest.skipIf(ChatCompletionRequest is None, "Requires the MLC Python environment")
class AdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_http_shapes_and_tool_result_history(self):
        class Usage:
            def model_dump(self, **kwargs):
                return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        class Engine:
            closed = False

            async def _handle_completion(self, request, *args, **kwargs):
                self.request = request
                try:
                    # Split inside a marker to exercise buffering across token chunks.
                    for text in ("<|tool_", 'call>call:read{path:<|"|>/tmp/a<|"|>}<tool_call|>'):
                        yield SimpleNamespace(
                            usage=None, choices=[SimpleNamespace(text=text, finish_reason="stop")]
                        )
                    yield SimpleNamespace(usage=Usage(), choices=[])
                finally:
                    self.closed = True

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["return_dict"] is False
                assert kwargs["enable_thinking"] is False
                self.messages = messages
                return [2, 105, 42]

        class RawRequest:
            async def is_disconnected(self):
                return False

        adapter = object.__new__(MODULE.Gemma4ToolAdapter)
        adapter.engine, adapter.tokenizer = Engine(), Tokenizer()
        payload = {
            "model": "nemotron",
            "temperature": 0,
            "messages": [{"role": "user", "content": "Read a file"}],
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
            ],
        }
        result = await adapter.handle(ChatCompletionRequest(**payload), RawRequest())
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        message = result["choices"][0]["message"]
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]), {"path": "/tmp/a"}
        )
        self.assertTrue(adapter.engine.closed)
        self.assertEqual(adapter.engine.request.prompt, [2, 105, 42])

        payload["messages"].extend(
            [
                message,
                {
                    "role": "tool",
                    "tool_call_id": message["tool_calls"][0]["id"],
                    "content": "actual contents",
                },
            ]
        )
        payload.update(stream=True, stream_options={"include_usage": True})
        result = await adapter.handle(ChatCompletionRequest(**payload), RawRequest())
        chunks = [chunk async for chunk in result.body_iterator]
        events = [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(events[2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(events[3]["usage"]["total_tokens"], 15)
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertEqual(adapter.tokenizer.messages[-1]["content"], "actual contents")


if __name__ == "__main__":
    unittest.main()
