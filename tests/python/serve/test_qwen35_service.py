"""CPU checks for the Qwen3.5 decoder's bookkeeping, its tool calls and its chat adapter.

Run directly with Python in the MLC environment; no GPU or pytest is needed.
"""

import asyncio
import json
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from gemma4_tools import ToolFormatError
from qwen35_service import PINNED, Qwen35Chat, before_call, parse_output
from qwen35_spec import BLOCK, OPEN, PAGE, Decoder

from mlc_llm.protocol.openai_api_protocol import ChatCompletionRequest

STOP = 7


def target_rule(tokens):
    """The token after `tokens`: any function of all of them will do."""
    if len(tokens) in (23, 41):
        return STOP
    return 10 + (sum(tokens) * 31 + len(tokens) * 7) % 40


def draft_rule(tokens):
    """The target's token, except after every fourth token."""
    return target_rule(tokens) if len(tokens) % 4 else 9


class Model:
    """Stands in for a Program: remembers what its cache and its state have read."""

    context = 64

    def __init__(self, rule, history):
        self.rule, self.history, self.device = rule, history, self
        self.cache = self.state = self.kept = self.block = None
        self.sequence = OPEN
        self.steps = 0  # single-token steps the state can go back

    def sync(self):
        pass

    def pages(self):
        """Pages of the conversation and, when it is no part of it, of the pinned block."""
        beside = self.block if self.sequence == OPEN else None
        return sum(-(-len(tokens) // PAGE) for tokens in (self.cache, beside) if tokens)

    def clear(self):
        self.cache = self.state = self.kept = None

    def start(self, block=False):
        assert self.block is None or not block
        self.cache, self.state, self.kept, self.steps = [], [], None, 0
        self.sequence = BLOCK if block else OPEN

    def pin(self):
        assert self.sequence == BLOCK and self.cache == self.state
        self.block = list(self.state)

    def unpin(self):
        if self.block is not None and self.sequence == BLOCK:
            self.clear()
        self.block = None

    def restore(self):
        self.cache, self.state, self.kept, self.steps = list(self.block), list(self.block), None, 0
        self.sequence = BLOCK

    def mark(self):
        assert self.cache == self.state and self.pages() <= self.context // PAGE
        self.kept = list(self.state)

    def recall(self):
        assert self.cache[: len(self.kept)] == self.kept
        self.cache, self.state, self.steps = list(self.kept), list(self.kept), 0

    def prefill(self, tokens):
        assert tokens and self.cache == self.state
        self.cache, self.state, self.steps = self.cache + tokens, self.state + tokens, 0
        return self.rule(self.state)

    def decode(self, token):
        assert self.cache == self.state
        self.cache, self.state = [*self.cache, token], [*self.state, token]
        self.steps = min(self.steps + 1, self.history - 1)
        return self.rule(self.state)

    def verify(self, tokens):
        assert 1 <= len(tokens) <= 4 and self.cache == self.state
        self.cache = self.cache + tokens
        assert self.pages() <= self.context // PAGE
        return [self.rule(self.state + tokens[: i + 1]) for i in range(len(tokens))], tokens

    def commit(self, held, rows, drop):
        assert rows >= 1 and rows + drop == len(held)
        self.cache = self.cache[: len(self.cache) - drop]
        self.state = self.state + held[:rows]
        assert self.cache == self.state

    def pop(self, count):
        assert 0 <= count <= self.steps
        self.steps -= count
        if count:
            self.cache, self.state = self.cache[:-count], self.state[:-count]


def reference(prompt, limit):
    output = []
    while len(output) < limit and (not output or output[-1] != STOP):
        output.append(target_rule(prompt + output))
    return output


def pair(proposals=3):
    target, draft = Model(target_rule, 1), Model(draft_rule, proposals + 1)
    return Decoder(target, draft, proposals), target, draft


class Decoding(unittest.TestCase):
    def test_tokens_and_states_for_every_width_and_limit(self):
        for proposals in (0, 1, 2, 3):
            for start in (1, 5, 12):
                for limit in range(1, 30):
                    prompt = list(range(30, 30 + start))
                    if proposals:
                        decoder, target, draft = pair(proposals)
                    else:
                        target, draft = Model(target_rule, 1), None
                        decoder = Decoder(target)
                    result = decoder.generate(prompt, limit, {STOP})
                    with self.subTest(proposals=proposals, start=start, limit=limit):
                        self.assertEqual(result["tokens"], reference(prompt, limit))
                        read = prompt + result["tokens"][:-1]
                        for model in filter(None, (target, draft)):
                            self.assertEqual((model.cache, model.state), (read, read))
                        self.assertEqual(result["stopped"], result["tokens"][-1] == STOP)

    def test_the_callback_sees_no_stop_token_and_can_end_the_answer(self):
        decoder, target, draft = pair()
        seen = []
        prompt = list(range(30, 42))
        result = decoder.generate(prompt, 40, {STOP}, seen.append)
        self.assertEqual(seen, result["tokens"][:1])
        seen.clear()
        decoder.forget()
        result = decoder.generate(prompt, 40, {STOP}, lambda t: seen.append(t) or len(seen) < 6)
        self.assertEqual(result["tokens"], reference(prompt, 40)[:6])
        self.assertEqual(target.state, prompt + result["tokens"][:-1])
        self.assertEqual(draft.state, target.state)
        decoder.forget()
        result = decoder.generate(prompt, 40, {STOP})
        self.assertEqual(result["tokens"][-1], STOP)
        self.assertEqual(len(seen), 6)

    def test_a_conversation_continues_from_its_end_or_from_its_prompt(self):
        decoder, target, draft = pair()
        prompt = list(range(30, 35))
        first = decoder.generate(prompt, 8, {STOP})
        self.assertEqual((first["resumed"], first["cached_tokens"]), (None, 0))
        longer = prompt + first["tokens"] + [50, 51]
        result = decoder.generate(longer, 8, {STOP})
        self.assertEqual((result["resumed"], result["cached_tokens"]), ("live", 12))
        self.assertEqual(result["prefill_tokens"], 3)
        self.assertEqual(result["tokens"], reference(longer, 8))
        # Another answer to the last prompt: only the prompt is still there.
        edited = [*longer, 60]
        result = decoder.generate(edited, 8, {STOP})
        self.assertEqual((result["resumed"], result["cached_tokens"]), ("anchor", len(longer)))
        self.assertEqual(result["tokens"], reference(edited, 8))
        # The same prompt again: nothing to read.
        result = decoder.generate(edited, 8, {STOP})
        self.assertEqual((result["resumed"], result["prefill_tokens"]), ("anchor", 0))
        self.assertEqual(result["tokens"], reference(edited, 8))
        self.assertEqual(draft.state, target.state)
        other = [1, 2, 3]
        result = decoder.generate(other, 8, {STOP})
        self.assertEqual((result["resumed"], result["tokens"]), (None, reference(other, 8)))

    def test_a_pinned_block_begins_other_conversations(self):
        decoder, target, draft = pair()
        block = list(range(100, 120))

        def answer(prompt, pinned=len(block)):
            result = decoder.generate(prompt, 8, {STOP}, block=pinned)
            self.assertEqual(result["tokens"], reference(prompt, 8))
            self.assertEqual(target.state, prompt + result["tokens"][:-1])
            self.assertEqual((draft.state, draft.block), (target.state, target.block))
            return result["resumed"], result["cached_tokens"], result["tokens"]

        first = [*block, 30, 31, 32, 33]
        resumed, cached, tokens = answer(first)
        self.assertEqual((resumed, cached, target.block), (None, 0, block))
        # Its own conversation goes on as any other does.
        self.assertEqual(answer([*first, *tokens, 50])[:2], ("live", len(first) + 7))
        # Another one with the same block reads what follows the block.
        second = [*block, 40, 41]
        self.assertEqual(answer(second)[:2], ("pinned", 20))
        self.assertEqual(answer([*second, 42])[0], "anchor")
        self.assertEqual(answer([*block, 60])[:2], ("pinned", 20))
        # A conversation without the block is read beside it and leaves it there.
        other = [1, 2, 3]
        resumed, _, tokens = answer(other, 0)
        self.assertEqual((resumed, target.sequence, target.block), (None, OPEN, block))
        self.assertEqual(answer([*other, *tokens, 4], 0)[0], "live")
        self.assertEqual(answer(second)[:2], ("pinned", 20))
        self.assertEqual(target.sequence, BLOCK)
        # The block alone has no token after it to answer with.
        self.assertEqual(answer(block, 0)[0], None)
        self.assertEqual(target.block, block)
        # Another block replaces it.
        new = [*range(200, 216), 5]
        self.assertEqual((answer(new, 16)[0], target.block), (None, new[:16]))
        self.assertEqual((answer(second)[0], target.block), (None, block))
        with self.assertRaises(ValueError):
            decoder.generate(second, 8, {STOP}, block=len(second))

    def test_the_pinned_block_gives_way_when_the_pages_run_out(self):
        decoder, target, draft = pair()
        block = list(range(100, 120))  # two of the four pages
        decoder.generate([*block, 30], 8, {STOP}, block=20)
        # 3 + 26 + 3 tokens fit the other two pages; one more does not.
        other = [1, 2, 3]
        result = decoder.generate(other, 26, {STOP})
        self.assertEqual((result["tokens"], target.block), (reference(other, 26), block))
        grown = [*other, *result["tokens"], 4]
        result = decoder.generate(grown, 8, {STOP})
        self.assertEqual((result["resumed"], result["tokens"]), ("live", reference(grown, 8)))
        self.assertEqual((target.block, draft.block, decoder.pinned), (None, None, None))
        decoder.generate([*block, 30], 8, {STOP}, block=20)
        result = decoder.generate(other, 27, {STOP})
        self.assertEqual((result["tokens"], target.block), (reference(other, 27), None))
        # A conversation that begins with the block has the whole context.
        decoder.generate([*block, 30], 8, {STOP}, block=20)
        long = [*block, *range(40, 70)]
        result = decoder.generate(long, 11, {STOP})
        self.assertEqual((result["resumed"], result["tokens"]), ("pinned", reference(long, 11)))

    def test_cancelling_and_failing(self):
        decoder, target, _ = pair()
        prompt = list(range(30, 35))
        result = decoder.generate(prompt, 20, {STOP}, cancelled=lambda: True)
        self.assertEqual(result["tokens"], reference(prompt, 1))
        self.assertEqual(target.state, prompt)

        def fail(_):
            raise RuntimeError("client code")

        with self.assertRaises(RuntimeError):
            decoder.generate([*prompt, 1], 20, {STOP}, fail)
        self.assertIsNone(target.state)
        self.assertEqual(decoder.generate(prompt, 20, {STOP})["resumed"], None)
        # Nothing outlives a failure, the pinned block included.
        decoder.generate([*range(100, 120), 1], 8, {STOP}, block=20)
        with self.assertRaises(RuntimeError):
            decoder.generate([*range(100, 120), 2], 8, {STOP}, fail)
        self.assertEqual((target.state, target.block, decoder.pinned), (None, None, None))
        for prompt, limit in (([], 1), ([1], 0), ([1] * 60, 2)):
            with self.assertRaises(ValueError):
                decoder.generate(prompt, limit, {STOP})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seek",
            "parameters": {"type": "object", "properties": {"line": {"type": "integer"}}},
        },
    },
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object"}}},
]


def call(name, **parameters):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in parameters.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class ToolCalls(unittest.TestCase):
    def arguments(self, text):
        content, calls = parse_output(text, TOOLS)
        return content, [
            (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls
        ]

    def test_values_keep_their_declared_types(self):
        text = "I will write it.\n\n" + call("write", path="12", content="a\n\n</parameter> b\n")
        self.assertEqual(
            self.arguments(text),
            ("I will write it.", [("write", {"path": "12", "content": "a\n\n</parameter> b\n"})]),
        )
        text = (
            call("seek", line="12", flags='{"a": [1, true]}', note="not json") + "\n" + call("now")
        )
        self.assertEqual(
            self.arguments(text),
            (
                "",
                [
                    ("seek", {"line": 12, "flags": {"a": [1, True]}, "note": "not json"}),
                    ("now", {}),
                ],
            ),
        )
        self.assertEqual(self.arguments("Just text."), ("Just text.", []))

    def test_text_is_sent_up_to_what_may_begin_a_call(self):
        for text, sent in [
            ("Compare a < b", "Compare a < b"),
            ("Compare a <", "Compare a"),
            ("Reading.\n\n<tool_c", "Reading."),
            ("Reading.\n\n<tool_call>\n<function=wr", "Reading."),
            ("<tool_call>", ""),
            (" \n", ""),
        ]:
            with self.subTest(text=text):
                self.assertEqual(before_call(text), sent)

    def test_undeclared_incomplete_and_malformed_calls_are_errors(self):
        for text in [
            call("remove", path="/"),
            "<tool_call>\n<function=write>\n<parameter=path>\na",
            call("write", path="a") + "\n</tool_call>",
            "<tool_call>\n<function=write>\nstray\n</function>\n</tool_call>",
            call("write", path="a").replace(
                "</parameter>\n", "</parameter>\n<parameter=path>\nb\n</parameter>\n"
            ),
        ]:
            with self.subTest(text=text), self.assertRaises(ToolFormatError):
                parse_output(text, TOOLS)


class Template:
    prompt = (1, 2, 3)

    def apply_chat_template(self, messages, tools=None, **options):
        self.messages, self.tools, self.options = messages, tools, options
        return list(self.prompt)


class Streamer:
    def put(self, tokens):
        return "".join(self.text[t] for t in tokens)

    def finish(self):
        return ""


class Engine:
    """Decodes a fixed answer; every piece of text is one token."""

    context = 4096

    def generate(self, prompt, limit, stops, on_token, cancelled, block):
        self.prompt, self.block, tokens = prompt, block, []
        for token in range(len(Streamer.text)):
            tokens.append(token)
            if not on_token(token) or cancelled() or len(tokens) == limit:
                break
        else:
            tokens.append(STOP)
        return dict(tokens=tokens, stopped=tokens[-1] == STOP, cached_tokens=2, accepted=1)


class Adapter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = object.__new__(Qwen35Chat)
        self.chat.pool = ThreadPoolExecutor(max_workers=1)
        self.chat.template, self.chat.decoder, self.chat.stops = Template(), Engine(), {STOP}
        self.chat.context_length, self.chat.streamer = 4096, Streamer
        self.chat.system, self.chat.end = [1, 2], 9
        self.addCleanup(self.chat.pool.shutdown)

    async def events(self, payload):
        result = await self.chat.handle(ChatCompletionRequest(**payload))
        chunks = [chunk async for chunk in result.body_iterator]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        return [json.loads(chunk[6:]) for chunk in chunks if chunk.startswith("data: {")]

    async def test_text_streams_and_a_call_is_sent_when_it_is_complete(self):
        Streamer.text = ["Hel", "lo", "."]
        payload = {"model": "qwen", "messages": [{"role": "user", "content": "Hi"}]}
        result = await self.chat.handle(ChatCompletionRequest(**payload))
        choice = result["choices"][0]
        self.assertEqual(
            (choice["message"]["content"], choice["finish_reason"]), ("Hello.", "stop")
        )
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 2)
        self.assertEqual(self.chat.template.options["enable_thinking"], False)
        result = await self.chat.handle(ChatCompletionRequest(**payload, max_tokens=2))
        choice = result["choices"][0]
        self.assertEqual(
            (choice["message"]["content"], choice["finish_reason"]), ("Hello", "length")
        )

        payload.update(stream=True, stream_options={"include_usage": True})
        events = await self.events(payload)
        deltas = [e["choices"][0]["delta"].get("content") for e in events[:-2]]
        self.assertEqual(deltas, ["", "Hel", "lo", "."])
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[-1]["usage"]["total_tokens"], 7)

        # With tools the text streams too; what may begin a call waits.
        payload["tools"] = TOOLS
        Streamer.text = ["Hel", "lo", ".\n"]
        events = await self.events(payload)
        deltas = [e["choices"][0]["delta"] for e in events[:-2]]
        self.assertEqual(
            deltas,
            [{"role": "assistant", "content": ""}]
            + [{"content": text} for text in ("Hel", "lo", ".")],
        )
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "stop")

        written = call("write", path="/tmp/a", content="x")
        Streamer.text = [
            "Read",
            "ing a <",
            " b.\n",
            "\n" + written[:5],
            written[5:40],
            written[40:],
        ]
        events = await self.events(payload)
        deltas = [e["choices"][0]["delta"] for e in events[1:-2]]
        self.assertEqual(deltas[:-1], [{"content": text} for text in ("Read", "ing a", " < b.")])
        self.assertEqual(list(deltas[-1]), ["tool_calls"])
        self.assertEqual(deltas[-1]["tool_calls"][0]["function"]["name"], "write")
        self.assertEqual(deltas[-1]["tool_calls"][0]["index"], 0)
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "tool_calls")

        # Text after a call is sent with the answer complete; a call that does not end
        # is an error after the text before it.
        Streamer.text = ["First.\n", written, "\nThen."]
        events = await self.events(payload)
        deltas = [e["choices"][0]["delta"] for e in events[1:-2]]
        self.assertEqual(deltas[:-1], [{"content": "First."}, {"content": "\n\nThen."}])
        Streamer.text = ["Reading.\n", written[:40]]
        events = await self.events(payload)
        self.assertEqual(events[1]["choices"][0]["delta"], {"content": "Reading."})
        self.assertEqual(events[2]["error"]["type"], "tool_format_error")
        self.assertEqual(len(events), 3)

        del payload["stream_options"]
        payload.update(stream=False, tool_choice="none")
        result = await self.chat.handle(ChatCompletionRequest(**payload))
        self.assertIn("<tool_call>", result["choices"][0]["message"]["content"])
        self.assertIsNone(self.chat.template.tools)

    async def test_a_long_system_block_is_offered_for_pinning(self):
        Streamer.text = ["Yes."]
        payload = {"model": "qwen", "messages": [{"role": "user", "content": "Hi"}]}
        body = [5] * (PINNED - 3)
        for prompt, block in [
            ([1, 2, *body, 9, 4, 4], PINNED),
            ([1, 2, *body, 9, 4, 9, 4], PINNED),
            ([1, 2, *body[1:], 9, 4, 4], 0),  # a short block
            ([1, 3, *body, 9, 4, 4], 0),  # no system block
            ([1, 2, *body, 9], 0),  # nothing after it
            ([1, 2, *body, 4], 0),
        ]:
            with self.subTest(prompt=prompt[-4:], size=len(prompt)):
                self.chat.template.prompt = prompt
                await self.chat.handle(ChatCompletionRequest(**payload))
                self.assertEqual(self.chat.decoder.block, block)

    async def test_requests_outside_the_service_fail_explicitly(self):
        import fastapi

        base = {"model": "qwen", "messages": [{"role": "user", "content": "Hi"}]}
        for extra in [
            {"temperature": 1},
            {"n": 2},
            {"stop": ["x"]},
            {"max_tokens": 4091},
            {"max_tokens": 0},
            {"tool_choice": {"type": "function", "function": {"name": "now"}}, "tools": TOOLS},
        ]:
            with self.subTest(extra=extra):
                with self.assertRaises(fastapi.HTTPException) as caught:
                    await self.chat.handle(ChatCompletionRequest(**base, **extra))
                self.assertEqual(caught.exception.status_code, 400)
        # The refusal of a request that is too long, in the words clients look for.
        with self.assertRaises(fastapi.HTTPException) as caught:
            await self.chat.handle(ChatCompletionRequest(**base, max_tokens=4091))
        self.assertRegex(caught.exception.detail, r"maximum context length is \d+ tokens")
        self.assertIn("reduce the length of the messages", caught.exception.detail)
        Streamer.text = [call("remove", path="/")]
        with self.assertRaises(fastapi.HTTPException) as caught:
            await self.chat.handle(ChatCompletionRequest(**base, tools=TOOLS))
        self.assertEqual(caught.exception.status_code, 502)
        events = await self.events({**base, "tools": TOOLS, "stream": True})
        self.assertEqual(events[-1]["error"]["type"], "tool_format_error")

    async def test_a_cancelled_request_waits_for_its_worker(self):
        Streamer.text = ["a"] * 50
        seen = []
        original = self.chat.decoder.generate

        def slow(prompt, limit, stops, on_token, cancelled, block):
            def step(token):
                seen.append(cancelled())
                import time

                time.sleep(0.01)
                return on_token(token)

            return original(prompt, limit, stops, step, cancelled, block)

        self.chat.decoder.generate = slow
        task = asyncio.create_task(self.chat.answer([1], 50, []))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        count = len(seen)
        self.assertLess(count, 50)
        await asyncio.sleep(0.05)
        self.assertEqual(len(seen), count)


if __name__ == "__main__":
    unittest.main()
