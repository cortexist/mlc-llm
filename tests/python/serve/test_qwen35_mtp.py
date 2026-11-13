"""CPU checks of the Qwen3.5 decoder with the target's own MTP head.

The Program of scripts/qwen35_spec.py runs here on a machine that only remembers: what the
target's cache layers and the head's layer hold for every token, and what the recurrent
state has read. A final hidden state is the number of the tokens read up to it, so the test
sees whether the head is always given the hidden state that belongs to its token.

Run directly with Python in the MLC environment; no GPU or pytest is needed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import qwen35_spec
from qwen35_spec import BLOCK, LIVE, OPEN, Decoder, Program

STOP = 7
VOCABULARY = 64


def target_rule(tokens):
    """The token after `tokens`: any function of all of them will do."""
    if len(tokens) in (23, 41):
        return STOP
    return 10 + (sum(tokens) * 31 + len(tokens) * 7) % 40


def head_rule(tokens):
    """The target's token, except after every fourth token."""
    return target_rule(tokens) if len(tokens) % 4 else 9


class Array:
    """A tensor of the machine."""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.dtype = self.data.shape, self.data.dtype

    def numpy(self):
        return self.data


class Machine:
    """Stands in for the compiled library and the two states of a Program."""

    def __init__(self):
        self.rows = {}  # cache sequence -> [token, read by the target, the head's pair]
        self.read = {}  # state sequence -> tokens the recurrent state has read
        self.numbers = {(): 0}  # tokens read -> the number that is their final hidden state
        self.pending = None  # rows of the forward that has begun
        self.steps = 0

    def hidden(self, tokens):
        return self.numbers.setdefault(tuple(tokens), len(self.numbers))

    def tokens(self, sequence):
        return [row[0] for row in self.rows[sequence]]

    def logits(self, choices):
        table = np.zeros((1, len(choices), VOCABULARY), "float32")
        table[0, range(len(choices)), choices] = 1
        return Array(table)

    def states(self, numbers):
        return Array(np.array(numbers, "float32").reshape(1, -1, 1))

    # What the library exports.
    def embed(self, ids, _params):
        return Array(ids.numpy().astype("float32").reshape(-1, 1))

    def take(self, embeds, prev_hidden):
        """The begun rows get the tokens; the head's layer pairs each with the state before."""
        sequence, count = self.pending
        self.pending = None
        tokens = embeds.numpy().reshape(-1).astype(int).tolist()
        assert len(tokens) == count
        before = self.tokens(sequence)
        # The target reads on from what its recurrent state has read, and nothing else.
        assert before == self.read[LIVE], (before, self.read[LIVE])
        assert all(row[1] for row in self.rows[sequence])
        assert int(prev_hidden.numpy().reshape(-1)[0]) == self.hidden(before)
        numbers = []
        for index, token in enumerate(tokens):
            pair = (token, self.hidden(before + tokens[:index]))
            self.rows[sequence].append([token, True, pair])
            numbers.append(self.hidden(before + tokens[: index + 1]))
        choices = [target_rule(before + tokens[: index + 1]) for index in range(count)]
        return tokens, choices, numbers

    def mtp_prefill(self, embeds, prev_hidden, last, cache, state, _params):
        tokens, choices, numbers = self.take(embeds, prev_hidden)
        self.read[LIVE] = self.read[LIVE] + tokens
        last = int(last.numpy()[0])
        return self.logits(choices[last:][:1]), self.states(numbers[last:][:1]), cache, state, None

    def mtp_verify(self, embeds, prev_hidden, cache, state, _params):
        tokens, choices, numbers = self.take(embeds, prev_hidden)
        return self.logits(choices), self.states(numbers), cache, state, tokens, None

    def commit_rows(self, held, rows, _state):
        self.read[LIVE] = self.read[LIVE] + held[: int(rows[0])]

    def mtp_step(self, embeds, hidden, cache, _params):
        sequence, count = self.pending
        self.pending = None
        assert count == 1
        rows = self.rows[sequence]
        # The head attends to its pairs of all tokens before: each has to be the right one.
        before = []
        for token, _, pair in rows:
            assert pair == (token, self.hidden(before)), (before, token, pair)
            before.append(token)
        assert int(hidden.numpy().reshape(-1)[0]) == self.hidden(before)
        token = int(embeds.numpy().reshape(-1)[0])
        rows.append([token, False, (token, self.hidden(before))])
        self.steps += 1
        after = [*before, token]
        return self.logits([head_rule(after)]), self.states([self.hidden(after)]), cache

    def __getitem__(self, name):
        return getattr(self, name)

    # What the runtime does to a cache ("cache") or a recurrent state ("state").
    def builtin(self, name):
        def call(kind, *args):
            held = self.rows if kind == "cache" else self.read
            if name == "kv_state_add_sequence":
                assert args[0] not in held
                held[args[0]] = []
            elif name == "kv_state_remove_sequence":
                del held[args[0]]
            elif name == "kv_state_fork_sequence":
                assert kind == "state" and args[1] not in held
                held[args[1]] = list(held[args[0]])
            elif name == "kv_state_popn":
                assert kind == "cache" and 0 < args[1] <= len(held[args[0]])
                del held[args[0]][-args[1] :]
            elif name == "kv_state_begin_forward" and kind == "cache":
                assert self.pending is None
                self.pending = (int(args[0][0]), int(args[1][0]))
            elif name == "reshape":
                return Array(kind.numpy().reshape(tuple(args[0])))

        return call


class Device:
    def sync(self):
        pass


def program(machine, context=64, chunk=8):
    """A Program on the machine, without a library to load."""
    result = object.__new__(Program)
    result.mod, result.params, result.device = machine, None, Device()
    result.cache, result.state = "cache", "state"
    result.context, result.chunk = context, chunk
    result.sequence, result.length, result.anchor, result.pinned = OPEN, None, None, None
    result.mtp = True
    result.hidden = result.anchor_hidden = result.pinned_hidden = None
    result.rows = result.zeros = None
    result.tensor = lambda values, dtype="int32": Array(np.array(values, dtype))
    return result


def reference(prompt, limit):
    output = []
    while len(output) < limit and (not output or output[-1] != STOP):
        output.append(target_rule(prompt + output))
    return output


class Head(unittest.TestCase):
    def setUp(self):
        self.builtin = qwen35_spec.builtin
        qwen35_spec.builtin = lambda name: self.machine.builtin(name)
        self.fresh()

    def tearDown(self):
        qwen35_spec.builtin = self.builtin

    def fresh(self):
        self.machine = Machine()
        self.target = program(self.machine)

    def check(self, read):
        """Cache, recurrent state, the head's layer and the kept hidden state agree on `read`."""
        machine, target = self.machine, self.target
        rows = machine.rows[target.sequence]
        self.assertEqual(machine.tokens(target.sequence), read)
        self.assertEqual(machine.read[LIVE], read)
        self.assertEqual(target.length, len(read))
        for index, (token, by_target, pair) in enumerate(rows):
            self.assertTrue(by_target)
            self.assertEqual(pair, (token, machine.hidden(read[:index])))
        self.assertEqual(int(target.hidden.numpy().reshape(-1)[0]), machine.hidden(read))

    def test_tokens_and_states_for_every_width_and_limit(self):
        for proposals in (1, 2, 3):
            for start in (1, 5, 12, 20):
                for limit in range(1, 30):
                    self.fresh()
                    decoder = Decoder(self.target, None, proposals)
                    prompt = list(range(30, 30 + start))
                    result = decoder.generate(prompt, limit, {STOP})
                    with self.subTest(proposals=proposals, start=start, limit=limit):
                        self.assertEqual(result["tokens"], reference(prompt, limit))
                        self.check(prompt + result["tokens"][:-1])
                        self.assertLessEqual(result["accepted"], result["proposed"])
                        self.assertEqual(self.machine.steps, result["proposed"])

    def test_some_proposals_are_accepted(self):
        decoder = Decoder(self.target, None, 3)
        result = decoder.generate(list(range(30, 35)), 16, {STOP})
        self.assertGreater(result["accepted"], 0)
        self.assertLess(result["accepted"], result["proposed"])

    def test_a_conversation_continues_from_its_end_or_from_its_prompt(self):
        decoder = Decoder(self.target, None, 3)
        prompt = list(range(30, 35))
        first = decoder.generate(prompt, 8, {STOP})
        longer = prompt + first["tokens"] + [50, 51]
        result = decoder.generate(longer, 8, {STOP})
        self.assertEqual((result["resumed"], result["cached_tokens"]), ("live", 12))
        self.assertEqual(result["tokens"], reference(longer, 8))
        self.check(longer + result["tokens"][:-1])
        # Another answer to the last prompt: the hidden state of its end is taken again.
        edited = [*longer, 60]
        result = decoder.generate(edited, 8, {STOP})
        self.assertEqual((result["resumed"], result["cached_tokens"]), ("anchor", len(longer)))
        self.assertEqual(result["tokens"], reference(edited, 8))
        self.check(edited + result["tokens"][:-1])
        result = decoder.generate(edited, 8, {STOP})
        self.assertEqual((result["resumed"], result["prefill_tokens"]), ("anchor", 0))
        self.assertEqual(result["tokens"], reference(edited, 8))
        self.check(edited + result["tokens"][:-1])
        other = [1, 2, 3]
        result = decoder.generate(other, 8, {STOP})
        self.assertEqual((result["resumed"], result["tokens"]), (None, reference(other, 8)))
        self.check(other + result["tokens"][:-1])

    def test_a_pinned_block_begins_other_conversations(self):
        decoder = Decoder(self.target, None, 3)
        block = list(range(100, 120))

        def answer(prompt, pinned=len(block)):
            result = decoder.generate(prompt, 8, {STOP}, block=pinned)
            self.assertEqual(result["tokens"], reference(prompt, 8))
            self.check(prompt + result["tokens"][:-1])
            return result["resumed"], result["cached_tokens"], result["tokens"]

        first = [*block, 30, 31, 32, 33]
        resumed, cached, tokens = answer(first)
        self.assertEqual((resumed, cached), (None, 0))
        self.assertEqual(answer([*first, *tokens, 50])[:2], ("live", len(first) + 7))
        second = [*block, 40, 41]
        self.assertEqual(answer(second)[:2], ("pinned", 20))
        self.assertEqual(answer([*second, 42])[0], "anchor")
        self.assertEqual(answer([*block, 60])[:2], ("pinned", 20))
        # A conversation without the block is read beside it and leaves it there.
        other = [1, 2, 3]
        resumed, _, tokens = answer(other, 0)
        self.assertEqual((resumed, self.target.sequence), (None, OPEN))
        self.assertEqual(answer([*other, *tokens, 4], 0)[0], "live")
        self.assertEqual(answer(second)[:2], ("pinned", 20))
        self.assertEqual(self.target.sequence, BLOCK)

    def test_a_long_prompt_is_read_in_chunks(self):
        decoder = Decoder(self.target, None, 3)
        prompt = [30 + index % 20 for index in range(21)]  # chunks of 8, 8 and 5 tokens
        result = decoder.generate(prompt, 8, {STOP})
        self.assertEqual(result["tokens"], reference(prompt, 8))
        self.check(prompt + result["tokens"][:-1])

    def test_one_source_of_proposals(self):
        with self.assertRaises(AssertionError):
            Decoder(self.target, self.target, 3)


if __name__ == "__main__":
    unittest.main()
