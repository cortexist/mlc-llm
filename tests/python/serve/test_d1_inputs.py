"""Bounded API input checks using the original checkpoint's tokenizer/schema."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from d1_inputs import DecisionInputs


@unittest.skipUnless(os.environ.get("D1_CHECKPOINT"), "requires local d1 checkpoint")
class InputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = DecisionInputs(os.environ["D1_CHECKPOINT"])

    def test_full_input_budget_includes_questions(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.inputs.pack(
                "network " * 250,
                {"q": dict(type="noul", instructions="Is this about a network? " * 15)},
            )

    def test_multiple_questions_and_padding(self):
        packed = self.inputs.pack(
            "DNS lookup failed.",
            {str(i): dict(type="noul", instructions="Is there a DNS issue?") for i in range(4)},
        )
        self.assertEqual(len(packed["slots"]), 4)
        self.assertEqual(packed["slots"][-1], packed["length"] - 1)
        self.assertTrue((packed["mask"][0, packed["length"] :] == 0).all())
        self.assertEqual(set(packed["groups"][0, : packed["length"]]), {-1, 1, 2, 3, 4})

    def test_rejects_unsupported_modalities_and_types(self):
        for questions in (
            {},
            {str(i): dict(type="noul", instructions="yes?") for i in range(5)},
            {"q": dict(type="text", instructions="Extract text")},
            {"q": dict(type="noul", instructions={"audio": "file.wav"})},
            {"q": dict(type="choice", instructions="Choose", multiple=True)},
        ):
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                self.inputs.pack("test", questions)

    def test_json_state_and_jev_criteria_use_native_formatter(self):
        inputs = DecisionInputs(os.environ["D1_CHECKPOINT"], context=2048)
        state = {"subject": "Résumé", "sender": {"address": "a@example.com"}}
        question = dict(
            type="noul",
            instructions="Is this spam?",
            criteria={"true": "Unwanted marketing", "false": "Wanted correspondence"},
        )
        packed = inputs.pack(state, {"spam": question})
        q = inputs.prompt.as_question(question)
        native = inputs.reader._sequences([(state, [q], [])])[0][2]
        self.assertEqual(packed["tokens"][0, : packed["length"]].tolist(), native)
        decoded = inputs.reader.tokenizer.decode(native)
        self.assertIn("Unwanted marketing", decoded)
        self.assertIn("Wanted correspondence", decoded)
        self.assertIn("a@example.com", decoded)

    def test_invalid_criteria_and_unknown_fields_are_rejected(self):
        for question in (
            dict(type="noul", instructions=""),
            dict(type="noul", instructions="yes?", criteria={"true": "yes"}),
            dict(type="noul", instructions="yes?", criteria={"true": {}, "false": "no"}),
            dict(type="noul", instructions="yes?", media=[]),
            dict(type="choice", instructions="pick", criteria={}),
            dict(type="choice", instructions="pick", criteria={"a": [], "b": "B"}),
        ):
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.inputs.pack({}, {"q": question})

    def test_two_k_budget_accepts_larger_state_and_rejects_overflow(self):
        inputs = DecisionInputs(os.environ["D1_CHECKPOINT"], context=2048)
        questions = {"q": dict(type="noul", instructions="Is there a network issue?")}
        packed = inputs.pack({"text": "network " * 600}, questions)
        self.assertGreater(packed["length"], 256)
        self.assertLess(packed["length"], 2048)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            inputs.pack({"text": "network " * 2100}, questions)


if __name__ == "__main__":
    unittest.main()
