"""CPU checks for the service/client prompt boundary."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from xavier_service_prompt import POLICY, SEPARATOR, prepend_policy


class ServicePromptTest(unittest.TestCase):
    def test_preserves_client_and_history(self):
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "proof", "tool_call_id": "a"},
        ]
        actual = prepend_policy(messages)
        self.assertEqual(actual[0]["content"], POLICY + SEPARATOR + "Return JSON only.")
        self.assertEqual(actual[1:], messages[1:])
        self.assertEqual(messages[0]["content"], "Return JSON only.")

    def test_text_blocks_and_no_system(self):
        for content in ("client", [{"type": "text", "text": "client"}]):
            self.assertTrue(
                prepend_policy([{"role": "system", "content": content}])[0]["content"].endswith(
                    "client"
                )
            )
        actual = prepend_policy([{"role": "user", "content": "Hi"}])
        self.assertTrue(actual[0]["content"].startswith(POLICY))
        self.assertEqual(actual[1]["content"], "Hi")

    def test_reject_nontext_system(self):
        with self.assertRaises(ValueError):
            prepend_policy([{"role": "system", "content": [{"type": "image_url"}]}])


if __name__ == "__main__":
    unittest.main()
