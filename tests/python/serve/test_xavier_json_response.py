"""JSON requests never reach DFlash's grammar matcher; invalid output fails closed."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import fastapi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from xavier_json_response import json_chat, validated_object


class JsonTest(unittest.IsolatedAsyncioTestCase):
    def test_rejects_invalid_incomplete_and_nonobject(self):
        for text in ['{"a":', "[]", '{"p":NaN}', 'Here is {"a":1}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validated_object(text)
        self.assertEqual(json.loads(validated_object('```json\n{"a":1}\n```')), {"a": 1})

    async def test_disables_grammar_and_checks_output(self):
        request = SimpleNamespace(
            stream=False,
            logprobs=False,
            response_format=SimpleNamespace(json_schema=None),
            model_copy=lambda update: SimpleNamespace(**update),
        )

        async def valid(copy, raw):
            self.assertIsNone(copy.response_format)
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"choices": [{"message": {"content": '{"ok":true}'}}]}
            )

        response = await json_chat(request, None, valid)
        self.assertEqual(
            json.loads(response.body)["choices"][0]["message"]["content"], '{"ok": true}'
        )

        async def invalid(copy, raw):
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"choices": [{"message": {"content": '{"ok":'}}]}
            )

        with self.assertRaises(fastapi.HTTPException) as error:
            await json_chat(request, None, invalid)
        self.assertEqual(error.exception.status_code, 502)

    async def test_unsupported_modes_never_call_backend(self):
        for stream, schema, logprobs in [
            (True, None, False),
            (False, "{}", False),
            (False, None, True),
        ]:
            request = SimpleNamespace(
                stream=stream,
                logprobs=logprobs,
                response_format=SimpleNamespace(json_schema=schema),
            )
            with self.assertRaises(fastapi.HTTPException) as error:
                await json_chat(request, None, None)
            self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
