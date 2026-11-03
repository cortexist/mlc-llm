"""Buffered JSON-object compatibility for the Xavier DFlash deployment.

The experimental DFlash path can commit a token rejected by XGrammar, killing
its worker thread. Generate without grammar constraints and validate the whole
object before returning it. Invalid output fails explicitly; it is never used
as a mail classification. This does not implement schema-constrained decoding.
"""

import json

import fastapi


def validated_object(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    value = json.loads(text, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


async def json_chat(request, raw_request, backend):
    if request.stream or request.response_format.json_schema is not None:
        raise fastapi.HTTPException(
            400, "Xavier JSON mode supports buffered json_object only, without a schema"
        )
    if request.logprobs:
        raise fastapi.HTTPException(400, "Validated JSON output does not support logprobs")
    generated = await backend(request.model_copy(update={"response_format": None}), raw_request)
    if isinstance(generated, fastapi.responses.Response):
        return generated  # Preserve a backend error/disconnection response.
    payload = generated.model_dump(mode="json")
    try:
        for choice in payload["choices"]:
            choice["message"]["content"] = validated_object(choice["message"]["content"])
    except (ValueError, TypeError, KeyError) as exc:
        raise fastapi.HTTPException(
            502, "Model did not return a complete JSON object; no result was accepted"
        ) from exc
    return fastapi.responses.JSONResponse(
        payload, headers={"X-Xavier-JSON-Mode": "validated-output"}
    )
