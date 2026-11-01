"""Serve the validated Xavier Nemotron/DFlash configuration on loopback."""

import json
import os
from pathlib import Path

import fastapi
import uvicorn

import mlc_llm
from mlc_llm.base import _LIB_PATH
from mlc_llm.protocol import error_protocol
from mlc_llm.serve import AsyncMLCEngine, EngineConfig
from mlc_llm.serve.entrypoints import openai_entrypoints
from mlc_llm.serve.server import ServerContext


class GreedyOnlyMiddleware:
    """Keep requests within the validated single-output Xavier DFlash sampler."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in ("/v1/chat/completions", "/v1/completions")
        ):
            return await self.app(scope, receive, send)
        request = fastapi.Request(scope, receive)
        body = await request.body()
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            payload = None
        if isinstance(payload, dict):
            if payload.get("temperature") not in (None, 0) or payload.get("n", 1) != 1:
                response = fastapi.responses.JSONResponse(
                    {
                        "error": {
                            "message": "This Xavier DFlash service supports temperature=0 and n=1.",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=400,
                )
                return await response(scope, receive, send)
            payload["temperature"] = 0
            body = json.dumps(payload).encode()
            scope = dict(scope)
            scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"content-length"]
        pending = True

        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


root = Path(__file__).resolve().parents[1]
if not Path(mlc_llm.__file__).resolve().is_relative_to(root / "python"):
    raise RuntimeError("Serve from the validated Xavier source checkout")
if not Path(_LIB_PATH).resolve().is_relative_to(root / "build"):
    raise RuntimeError("Serve with the validated Xavier runtime library")
key = Path(os.environ["MLC_NEMOTRON_API_KEY_FILE"]).read_text().strip()
if not key:
    raise ValueError("An API key is required")
variant = os.environ.get("MLC_NEMOTRON_PREFILL", "hybrid")
if variant != "baseline":
    from mlc_llm.op.xavier_prefill import install

    receipt = install(variant)
else:
    receipt = None
engine = AsyncMLCEngine(
    str(root / "dist/target"),
    device="cuda",
    model_lib=str(root / "dist/libs/target-dflash-dp4a-attn-sm72.so"),
    mode="interactive",
    engine_config=EngineConfig(
        additional_models=[
            (str(root / "dist/draft"), str(root / "dist/libs/draft-cached-tc-sm72.so"))
        ],
        max_num_sequence=1,
        max_total_sequence_length=8192,
        max_single_sequence_length=8192,
        prefill_chunk_size=512,
        prefix_cache_mode="disable",
        speculative_mode="dflash",
        spec_draft_length=3,
    ),
)
if receipt is not None and receipt["target_caches"] != 1:
    raise RuntimeError(f"Unexpected cache installation: {receipt}")
with ServerContext() as context:
    context.add_model("nemotron", engine)
    context.api_key = key
    app = fastapi.FastAPI(title="Cerebellum Nemotron API", docs_url=None, redoc_url=None)
    app.add_middleware(GreedyOnlyMiddleware)
    app.include_router(openai_entrypoints.app)
    app.exception_handler(error_protocol.BadRequestError)(error_protocol.bad_request_error_handler)

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok", "model": "nemotron"}

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)
