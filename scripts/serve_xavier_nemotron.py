"""Serve the validated Xavier Nemotron/DFlash configuration on loopback."""

import json
import os
from pathlib import Path

import fastapi
import mlc_llm
import uvicorn
from gemma4_tools import Gemma4ToolAdapter
from mlc_llm.base import _LIB_PATH
from mlc_llm.protocol import error_protocol
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from mlc_llm.serve import AsyncMLCEngine, EngineConfig
from mlc_llm.serve.entrypoints import openai_entrypoints
from mlc_llm.serve.server import ServerContext
from xavier_serve_queue import GenerationQueue, SharedGenerationQueue


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
                            "message": "This Xavier service supports temperature=0 and n=1.",
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
mtp = None
if os.environ.get("MLC_GEMMA4_MTP") == "1":
    from gemma4_mtp_service import MTPChat

    # Reserve/warm the larger target's workspace before the second engine's
    # allocations. Xavier NvMap can reject a large late startup allocation.
    mtp = MTPChat(root)
engine = AsyncMLCEngine(
    str(root / "dist/target-16k"),
    device="cuda",
    model_lib=str(root / "dist/libs/target-dflash-dp4a-attn-16k-sm72.so"),
    mode="interactive",
    engine_config=EngineConfig(
        additional_models=[
            (str(root / "dist/draft-16k"), str(root / "dist/libs/draft-cached-tc-16k-sm72.so"))
        ],
        max_num_sequence=1,
        max_total_sequence_length=16384,
        max_single_sequence_length=16384,
        prefill_chunk_size=512,
        prefix_cache_mode="disable",
        speculative_mode="dflash",
        spec_draft_length=3,
    ),
)
if receipt is not None and receipt["target_caches"] != 1:
    raise RuntimeError(f"Unexpected cache installation: {receipt}")
tool_adapter = Gemma4ToolAdapter(
    engine,
    os.environ.get(
        "MLC_NEMOTRON_TOKENIZER", str(Path.home() / "Models/nemotron-3.5-Bolt-E4B-BF16")
    ),
)
queue = GenerationQueue()
with ServerContext() as context:
    context.add_model("nemotron", engine)
    context.api_key = key
    app = fastapi.FastAPI(title="Cerebellum Nemotron API", docs_url=None, redoc_url=None)
    app.add_middleware(GreedyOnlyMiddleware)
    app.add_middleware(SharedGenerationQueue, queue=queue, api_key=key)

    @app.post(
        "/v1/chat/completions", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)]
    )
    async def chat(request: ChatCompletionRequest, raw_request: fastapi.Request):
        if mtp is not None and request.model == mtp.model_id:
            queue.active_model = mtp.model_id
            return await mtp.handle(request)
        if request.model not in (None, "nemotron", "nemotron:e4b"):
            raise fastapi.HTTPException(404, "Unknown model; see /v1/models")
        queue.active_model = "nemotron"
        if request.tools or any(m.tool_calls or m.role == "tool" for m in request.messages):
            return await tool_adapter.handle(request, raw_request)
        return await openai_entrypoints.request_chat_completion(request, raw_request)

    @app.post("/v1/completions", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def complete(request: CompletionRequest, raw_request: fastapi.Request):
        if mtp is not None and request.model == mtp.model_id:
            raise fastapi.HTTPException(400, "gemma4-12b supports /v1/chat/completions only")
        if request.model not in (None, "nemotron", "nemotron:e4b"):
            raise fastapi.HTTPException(404, "Unknown model; see /v1/models")
        queue.active_model = "nemotron"
        return await openai_entrypoints.request_completion(request, raw_request)

    @app.get("/v1/models", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def models():
        entries = [dict(id="nemotron", object="model", owned_by="cerebellum", context_length=16384)]
        if mtp is not None:
            entries.append(
                dict(
                    id=mtp.model_id,
                    object="model",
                    owned_by="cerebellum",
                    context_length=mtp.context_length,
                )
            )
        return dict(object="list", data=entries)

    app.include_router(openai_entrypoints.app)
    app.exception_handler(error_protocol.BadRequestError)(error_protocol.bad_request_error_handler)

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {
            "status": "ok",
            "model": "nemotron",
            "models": ["nemotron"] + ([mtp.model_id] if mtp is not None else []),
            "queue": queue.status(),
        }

    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)
    finally:
        if mtp is not None:
            mtp.pool.shutdown(wait=True)
