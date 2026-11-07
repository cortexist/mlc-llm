"""Serve resident Xavier Nemotron with DFlash or the Gemma E4B MTP adapter."""

import asyncio
import json
import os
from pathlib import Path

import fastapi
import uvicorn
from gemma4_tools import Gemma4ToolAdapter
from xavier_serve_queue import GenerationQueue, SharedGenerationQueue

import mlc_llm
from mlc_llm.base import _LIB_PATH
from mlc_llm.protocol import error_protocol
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
)
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

    receipt = install(variant, root / "dist/target-16k/mlc-chat-config.json")
else:
    receipt = None
mtp = None
if os.environ.get("MLC_GEMMA4_MTP") == "1":
    from gemma4_mtp_service import MTPChat

    # Reserve/warm the larger target's workspace before the second engine's
    # allocations. Xavier NvMap can reject a large late startup allocation.
    mtp = MTPChat(root)
embedding = None
decision = None
speculative = os.environ.get("MLC_NEMOTRON_SPECULATIVE", "dflash")
if speculative == "gemma_mtp":
    from nemotron_mtp_service import NemotronMTP

    engine = NemotronMTP(root)
elif speculative == "dflash":
    engine = AsyncMLCEngine(
        str(root / "dist/target-16k"),
        device="cuda",
        model_lib=str(
            root
            / os.environ.get(
                "MLC_NEMOTRON_MODEL_LIB", "dist/libs/target-dflash-dp4a-attn-16k-sm72.so"
            )
        ),
        mode="interactive",
        engine_config=EngineConfig(
            additional_models=[
                (str(root / "dist/draft-16k"), str(root / "dist/libs/draft-cached-tc-16k-sm72.so"))
            ],
            max_num_sequence=1,
            max_total_sequence_length=int(os.environ.get("MLC_NEMOTRON_KV_BUDGET", "16384")),
            max_single_sequence_length=16384,
            prefill_chunk_size=512,
            prefix_cache_mode=os.environ.get("MLC_NEMOTRON_PREFIX_CACHE", "disable"),
            prefix_cache_max_num_recycling_seqs=int(
                os.environ.get("MLC_NEMOTRON_PREFIX_SLOTS", "1")
            ),
            speculative_mode="dflash",
            spec_draft_length=3,
        ),
    )
else:
    raise ValueError("MLC_NEMOTRON_SPECULATIVE must be gemma_mtp or dflash")

if receipt is not None and receipt["target_caches"] != 1:
    raise RuntimeError(f"Unexpected cache installation: {receipt}")
tool_adapter = Gemma4ToolAdapter(
    engine,
    os.environ.get(
        "MLC_NEMOTRON_TOKENIZER", str(Path.home() / "Models/nemotron-3.5-Bolt-E4B-BF16")
    ),
)
server_prompt = None
if os.environ.get("MLC_NEMOTRON_SERVER_PROMPT") == "1":
    if os.environ.get("MLC_NEMOTRON_PREFIX_CACHE") != "radix":
        raise RuntimeError("A protected service prefix requires radix caching")
    if os.environ.get("MLC_XAVIER_LINEAR_FORK") != "1":
        raise RuntimeError("A protected service prefix requires safe linear KV copying")
    from xavier_service_prompt import VERSION, prefix_tokens, prepend_policy

    server_prompt = prefix_tokens(tool_adapter.tokenizer)
queue = GenerationQueue()
if speculative == "gemma_mtp":
    queue.drain = engine.drain
with ServerContext() as context:
    context.add_model("nemotron", engine)
    context.api_key = key
    app = fastapi.FastAPI(title="Xavier Nemotron API", docs_url=None, redoc_url=None)
    app.add_middleware(GreedyOnlyMiddleware)
    app.add_middleware(SharedGenerationQueue, queue=queue, api_key=key)

    @app.on_event("startup")
    async def warm_and_load_specialists():
        # Reserve the maximum prefill workspace before specialist allocations.
        # Late 1 GiB NvMap allocations can fail despite ample total free RAM.
        # Use the serving event loop: MLC retains its first async loop reference.
        global embedding, decision
        if speculative == "gemma_mtp":
            await engine.warm_and_pin(server_prompt)
        else:
            await asyncio.wait_for(
                engine.completions.create(
                    prompt=[100] * 512, model="nemotron", max_tokens=4, temperature=0
                ),
                timeout=90,
            )
            if server_prompt is not None:
                await engine.completions.create(
                    prompt=server_prompt,
                    model="nemotron",
                    max_tokens=1,
                    temperature=0,
                    extra_body={"debug_config": {"pinned_system_prompt": True}},
                )
        if server_prompt is not None:
            print(
                f"Protected service prefix: {len(server_prompt)} tokens, policy {VERSION}",
                flush=True,
            )
        if os.environ.get("MLC_EMBEDDINGGEMMA2") == "1":
            from embeddinggemma2_service import ResidentEmbedding

            embedding = await asyncio.to_thread(ResidentEmbedding, root)
        if os.environ.get("MLC_D1") == "1":
            from d1_service import ResidentDecision

            decision = await asyncio.to_thread(ResidentDecision, root)

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
        if server_prompt is not None:
            try:
                messages = prepend_policy(
                    [m.model_dump(exclude_none=True) for m in request.messages]
                )
                request = request.model_copy(
                    update={"messages": [ChatCompletionMessage.model_validate(m) for m in messages]}
                )
            except ValueError as exc:
                raise fastapi.HTTPException(400, str(exc)) from exc
        if request.tools or any(m.tool_calls or m.role == "tool" for m in request.messages):
            return await tool_adapter.handle(request, raw_request)
        if request.response_format is not None and request.response_format.type == "json_object":
            from xavier_json_response import json_chat

            return await json_chat(request, raw_request, openai_entrypoints.request_chat_completion)
        return await openai_entrypoints.request_chat_completion(request, raw_request)

    @app.post("/v1/completions", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def complete(request: CompletionRequest, raw_request: fastapi.Request):
        if request.response_format is not None and request.response_format.type == "json_object":
            raise fastapi.HTTPException(400, "Use buffered /v1/chat/completions for JSON objects")
        if mtp is not None and request.model == mtp.model_id:
            raise fastapi.HTTPException(400, "gemma4-12b supports /v1/chat/completions only")
        if request.model not in (None, "nemotron", "nemotron:e4b"):
            raise fastapi.HTTPException(404, "Unknown model; see /v1/models")
        queue.active_model = "nemotron"
        return await openai_entrypoints.request_completion(request, raw_request)

    @app.post("/v1/embeddings", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def embed(request: EmbeddingRequest):
        if embedding is None or request.model not in (None, embedding.model_id):
            raise fastapi.HTTPException(404, "Embedding model unavailable; see /v1/models")
        queue.active_model = embedding.model_id
        return await embedding.handle(request)

    from d1_service import DecisionRequest

    @app.post("/v1/decisions", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    @app.post("/v1/systemone", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def decide(request: DecisionRequest):
        if decision is None or request.model != decision.model_id:
            raise fastapi.HTTPException(404, "Decision model unavailable; see /v1/models")
        queue.active_model = decision.model_id
        return await decision.handle(request)

    @app.get("/v1/models", dependencies=[fastapi.Depends(openai_entrypoints.verify_api_key)])
    async def models():
        entries = [dict(id="nemotron", object="model", owned_by="xavier", context_length=16384)]
        if decision is not None:
            entries.append(
                dict(
                    id=decision.model_id,
                    object="model",
                    owned_by="xavier",
                    context_length=decision.context_length,
                    task="decision",
                )
            )
        if mtp is not None:
            entries.append(
                dict(
                    id=mtp.model_id,
                    object="model",
                    owned_by="xavier",
                    context_length=mtp.context_length,
                )
            )
        if embedding is not None:
            entries.append(
                dict(
                    id=embedding.model_id,
                    object="model",
                    owned_by="xavier",
                    context_length=embedding.context_length,
                    dimensions=embedding.dimension,
                    task="embedding",
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
            "models": ["nemotron"]
            + ([mtp.model_id] if mtp is not None else [])
            + ([embedding.model_id] if embedding is not None else [])
            + ([decision.model_id] if decision is not None else []),
            "queue": queue.status(),
            "speculative": engine.status()
            if speculative == "gemma_mtp"
            else {"mode": "dflash", "draft_tokens": 2},
            "service_prefix_tokens": len(server_prompt) if server_prompt else 0,
            "service_policy": VERSION if server_prompt is not None else None,
        }

    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", access_log=False)
    finally:
        if mtp is not None:
            mtp.pool.shutdown(wait=True)
        if embedding is not None:
            embedding.pool.shutdown(wait=True)
        if decision is not None:
            decision.pool.shutdown(wait=True)
