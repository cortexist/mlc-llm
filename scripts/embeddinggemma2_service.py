"""Resident text-only FP32 EmbeddingGemma 2 worker for the Xavier PoC."""

import asyncio
import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import anyio
import fastapi
import numpy as np


class ResidentEmbedding:
    model_id = "embeddinggemma-2"
    context_length = 128
    dimension = 768

    def __init__(self, root):
        self.root = root
        self.instance = uuid.uuid4().hex
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embeddinggemma2")
        try:
            self.pool.submit(self.load).result()
        except Exception:
            self.pool.shutdown(wait=True)
            raise

    def load(self):
        import tvm_tirx
        from bench_xavier_mtp import Program

        from mlc_llm.tokenizers import Tokenizer

        self.device = tvm_tirx.cuda(0)
        self.tensor = tvm_tirx.runtime.tensor
        self.model = Program(
            self.root / "dist/embeddinggemma2-text-q0f32",
            self.root / "dist/libs/embeddinggemma2-text-f32-s128-sm72.so",
        )
        if self.model.context_length != self.context_length:
            raise RuntimeError("Embedding library capacity mismatch")
        self.tokenizer = Tokenizer(str(self.root / "dist/embeddinggemma2-text-q0f32"))
        self.embed(["task: search result | query: a warmup query"], 768)
        print(f"Resident {self.model_id} loaded once: {self.instance}", flush=True)

    def embed(self, texts, dimensions):
        start = time.perf_counter()
        # MLC's tokenizer encodes content without HF post-processing. The
        # checkpoint's text template is BOS + content + EOS, all mean-pooled.
        encoded = [[2, *self.tokenizer.encode(text), 1] for text in texts]
        if any(not 0 < len(ids) <= self.context_length for ids in encoded):
            raise ValueError(
                "Embedding PoC accepts at most 128 tokens including BOS/EOS; no truncation"
            )
        tokenized = time.perf_counter()
        vectors = []
        forward = 0.0
        for ids in encoded:
            tokens = np.zeros((1, self.context_length), dtype="int32")
            mask = np.zeros_like(tokens)
            tokens[0, : len(ids)] = ids
            mask[0, : len(ids)] = 1
            gpu_ids = self.tensor(tokens, self.device)
            gpu_mask = self.tensor(mask, self.device)
            self.device.sync()
            begin = time.perf_counter()
            output = self.model.call("prefill", gpu_ids, gpu_mask).numpy()
            forward += time.perf_counter() - begin
            if not np.isfinite(output).all():
                raise RuntimeError("Non-finite FP32 encoder output")
            pooled = output[0, : len(ids)].mean(axis=0)[:dimensions]
            norm = float(np.linalg.norm(pooled))
            if not np.isfinite(norm) or norm == 0:
                raise RuntimeError("Invalid embedding norm")
            vectors.append(pooled / norm)
        return vectors, dict(
            tokens=sum(map(len, encoded)),
            tokenize_ms=1000 * (tokenized - start),
            inference_ms=1000 * forward,
            total_ms=1000 * (time.perf_counter() - start),
        )

    async def work(self, function, *args):
        future = asyncio.get_running_loop().run_in_executor(self.pool, function, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                await asyncio.shield(future)
            raise

    async def handle(self, request):
        texts = [request.input] if isinstance(request.input, str) else request.input
        dimensions = self.dimension if request.dimensions is None else request.dimensions
        if (
            not isinstance(texts, list)
            or not 0 < len(texts) <= 16
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise fastapi.HTTPException(400, "Embedding PoC accepts 1..16 nonempty text strings")
        if dimensions not in (128, 256, 512, 768) or request.encoding_format not in (
            "float",
            "base64",
        ):
            raise fastapi.HTTPException(
                400, "Supported dimensions: 128,256,512,768; encoding: float/base64"
            )
        try:
            vectors, metrics = await self.work(self.embed, texts, dimensions)
        except ValueError as exc:
            raise fastapi.HTTPException(400, str(exc)) from exc

        def encode(vector):
            return (
                vector.tolist()
                if request.encoding_format == "float"
                else base64.b64encode(vector.astype("<f4").tobytes()).decode()
            )

        payload = dict(
            object="list",
            model=self.model_id,
            data=[
                dict(object="embedding", index=i, embedding=encode(v))
                for i, v in enumerate(vectors)
            ],
            usage=dict(prompt_tokens=metrics["tokens"], total_tokens=metrics["tokens"]),
        )
        return fastapi.responses.JSONResponse(
            payload,
            headers={
                "X-Xavier-Embedding-Inference-Ms": str(metrics["inference_ms"]),
                "X-Xavier-Embedding-Total-Ms": str(metrics["total_ms"]),
                "X-Xavier-Embedding-Instance": self.instance,
            },
        )
