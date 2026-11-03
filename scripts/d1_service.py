"""Optional bounded d1 text-decision worker; executed under the shared GPU FIFO."""

import os
import time
from pathlib import Path

import fastapi
import numpy as np
from d1_inputs import DecisionInputs
from embeddinggemma2_service import ResidentEmbedding
from pydantic import BaseModel, ConfigDict, JsonValue


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "d1-omni-2b"
    state: JsonValue
    questions: dict


class ResidentDecision(ResidentEmbedding):
    model_id = "d1-omni-2b"
    context_length = 256

    def load(self):
        import tvm_tirx
        from bench_xavier_mtp import Program

        self.device = tvm_tirx.cuda(0)
        self.tensor = tvm_tirx.runtime.tensor
        self.context_length = int(os.environ.get("MLC_D1_CONTEXT", "256"))
        if self.context_length not in (256, 512, 1024, 2048):
            raise ValueError("Unsupported d1 capacity")
        self.inputs = DecisionInputs(
            Path(os.environ.get("MLC_D1_CHECKPOINT", str(Path.home() / "Models/d1-Omni-2B"))),
            self.context_length,
        )
        self.rows = {}
        self.precision = os.environ.get("MLC_D1_PRECISION", "f32")
        if self.precision not in ("f32", "f16"):
            raise ValueError("Unsupported d1 precision")
        self.model = Program(
            self.root / f"dist/d1-text-q0{self.precision}",
            self.root / f"dist/libs/d1-text-{self.precision}-s{self.context_length}-sm72.so",
        )
        if self.model.context_length != self.context_length:
            raise RuntimeError("d1 library capacity mismatch")
        self.decide(
            "The network is down.",
            {"network": dict(type="noul", instructions="Is this a network issue?")},
        )
        print(f"Resident {self.model_id} loaded once: {self.instance}", flush=True)

    def embedding_rows(self, ids):
        from safetensors import safe_open

        missing = [i for i in ids if i not in self.rows]
        if missing:
            with safe_open(
                self.inputs.source / "model.safetensors", framework="pt", device="cpu"
            ) as f:
                selected = (
                    f.get_tensor("model.language_model.embed_tokens.weight")[missing]
                    .float()
                    .numpy()
                )
            self.rows.update(zip(missing, selected))
        return np.array([self.rows[i] for i in ids])

    def decide(self, state, questions):
        start = time.perf_counter()
        packed = self.inputs.pack(state, questions)
        tensors = [self.tensor(packed[k], self.device) for k in ("tokens", "mask", "groups")]
        self.device.sync()
        begin = time.perf_counter()
        hidden = self.model.call("prefill", *tensors).numpy()
        forward_ms = 1000 * (time.perf_counter() - begin)
        if not np.isfinite(hidden).all():
            raise RuntimeError("Nonfinite d1 backbone output")
        payload = self.inputs.read(packed, hidden, self.embedding_rows)
        payload["model"] = self.model_id
        return payload, dict(inference_ms=forward_ms, total_ms=1000 * (time.perf_counter() - start))

    async def handle(self, request):
        try:
            payload, metrics = await self.work(self.decide, request.state, request.questions)
        except (ValueError, TypeError, KeyError) as exc:
            raise fastapi.HTTPException(400, str(exc)) from exc
        return fastapi.responses.JSONResponse(
            payload,
            headers={
                "X-Xavier-Decision-Inference-Ms": str(metrics["inference_ms"]),
                "X-Xavier-Decision-Total-Ms": str(metrics["total_ms"]),
                "X-Xavier-Decision-Instance": self.instance,
            },
        )
