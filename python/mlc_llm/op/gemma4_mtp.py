"""Vocabulary selection and attention kernels for the E4B shared-KV assistant."""

from pathlib import Path

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn.exporter import Exporter


def _attention_outputs(query, key, window):
    heads, groups, dim = query.shape[1:]
    capacity = key.shape[1]
    chunks = ((window + 1 if window else capacity) + 63) // 64
    return [
        nn.Tensor.placeholder(query.shape, "float16"),
        # Scratch: scores, weights and per-tile sums.
        nn.Tensor.placeholder((heads * groups, capacity), "float32"),
        nn.Tensor.placeholder((heads * groups, capacity), "float16"),
        nn.Tensor.placeholder((heads * groups, chunks, dim), "float32"),
    ]


def shared_attention(query, key, value, length, window):
    """Attend from one position to the valid keys of [1, capacity, 2, D] shared K/V.

    The cost follows the context length (or the window), not the capacity.
    """
    if window not in (0, 512):
        raise ValueError("The assistant attention kernel has a 512-token local window")
    exporter = Exporter.current()
    if not hasattr(exporter, "_gemma4_mtp_attention"):
        module = nn.SourceModule(
            {
                "gemma4_mtp_attention_local": lambda q, k, _v, _n: _attention_outputs(q, k, 512),
                "gemma4_mtp_attention_global": lambda q, k, _v, _n: _attention_outputs(q, k, 0),
            },
            Path(__file__).parent / "cuda/gemma4_mtp_attention.cu",
            "cu",
            compile_options=[*nn.SourceModule.get_compile_options("cu"), "-arch=sm_72"],
            compiler="nvcc",
        )
        nn.add_extern(module)
        exporter._gemma4_mtp_attention = module
    name = "gemma4_mtp_attention_local" if window else "gemma4_mtp_attention_global"
    return nn.op.extern(
        name, args=[query, key, value, length], out=_attention_outputs(query, key, window)
    )[0]


def centroid_topk(scores):
    """Select 32 of 2048 finite FP16 scores; equal scores prefer lower IDs."""
    exporter = Exporter.current()
    if not hasattr(exporter, "_gemma4_mtp_topk"):
        module = nn.SourceModule(
            {"gemma4_mtp_topk": lambda x: nn.Tensor.placeholder((1, 32), "int32")},
            Path(__file__).parent / "cuda/gemma4_mtp_topk.cu",
            "cu",
            compile_options=[*nn.SourceModule.get_compile_options("cu"), "-arch=sm_72"],
            compiler="nvcc",
        )
        nn.add_extern(module)
        exporter._gemma4_mtp_topk = module
    return nn.op.extern(
        "gemma4_mtp_topk", args=[scores], out=nn.Tensor.placeholder((1, 32), "int32")
    )
