"""Small vocabulary selection kernel for the E4B shared-KV assistant."""

from pathlib import Path

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn.exporter import Exporter


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
