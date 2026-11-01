"""Opt-in sm_72 small-batch kernels, with the original checkpoint weight layout."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn.exporter import Exporter

_ACTIVE = ContextVar("xavier_small_batch", default=False)


@contextmanager
def small_batch():
    token = _ACTIVE.set(os.environ.get("MLC_XAVIER_Q4") == "1")
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def enabled(config):
    return _ACTIVE.get() and config.name == "q4f16_1"


def q4_linear(x, weight, scale, out_dtype=None):
    from .extern import get_store

    target = get_store().target
    if target is not None and (target.kind.name != "cuda" or target.attrs.get("arch") != "sm_72"):
        raise ValueError("MLC_XAVIER_Q4 is restricted to sm_72")
    exporter = Exporter.current()
    if not hasattr(exporter, "_xavier_q4"):

        def infer(a, w, _s):
            k = a.shape[-1]
            return (
                nn.Tensor.placeholder((*a.shape[:-1], w.shape[0]), dtype=out_dtype or a.dtype),
                nn.Tensor.placeholder((*a.shape[:-1], k // 2), dtype="uint32"),
                nn.Tensor.placeholder((*a.shape[:-1], k // 32, 2), dtype="float32"),
            )

        # Inference is selected per call below because the LM head returns FP32.
        module = nn.SourceModule(
            {"xavier_q4": infer},
            Path(__file__).parent / "cuda/xavier_q4.cu",
            "cu",
            compile_options=[
                *nn.SourceModule.get_compile_options("cu"),
                "-arch=sm_72",
                "-DMLC_XAVIER_FFI=1",
            ],
            compiler="nvcc",
        )
        nn.add_extern(module)
        exporter._xavier_q4 = module

    # nn.extern has per-call shape inference, unlike a shared SourceModule symbol.
    shape = (*x.shape[:-1], weight.shape[0])
    k = x.shape[-1]
    result = nn.op.extern(
        "xavier_q4",
        args=[x, weight, scale],
        out=[
            nn.Tensor.placeholder(shape, dtype=out_dtype or x.dtype),
            nn.Tensor.placeholder((*x.shape[:-1], k // 2), dtype="uint32"),
            nn.Tensor.placeholder((*x.shape[:-1], k // 32, 2), dtype="float32"),
        ],
    )
    return result[0]
