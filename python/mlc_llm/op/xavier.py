"""Opt-in sm_72 projection kernels, with the original checkpoint weight layout."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn.exporter import Exporter

# MLC_XAVIER_Q4 selects the small-batch kernel: "1" applies the scales of a product per
# chunk of 8 weights, "2" per group of 32. They round differently.
_KERNELS = {"1": "xavier_q4", "2": "xavier_q4_grouped"}
_ACTIVE = ContextVar("xavier_small_batch", default=None)
_PREFILL = ContextVar("xavier_prefill", default=False)


@contextmanager
def small_batch():
    token = _ACTIVE.set(_KERNELS.get(os.environ.get("MLC_XAVIER_Q4")))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def enabled(config):
    return _ACTIVE.get() is not None and config.name == "q4f16_1"


@contextmanager
def prefill():
    token = _PREFILL.set(os.environ.get("MLC_XAVIER_DEQUANT") == "1")
    try:
        yield
    finally:
        _PREFILL.reset(token)


def dequant_enabled(config):
    return _PREFILL.get() and _ACTIVE.get() is None and config.name == "q4f16_1"


def dequant_weight(weight, scale):
    """Vectorized expansion; retain the existing FP16 weights and BLAS arithmetic."""
    from .extern import get_store

    target = get_store().target
    if target is not None and (target.kind.name != "cuda" or target.attrs.get("arch") != "sm_72"):
        raise ValueError("MLC_XAVIER_DEQUANT is restricted to sm_72")
    exporter = Exporter.current()
    if not hasattr(exporter, "_xavier_dequant"):
        module = nn.SourceModule(
            {
                "xavier_dequant": lambda w, s: nn.Tensor.placeholder(
                    (w.shape[0], w.shape[1] * 8), "float16"
                )
            },
            Path(__file__).parent / "cuda/xavier_dequant.cu",
            "cu",
            compile_options=[*nn.SourceModule.get_compile_options("cu"), "-arch=sm_72"],
            compiler="nvcc",
        )
        nn.add_extern(module)
        exporter._xavier_dequant = module
    return nn.op.extern(
        "xavier_dequant",
        args=[weight, scale],
        out=nn.Tensor.placeholder((weight.shape[0], weight.shape[1] * 8), "float16"),
    )


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
            dict.fromkeys(_KERNELS.values(), infer),
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
        _ACTIVE.get(),
        args=[x, weight, scale],
        out=[
            nn.Tensor.placeholder(shape, dtype=out_dtype or x.dtype),
            nn.Tensor.placeholder((*x.shape[:-1], k // 2), dtype="uint32"),
            nn.Tensor.placeholder((*x.shape[:-1], k // 32, 2), dtype="float32"),
        ],
    )
    return result[0]
