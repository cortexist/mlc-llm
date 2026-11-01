"""Wrap paged-attention functions while retaining their prefill fallbacks."""

from pathlib import Path

import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn


def source_module():
    return nn.SourceModule(
        {
            "xavier_local_paged_attention": lambda *args: None,
            "xavier_global_paged_attention": lambda *args: None,
            "xavier_wrap_attention": lambda *args: None,
        },
        Path(__file__).parent / "cuda/xavier_attention.cu",
        "cu",
        compile_options=[
            *nn.SourceModule.get_compile_options("cu"),
            "-arch=sm_72",
            "-DMLC_XAVIER_FFI=1",
        ],
        compiler="nvcc",
    )


def wrap_creation(expr, lookup):
    """Wrap both paged prefill callbacks, retaining their larger-batch kernels."""
    count = 0

    def rewrite(value):
        nonlocal count
        if isinstance(value, relax.Var):
            binding = lookup(value)
            return rewrite(binding) if binding is not None else value
        if isinstance(value, relax.Tuple):
            fields = value.fields
            if (
                len(fields) == 2
                and isinstance(fields[0], tvm_tirx.ir.StringImm)
                and fields[0].value == "tirx"
                and isinstance(fields[1], tvm_tirx.ir.GlobalVar)
                and fields[1].name_hint
                in ["tir_attention_prefill", "tir_attention_prefill_sliding_window"]
            ):
                count += 1
                callback = relax.call_pure_packed(
                    "xavier_wrap_attention",
                    fields[1],
                    tvm_tirx.ir.IntImm(
                        "int64", int(fields[1].name_hint.endswith("sliding_window"))
                    ),
                    ty_args=relax.AnyType(),
                )
                return relax.Tuple([fields[0], callback])
            return relax.Tuple([rewrite(field) for field in fields])
        if isinstance(value, relax.Call):
            return relax.Call(
                value.op,
                [rewrite(arg) for arg in value.args],
                value.attrs,
                value.ty_args,
            )
        return value

    result = rewrite(expr)
    if count != 2:
        raise ValueError(f"Expected two paged-attention callbacks, found {count}")
    return result
