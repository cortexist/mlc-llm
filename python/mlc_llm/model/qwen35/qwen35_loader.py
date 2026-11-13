"""Weight mapping from Hugging Face to MLC for the Qwen3.5 text model.

Hugging Face stores the text weights under model.language_model, the MLC text model
under model. Q, K and V are fused into c_attn and gate and up into gate_up_proj. The
multi-token prediction head, when the model is built with it, is mtp in both.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen35_model import Qwen35Config, Qwen35LMHeadModel


def huggingface(model_config: Qwen35Config, quantization: Quantization) -> ExternMapping:
    model = Qwen35LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()
    map_language_model(mapping, named_parameters, model_config, "model.language_model", "model")

    def _mlc_to_hf(mlc_name: str) -> str:
        if mlc_name.startswith("model."):
            return mlc_name.replace("model.", "model.language_model.", 1)
        return mlc_name

    map_remaining(mapping, named_parameters, _mlc_to_hf, "model")
    return mapping


def map_language_model(  # pylint: disable=too-many-locals
    mapping: ExternMapping,
    named_parameters: dict,
    model_config: Qwen35Config,
    hf: str,
    mlc: str,
) -> None:
    """Add the fused and renamed language model weights under the given prefixes."""

    def cast(dtype):
        return functools.partial(lambda x, dtype: x.astype(dtype), dtype=dtype)

    layer_types = model_config.layer_types()
    # The block of the MTP head is one more full-attention layer, under a prefix of its own.
    blocks = [
        (f"{mlc}.layers.{i}", f"{hf}.layers.{i}", layer_types[i])
        for i in range(model_config.num_hidden_layers)
    ]
    blocks.append(("mtp.layers.0", "mtp.layers.0", "full_attention"))
    for mlc_layer, hf_layer, layer_type in blocks:
        if layer_type == "full_attention":
            mlc_attn = f"{mlc_layer}.self_attn"
            hf_attn = f"{hf_layer}.self_attn"
            mlc_name = f"{mlc_attn}.c_attn.weight"
            if mlc_name in named_parameters:
                mapping.add_mapping(
                    mlc_name,
                    [f"{hf_attn}.{p}_proj.weight" for p in "qkv"],
                    functools.partial(
                        lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                        dtype=named_parameters[mlc_name].dtype,
                    ),
                )
        else:
            mlc_lin = f"{mlc_layer}.linear_attn"
            hf_lin = f"{hf_layer}.linear_attn"
            # A_log and dt_bias carry no .weight suffix, and conv1d is stored flat.
            for mlc_suffix, hf_suffix in [
                ("in_proj_qkv.weight", "in_proj_qkv.weight"),
                ("A_log", "A_log"),
                ("dt_bias", "dt_bias"),
                ("conv1d_weight", "conv1d.weight"),
            ]:
                mlc_name = f"{mlc_lin}.{mlc_suffix}"
                if mlc_name in named_parameters:
                    mapping.add_mapping(
                        mlc_name,
                        [f"{hf_lin}.{hf_suffix}"],
                        cast(named_parameters[mlc_name].dtype),
                    )

        mlc_name = f"{mlc_layer}.mlp.gate_up_proj.weight"
        if mlc_name in named_parameters:
            mapping.add_mapping(
                mlc_name,
                [f"{hf_layer}.mlp.{p}_proj.weight" for p in ("gate", "up")],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=named_parameters[mlc_name].dtype,
                ),
            )


def map_remaining(mapping: ExternMapping, named_parameters: dict, mlc_to_hf, mlc: str) -> None:
    """Map every parameter not yet covered one to one, adding 1 to the RMSNorm weights.

    Qwen3.5's RMSNorm computes norm(x) * (1 + weight) and the gated norm in the linear
    attention layers computes norm(x) * weight, so only the former get the offset.
    """
    norms = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    whole = (
        f"{mlc}.norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    )

    def cast(x, dtype):
        return x.astype(dtype)

    def offset(x, dtype):
        return (x.astype("float32") + 1.0).astype(dtype)

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name in mapping.param_map:
            continue
        convert = offset if mlc_name.endswith(norms) or mlc_name in whole else cast
        mapping.add_mapping(
            mlc_name, [mlc_to_hf(mlc_name)], functools.partial(convert, dtype=mlc_param.dtype)
        )
