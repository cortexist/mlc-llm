"""E4B shared-KV assistant with its trained centroid-filtered vocabulary head."""

import dataclasses
import os

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm import op as op_ext
from mlc_llm.loader.standard_loader import make_standard_hf_loader
from mlc_llm.op import xavier

from .gemma4_config import _as_config_dict
from .gemma4_model import Gemma4ForConditionalGeneration, _slice_head_dim
from .gemma4_mtp import (
    AssistantTextConfig,
    Gemma4UnifiedAssistantConfig,
    Gemma4UnifiedAssistantForCausalLM,
    Gemma4UnifiedMTPTarget,
)


@dataclasses.dataclass
class E4BAssistantConfig(Gemma4UnifiedAssistantConfig):
    backbone_hidden_size: int = 2560
    use_ordered_embeddings: bool = True
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32

    def __post_init__(self):
        text = dict(_as_config_dict(self.text_config, {}))
        text["num_global_key_value_heads"] = (
            text.get("num_global_key_value_heads") or text["num_key_value_heads"]
        )
        self.text_config = AssistantTextConfig.from_dict(text)
        if (
            self.text_config.num_attention_heads,
            self.text_config.num_key_value_heads,
            self.text_config.num_global_key_value_heads,
        ) != (4, 2, 2):
            raise ValueError("Expected E4B assistant attention heads")
        if (
            not self.use_ordered_embeddings
            or self.tensor_parallel_shards != 1
            or self.max_batch_size != 1
        ):
            raise ValueError("E4B MTP requires the ordered head and one device/sequence")
        if (
            self.vocab_size % self.num_centroids
            or not 0 < self.centroid_intermediate_top_k <= self.num_centroids
        ):
            raise ValueError("Invalid centroid configuration")
        if not 0 < self.context_window_size <= self.text_config.max_position_embeddings:
            raise ValueError("Invalid context window")


class MaskedEmbedding(nn.Module):
    def __init__(self, config):
        self.centroids = nn.Linear(config.text_config.hidden_size, config.num_centroids, bias=False)
        self.token_ordering = nn.Parameter((config.vocab_size,), "int32")

    def to(self, dtype=None):
        self.centroids.to(dtype=dtype)


class E4BAssistant(Gemma4UnifiedAssistantForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.masked_embedding = MaskedEmbedding(config)

    def mtp_step(
        self,
        embedding,
        previous_hidden,
        local_key,
        local_value,
        global_key,
        global_value,
        position,
        context_length,
    ):
        hidden = self.pre_projection(op.concat([embedding, previous_hidden], dim=-1))
        for i, layer in enumerate(self.model.layers):
            key, value = (
                (global_key, global_value)
                if self.config.text_config.layer_types[i] == "full_attention"
                else (local_key, local_value)
            )
            hidden = layer(hidden, key, value, position, context_length)
        hidden = self.model.norm(hidden)
        scores = op.reshape(self.masked_embedding.centroids(hidden), (1, self.config.num_centroids))
        if os.environ.get("MLC_E4B_MTP_TOPK") == "1":
            from mlc_llm.op.gemma4_mtp import centroid_topk

            if (self.config.num_centroids, self.config.centroid_intermediate_top_k, self.dtype) != (
                2048,
                32,
                "float16",
            ):
                raise ValueError("Parallel centroid selection requires FP16 E4B 2048/32")
            clusters = centroid_topk(scores)
        else:
            _, clusters = op_ext.moe_misc.gating_topk(
                scores, self.config.centroid_intermediate_top_k
            )
        ordering = op.reshape(
            self.masked_embedding.token_ordering,
            (self.config.num_centroids, self.config.vocab_size // self.config.num_centroids),
        )
        ids = op.reshape(op.take(ordering, op.reshape(clusters, (-1,)), axis=0), (-1,))
        weights = op.take(self.lm_head.weight, ids, axis=0)
        logits = op.matmul(op.reshape(hidden, (1, -1)), op.permute_dims(weights))
        return logits.astype("float32"), self.post_projection(hidden), ids

    def get_default_spec(self):
        context = self.config.context_window_size
        hidden = self.config.backbone_hidden_size
        return nn.spec.ModuleSpec.from_raw(
            {
                "mtp_step": {
                    "embedding": nn.spec.Tensor([1, 1, hidden], self.dtype),
                    "previous_hidden": nn.spec.Tensor([1, 1, hidden], self.dtype),
                    "local_key": nn.spec.Tensor([1, context, 2, 256], self.dtype),
                    "local_value": nn.spec.Tensor([1, context, 2, 256], self.dtype),
                    "global_key": nn.spec.Tensor([1, context, 2, 512], self.dtype),
                    "global_value": nn.spec.Tensor([1, context, 2, 512], self.dtype),
                    "position": nn.spec.Tensor([1], "int32"),
                    "context_length": nn.spec.Tensor([1], "int32"),
                    "$": {"param_mode": "packed", "effect_mode": "none"},
                }
            },
            self,
        )


def huggingface(config, quantization):
    mapping = make_standard_hf_loader(
        model_cls=E4BAssistant,
        layer_prefix="model.layers",
        include_qkv=False,
        gate_up_target_name="gate_up_proj",
        num_layers_getter=lambda cfg: cfg.text_config.num_hidden_layers,
    )(config, quantization)
    mapping.add_mapping(
        "lm_head.weight",
        ["model.embed_tokens.weight"],
        lambda weight: weight.astype(quantization.model_dtype),
    )
    mapping.add_mapping(
        "masked_embedding.token_ordering",
        ["masked_embedding.token_ordering"],
        lambda values: values.astype("int32"),
    )
    return mapping


class E4BMTPTarget(Gemma4ForConditionalGeneration):
    def _mtp_forward(self, tokens, cache, depth=None):
        op_ext.configure()
        body = self.language_model
        hidden = body.embed(tokens)
        positions = cache.get_query_positions(tokens.shape[0] * tokens.shape[1])
        per_layer = body._per_layer_inputs(hidden, tokens, None)
        shared = {}
        for i in range(depth or len(body.layers)):
            kind = self.config.text_config.layer_types[i]
            hidden, kv = body.layers[i](hidden, per_layer[i], cache, positions, shared.get(kind))
            if i in body.shared_kv_source_layers:
                shared[kind] = kv
        local = [_slice_head_dim(v, 256) for v in shared["sliding_attention"]]
        return body.norm(hidden), *local, *shared["full_attention"]

    @xavier.prefill()
    def mtp_prefill_kv(self, tokens, cache):
        """Prompt tokens that need no prediction: only the layers that store K/V run.

        The name matters: the cuBLAS dispatch selects functions named "prefill".
        So does returning the hidden state: the cache call is declared pure, and
        the last layer's would be removed, unwritten K/V and all, were its result unused.
        """
        hidden, *states = self._mtp_forward(
            tokens, cache, self.config.text_config.first_kv_shared_layer
        )
        return hidden, *states, cache

    mtp_prefill = Gemma4UnifiedMTPTarget.mtp_prefill
    mtp_verify = Gemma4UnifiedMTPTarget.mtp_verify

    def get_default_spec(self):
        spec = Gemma4UnifiedMTPTarget._mtp_spec(self)
        spec["mtp_prefill_kv"] = dict(spec["mtp_prefill"])
        return nn.spec.ModuleSpec.from_raw(spec, self)
