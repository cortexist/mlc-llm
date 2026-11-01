"""CPU checks of the Unified 12B projection/normalization/RoPE port against HF.

Run with MLC_GEMMA4_TEXT_ONLY=1 in the MLC virtual environment.
"""

import os
import unittest

import numpy as np
import torch
import tvm_tirx
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedTextConfig as HFConfig,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextAttention as HFAttention,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextRotaryEmbedding as HFRope,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    apply_rotary_pos_emb,
)
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op

from mlc_llm.model.gemma4.gemma4_config import Gemma4TextConfig
from mlc_llm.model.gemma4.gemma4_unified import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedTextConfig,
    UnifiedAttention,
)


@unittest.skipUnless(os.environ.get("MLC_GEMMA4_TEXT_ONLY") == "1", "Requires text-only export")
class UnifiedTest(unittest.TestCase):
    def test_dflash_feature_export_contract(self):
        values = dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=1,
            head_dim=256,
            global_head_dim=512,
            attention_k_eq_v=True,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            vocab_size=256,
            use_double_wide_mlp=False,
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )
        config = Gemma4UnifiedConfig.from_dict(
            {"text_config": values, "dflash_target_layer_ids": [0, 1]}
        )
        model = Gemma4UnifiedForConditionalGeneration(config)
        mod, _, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        result = mod["batch_verify_with_hidden_states"].ret_ty
        print("DFlash feature export:", result, flush=True)
        self.assertEqual(len(result.fields), 3)
        self.assertEqual(int(result.fields[1].shape.values[-1]), 64)
        for invalid in ([1, 1], [-1], [2], [1, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config.kwargs["dflash_target_layer_ids"] = invalid
                Gemma4UnifiedForConditionalGeneration(config)

    def test_e_series_config_unchanged(self):
        config = Gemma4TextConfig()
        self.assertEqual(config.first_kv_shared_layer, 15)
        self.assertEqual(config.hidden_size_per_layer_input, 256)

    def test_local_and_global_qkv_match_transformers(self):
        torch.set_num_threads(2)
        torch.manual_seed(2718)
        values = dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=1,
            head_dim=256,
            global_head_dim=512,
            attention_k_eq_v=True,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=0,
            use_double_wide_mlp=False,
            use_bidirectional_attention="vision",
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        )
        config, hf_config = Gemma4UnifiedTextConfig.from_dict(values), HFConfig(**values)
        inputs = torch.randn(1, 3, 32) * 0.2
        positions = torch.tensor([0, 17, 1025], dtype=torch.int32)

        for index in (0, 1):
            with self.subTest(layer=index):
                reference = HFAttention(hf_config, index).eval()
                rope = HFRope(hf_config)

                class CaptureCache:
                    def attention_with_fused_qkv(self, layer, qkv, num_heads, sm_scale):
                        self.qkv = qkv
                        return op.split(qkv, [num_heads], axis=2)[0]

                class Module(nn.Module):
                    def __init__(self):
                        self.attn = UnifiedAttention(config, index)

                    def forward(self, hidden_states, positions):
                        cache = CaptureCache()
                        self.attn(hidden_states, cache, positions, None)
                        return cache.qkv

                mod, parameters, _ = Module().export_tvm(
                    spec={
                        "forward": {
                            "hidden_states": nn.spec.Tensor([1, 3, 32], "float32"),
                            "positions": nn.spec.Tensor([3], "int32"),
                        }
                    },
                    allow_extern=True,
                )
                vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
                state = reference.state_dict()
                weights = [
                    tvm_tirx.runtime.tensor(state[name.removeprefix("attn.")].numpy())
                    for name, _ in parameters
                ]
                actual = vm["forward"](
                    tvm_tirx.runtime.tensor(inputs.numpy()),
                    tvm_tirx.runtime.tensor(positions.numpy()),
                    *weights,
                ).numpy()
                with torch.no_grad():
                    dim = reference.head_dim
                    query = reference.q_norm(reference.q_proj(inputs).reshape(1, 3, 16, dim))
                    raw_key = reference.k_proj(inputs).reshape(1, 3, -1, dim)
                    raw_value = (
                        reference.v_proj(inputs).reshape(1, 3, -1, dim)
                        if reference.v_proj is not None
                        else raw_key
                    )
                    key, value = reference.k_norm(raw_key), reference.v_norm(raw_value)
                    cos, sin = rope(inputs, positions[None, :], config.layer_types[index])
                    query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2)
                    key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2)
                    if index == 1:
                        key, value = key.expand(-1, -1, 8, -1), value.expand(-1, -1, 8, -1)
                    padded = [
                        torch.nn.functional.pad(v, (0, 512 - dim)) for v in (query, key, value)
                    ]
                    expected = torch.cat(padded, dim=2).numpy()
                np.testing.assert_allclose(actual[:, :2], expected[:, :2], atol=1e-5, rtol=2e-5)
                # FP32 inverse-frequency rounding is amplified by position 1025;
                # LLVM's power/division and HF's inverse/multiply differ by <2e-4.
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=5e-5)
                print("HF QKV comparison passed:", config.layer_types[index], flush=True)


if __name__ == "__main__":
    unittest.main()
