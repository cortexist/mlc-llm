"""Compare an entire MLC MTP assistant step with Transformers on CPU."""

import os
import unittest

import numpy as np
import torch
import tvm_tirx
from transformers import Gemma4UnifiedAssistantConfig as HFConfig
from transformers import Gemma4UnifiedAssistantForCausalLM as HFModel
from transformers import Gemma4UnifiedTextConfig as HFTextConfig
from tvm import relax

from mlc_llm.model.gemma4.gemma4_mtp import (
    Gemma4UnifiedAssistantConfig,
    Gemma4UnifiedAssistantForCausalLM,
    Gemma4UnifiedMTPTarget,
)
from mlc_llm.model.gemma4.gemma4_unified import Gemma4UnifiedConfig


def text_config(shared):
    return dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        attention_k_eq_v=True,
        num_kv_shared_layers=4 if shared else 0,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        vocab_size=64,
        use_double_wide_mlp=False,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        use_bidirectional_attention="vision",
        sliding_window=1024,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )


@unittest.skipUnless(os.environ.get("MLC_GEMMA4_TEXT_ONLY") == "1", "Requires text-only exports")
class MTPTest(unittest.TestCase):
    def test_assistant_two_recurrent_steps_match_hf(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        cfg = text_config(True)
        reference = HFModel(
            HFConfig(text_config=HFTextConfig(**cfg), backbone_hidden_size=48)
        ).eval()
        model = Gemma4UnifiedAssistantForCausalLM(
            Gemma4UnifiedAssistantConfig.from_dict(
                {
                    "text_config": cfg,
                    "backbone_hidden_size": 48,
                }
            )
        )
        mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        with tvm_tirx.transform.PassContext(disabled_pass=["RewriteDataflowReshape"]):
            vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
        state = reference.state_dict()
        weights = []
        for name, _ in params:
            if name == "lm_head.weight":
                value = state["model.embed_tokens.weight"]
            elif ".gate_up_proj." in name:
                value = torch.cat(
                    [
                        state[name.replace("gate_up_proj", "gate_proj")],
                        state[name.replace("gate_up_proj", "up_proj")],
                    ],
                    dim=0,
                )
            else:
                value = state[name]
            weights.append(tvm_tirx.runtime.tensor(value.numpy()))
        rng = np.random.default_rng(13)
        arrays = [
            rng.normal(0, 0.3, shape).astype("float32")
            for shape in (
                (1, 1, 48),
                (1, 1, 48),
                (1, 7, 8, 256),
                (1, 7, 8, 256),
                (1, 7, 1, 512),
                (1, 7, 1, 512),
            )
        ]
        position = np.array([17], "int32")
        shared = {
            "sliding_attention": tuple(
                torch.from_numpy(a).permute(0, 2, 1, 3) for a in arrays[2:4]
            ),
            "full_attention": tuple(torch.from_numpy(a).permute(0, 2, 1, 3) for a in arrays[4:6]),
        }
        expected_hidden = torch.from_numpy(arrays[1])
        for step in range(2):
            actual = vm["mtp_step"](
                *[
                    tvm_tirx.runtime.tensor(
                        a if i < 2 else np.pad(a, ((0, 0), (0, 1017), (0, 0), (0, 0)))
                    )
                    for i, a in enumerate(arrays)
                ],
                tvm_tirx.runtime.tensor(position),
                tvm_tirx.runtime.tensor(np.array([7], "int32")),
                weights,
            )
            with torch.no_grad():
                expected = reference(
                    inputs_embeds=torch.cat([torch.from_numpy(arrays[0]), expected_hidden], dim=-1),
                    position_ids=torch.from_numpy(position[None, :]).long(),
                    shared_kv_states=shared,
                    use_cache=False,
                )
            np.testing.assert_allclose(
                actual[0].numpy(), expected.logits.numpy(), atol=2e-5, rtol=2e-4
            )
            np.testing.assert_allclose(
                actual[1].numpy(), expected.last_hidden_state.numpy(), atol=2e-5, rtol=2e-4
            )
            print(
                "HF assistant step passed:",
                step,
                "logit max error",
                np.max(np.abs(actual[0].numpy() - expected.logits.numpy())),
                flush=True,
            )
            arrays[1] = actual[1].numpy()
            expected_hidden = expected.last_hidden_state
            arrays[0] = rng.normal(0, 0.3, (1, 1, 48)).astype("float32")

    def test_target_state_export_shapes(self):
        model = Gemma4UnifiedMTPTarget(
            Gemma4UnifiedConfig.from_dict(
                {"text_config": text_config(False), "context_window_size": 1024}
            )
        )
        mod, _, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        result = mod["mtp_verify"].ret_ty
        self.assertEqual(len(result.fields), 7)
        self.assertEqual(int(result.fields[2].shape.values[-2]), 8)
        self.assertEqual(int(result.fields[2].shape.values[-1]), 256)
        self.assertEqual(int(result.fields[4].shape.values[-2]), 1)
        self.assertEqual(int(result.fields[4].shape.values[-1]), 512)


if __name__ == "__main__":
    unittest.main()
