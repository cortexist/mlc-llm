"""Full small encoder comparison, including bidirectionality, PLE and padding."""

import unittest

import numpy as np
import torch
import tvm_tirx
from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel
from tvm import relax

from mlc_llm.model.gemma4.embeddinggemma2 import EmbeddingConfig, EmbeddingGemma2


class EncoderTest(unittest.TestCase):
    def test_fp32_encoder_matches_transformers(self):
        torch.set_num_threads(2)
        torch.manual_seed(43)
        config = dict(
            vocab_size=64,
            hidden_size=16,
            hidden_size_per_layer_input=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            embedding_dim=24,
            intermediate_size=32,
            rms_norm_eps=1e-6,
            sliding_window=2,
            layer_types=["sliding_attention", "full_attention"],
            per_layer_config={"01": {"head_dim": 32, "num_key_value_heads": 1}},
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
            },
        )
        hf = EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(**config)).eval()
        hf.set_attn_implementation("eager")
        model = EmbeddingGemma2(
            EmbeddingConfig(text_config=config, context_window_size=8, prefill_chunk_size=8)
        )
        mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        with tvm_tirx.transform.PassContext(disabled_pass=["RewriteDataflowReshape"]):
            vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
        state = hf.state_dict()
        weights = [
            tvm_tirx.runtime.tensor(state[name.removeprefix("language_model.")].numpy())
            for name, _ in params
        ]
        for length in [5, 8]:
            ids = np.array([[2, 10, 3, 22, 1, 7, 8, 1]], dtype="int32")
            ids[:, length:] = 0
            mask = np.array([[int(i < length) for i in range(8)]], dtype="int32")
            actual = vm["prefill"](
                tvm_tirx.runtime.tensor(ids), tvm_tirx.runtime.tensor(mask), weights
            ).numpy()
            with torch.no_grad():
                expected = hf(
                    input_ids=torch.tensor(ids).long(), attention_mask=torch.tensor(mask)
                ).last_hidden_state.numpy()
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(
                actual[:, :length], expected[:, :length], atol=2e-5, rtol=2e-4
            )
            print(
                "Encoder matched HF:",
                length,
                float(np.max(np.abs(actual[:, :length] - expected[:, :length]))),
                flush=True,
            )

    def test_fp16_is_rejected(self):
        # Reject before allocating any model or touching a GPU.
        model = EmbeddingGemma2.__new__(EmbeddingGemma2)
        with self.assertRaises(ValueError):
            model.to("float16")


if __name__ == "__main__":
    unittest.main()
