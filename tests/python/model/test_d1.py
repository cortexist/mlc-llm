"""Compare every layer type against the checkpoint's independent HF implementation.

Run with D1_CHECKPOINT pointing to the original local model/code directory.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import tvm_tirx as tvm
from tvm import relax

from mlc_llm.model.d1 import D1Config, D1Text


class D1Test(unittest.TestCase):
    def test_backbone(self):
        source = Path(os.environ["D1_CHECKPOINT"])
        sys.path.insert(0, str(source.parent))
        hf = importlib.import_module(source.name + ".modeling_lfm3")
        cfgmod = importlib.import_module(source.name + ".configuration_lfm3")
        torch.set_num_threads(2)
        torch.manual_seed(17)
        c = dict(
            hidden_size=16,
            intermediate_size=32,
            vocab_size=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            norm_eps=1e-5,
            conv_L_cache=7,
            conv_bias=False,
            layer_types=["conv", "sliding_attention", "full_attention"],
            sliding_window=3,
            use_sandwich_norm=True,
            rope_parameters={
                "sliding_attention": dict(rope_type="default", rope_theta=10000),
                "full_attention": dict(
                    rope_type="default", rope_theta=1000000, partial_rotary_factor=0.5
                ),
            },
        )
        cfg = cfgmod.Lfm3Config(**c)
        cfg._attn_implementation = "eager"
        ref = hf.Lfm3Model(cfg).eval()
        model = D1Text(D1Config(text_config=c, context_window_size=12, prefill_chunk_size=12))
        mod, params, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
        with tvm.transform.PassContext(disabled_pass=["RewriteDataflowReshape"]):
            vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
        state = ref.state_dict()
        weights = [
            tvm.runtime.tensor(state[n.removeprefix("language_model.")].numpy()) for n, _ in params
        ]
        for length in (8, 12):
            ids = np.arange(12, dtype="int32")[None] + 1
            mask = (np.arange(12)[None] < length).astype("int32")
            groups = np.array([[-1] * 3 + [0] * 4 + [1] * 5], dtype="int32")
            actual = vm["prefill"](
                *[tvm.runtime.tensor(a) for a in (ids, mask, groups)], weights
            ).numpy()
            with torch.no_grad():
                x = ref.embed_tokens(torch.tensor(ids).long())
                pos = torch.arange(12)[None]
                for layer, kind in zip(ref.layers, c["layer_types"]):
                    if kind == "conv":
                        x = layer(x, attention_mask=torch.tensor(mask))
                    else:
                        i, j = np.arange(12)[:, None], np.arange(12)[None, :]
                        g = groups[0]
                        allow = (
                            ((j <= i) & (j > i - 3))
                            if kind == "sliding_attention"
                            else ((g[:, None] < 0) | (g[None, :] < 0) | (g[:, None] == g[None, :]))
                        )
                        allow &= mask.astype(bool)
                        additive = torch.tensor(
                            np.where(allow, 0, np.finfo("float32").min).astype("float32")
                        )[None, None]
                        x = layer(
                            x,
                            position_embeddings=ref.rotary_emb(x, pos, kind),
                            attention_mask=additive,
                        )
                expected = ref.embedding_norm(x).numpy()
            np.testing.assert_allclose(
                actual[:, :length], expected[:, :length], atol=2e-5, rtol=2e-4
            )
            print(
                "d1 tiny backbone matches HF",
                length,
                float(np.max(np.abs(actual[:, :length] - expected[:, :length]))),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
