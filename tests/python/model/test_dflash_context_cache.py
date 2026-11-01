"""Compare incremental context K/V caching with recomputation, including GQA/SWA."""

import numpy as np
import tvm_tirx
from tvm import relax
from tvm.relax.frontend import nn

from mlc_llm.model.dflash.dflash_model import DFlashConfig, DFlashDraftModel


def build(cached, window, arrays, target_embedding_scale=1.0):
    cfg = DFlashConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000,
        vocab_size=128,
        block_size=3,
        context_window_size=64,
        target_model_dtype="float32",
        target_embedding_scale=target_embedding_scale,
        cache_context_kv=cached,
        sliding_window=window,
        dflash_config={"target_layer_ids": [0, 2, 4], "mask_token_id": 4},
    )
    model = DFlashDraftModel(cfg)
    packed = {"param_mode": "packed", "effect_mode": "none"}
    specs = {
        "project_target_hidden": {
            "target_hidden": nn.spec.Tensor([1, "seq_len", 96], "float32"),
            "$": packed,
        },
        "draft_forward": {
            "noise_embedding": nn.spec.Tensor([1, 3, 32], "float32"),
            "projected_target_hidden": nn.spec.Tensor(
                [1, "ctx_len", model.context_width], "float32"
            ),
            "cos": nn.spec.Tensor([1, "total_len", 1, 8], "float32"),
            "sin": nn.spec.Tensor([1, "total_len", 1, 8], "float32"),
            "attention_mask": nn.spec.Tensor([1, 1, 3, "total_len"], "float32"),
            "$": packed,
        },
    }
    mod, params = model.export_tvm(spec=specs)
    # The current TVM reshape detector cannot prove divisibility of dynamic
    # context extents in the broadcast GQA transpose. It is an optional pass.
    with tvm_tirx.transform.PassContext(disabled_pass=["RewriteDataflowReshape"]):
        executable = relax.build(mod, target="llvm")
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())
    rng = np.random.default_rng(7)
    weights = []
    for name, param in params:
        if name not in arrays:
            values = np.ones(param.shape) if "norm" in name else rng.normal(0, 0.1, param.shape)
            arrays[name] = values.astype("float32")
        weights.append(tvm_tirx.runtime.tensor(arrays[name]))
    return vm, weights


def test_context_cache():
    tensor = tvm_tirx.runtime.tensor
    rng = np.random.default_rng(8)
    arrays = {}
    for window in [0, 5]:
        uncached, uw = build(False, window, arrays)
        cached, cw = build(True, window, arrays)
        features = rng.normal(0, 1, (1, 11, 96)).astype("float32")
        noise = tensor(rng.normal(0, 1, (1, 3, 32)).astype("float32"))
        # A speculative verification projects more positions than it commits.
        # Preserve only a prefix before appending newly committed positions.
        for length in [2, 7, 11]:
            total = length + 3
            angles = np.arange(total)[:, None] / (10000 ** (np.arange(4) / 4))
            angles = np.concatenate([angles, angles], axis=-1)[None, :, None, :]
            cos, sin = (
                tensor(np.cos(angles).astype("float32")),
                tensor(np.sin(angles).astype("float32")),
            )
            mask = tensor(np.zeros((1, 1, 3, total), "float32"))
            all_cached = cached["project_target_hidden"](tensor(features), cw).numpy()
            split = length // 2
            tail = cached["project_target_hidden"](tensor(features[:, split:length]), cw).numpy()
            incremental = np.concatenate([all_cached[:, :split], tail], axis=1)
            np.testing.assert_allclose(incremental, all_cached[:, :length], rtol=2e-5, atol=2e-5)
            projected = uncached["project_target_hidden"](tensor(features[:, :length]), uw)
            expected = uncached["draft_forward"](noise, projected, cos, sin, mask, uw).numpy()
            actual = cached["draft_forward"](noise, tensor(incremental), cos, sin, mask, cw).numpy()
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            assert np.isfinite(actual).all()
            if window and length >= window:
                # Serving can omit keys outside every draft query's window,
                # but must preserve absolute RoPE positions in the suffix view.
                offset = length - (window - 1)
                short_cos = tensor(np.cos(angles[:, offset:]).astype("float32"))
                short_sin = tensor(np.sin(angles[:, offset:]).astype("float32"))
                short_mask = tensor(np.zeros((1, 1, 3, total - offset), "float32"))
                short = cached["draft_forward"](
                    noise,
                    tensor(incremental[:, offset:]),
                    short_cos,
                    short_sin,
                    short_mask,
                    cw,
                ).numpy()
                np.testing.assert_allclose(short, expected, rtol=2e-4, atol=2e-4)
            print(
                f"window={window} context={length} "
                f"max_abs_error={np.max(np.abs(actual - expected))}",
                flush=True,
            )


def test_target_embedding_scale():
    """A scaled target embedder must reproduce a raw-embedding draft run."""
    tensor = tvm_tirx.runtime.tensor
    arrays = {}
    raw, rw = build(True, 0, arrays)
    scale = 32**0.5
    scaled, sw = build(True, 0, arrays, target_embedding_scale=scale)
    rng = np.random.default_rng(19)
    features = tensor(rng.normal(0, 0.1, (1, 2, 96)).astype("float32"))
    noise = rng.normal(0, 1, (1, 3, 32)).astype("float32")
    cos = tensor(np.ones((1, 5, 1, 8), "float32"))
    sin = tensor(np.zeros((1, 5, 1, 8), "float32"))
    mask = tensor(np.zeros((1, 1, 3, 5), "float32"))
    raw_context = raw["project_target_hidden"](features, rw)
    scaled_context = scaled["project_target_hidden"](features, sw)
    expected = raw["draft_forward"](tensor(noise), raw_context, cos, sin, mask, rw).numpy()
    actual = scaled["draft_forward"](
        tensor(noise * scale), scaled_context, cos, sin, mask, sw
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    print(
        "Scaled target embeddings match raw DFlash input:",
        np.max(np.abs(actual - expected)),
        flush=True,
    )


if __name__ == "__main__":
    test_context_cache()
    test_target_embedding_scale()
