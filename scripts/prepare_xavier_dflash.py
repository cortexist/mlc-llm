"""Prepare isolated MLC packages from brain1's existing target and F16 draft."""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

root = Path(__file__).resolve().parents[1]
home = Path.home()
sys.path.insert(0, str(home / "Workspaces/llama.cpp/gguf-py"))
from gguf import GGUFReader

original = home / "Workspaces/mlc-jp5/dist/nemotron-3.5-Bolt-E4B-q4f16_1-MLC"
target = root / "dist/target"
draft_hf = root / "dist/draft-hf"
draft = root / "dist/draft"
for directory in [target, draft_hf, draft, root / "dist/libs"]:
    directory.mkdir(parents=True, exist_ok=True)

reader = GGUFReader(
    str(home / "Models/nemotron-3.5-Bolt-E4B-4GB/Nemotron-3.5-Bolt-E4B-DFlash-F16.gguf")
)
meta = lambda key: reader.fields[key].contents()
assert meta("general.architecture") == "dflash"
layers = [int(i) - 1 for i in meta("dflash.target_layers")]
assert layers == [5, 11, 17, 23, 35]
config = {
    "model_type": "dflash",
    "architectures": ["DFlashDraftModel"],
    "hidden_size": int(meta("dflash.embedding_length")),
    "intermediate_size": int(meta("dflash.feed_forward_length")),
    "num_hidden_layers": int(meta("dflash.block_count")),
    "num_attention_heads": int(meta("dflash.attention.head_count")),
    "num_key_value_heads": int(meta("dflash.attention.head_count_kv")),
    "head_dim": int(meta("dflash.attention.key_length")),
    "rms_norm_eps": float(meta("dflash.attention.layer_norm_rms_epsilon")),
    "rope_theta": float(meta("dflash.rope.freq_base")),
    "hidden_act": "silu",
    "vocab_size": 262144,
    "block_size": int(meta("dflash.block_size")),
    "num_target_layers": 42,
    "context_window_size": 8192,
    "prefill_chunk_size": 512,
    "max_batch_size": 1,
    "sliding_window": int(meta("dflash.attention.sliding_window")),
    "dflash_config": {"mask_token_id": 4, "target_layer_ids": layers},
    "target_model_dtype": "float16",
}
(draft_hf / "config.json").write_text(json.dumps(config, indent=2) + "\n")

mapping = {
    "fc.weight": "fc.weight",
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
}
parts = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
for i in range(config["num_hidden_layers"]):
    mapping.update(
        {f"blk.{i}.{key}.weight": f"layers.{i}.{value}.weight" for key, value in parts.items()}
    )
assert set(mapping) == {t.name for t in reader.tensors}
weights = {}
for tensor in reader.tensors:
    assert int(tensor.tensor_type) in (0, 1), (tensor.name, tensor.tensor_type)
    weights[mapping[tensor.name]] = np.asarray(tensor.data).reshape(tuple(reversed(tensor.shape)))
save_file(weights, str(draft_hf / "model.safetensors"))
print("Exported", len(weights), "F16/F32 tensors without requantization", flush=True)

base_config = json.loads((original / "mlc-chat-config.json").read_text())
target_config = json.loads(json.dumps(base_config))
target_config["model_config"]["dflash_target_layer_ids"] = layers
target_config["model_config"]["max_batch_size"] = 1
(target / "mlc-chat-config.json").write_text(json.dumps(target_config, indent=2) + "\n")
for path in original.iterdir():
    if path.name != "mlc-chat-config.json" and not (target / path.name).exists():
        (target / path.name).symlink_to(path)
draft_config = dict(base_config)
draft_config.update(
    model_type="dflash",
    quantization="q0f16",
    model_config=config,
    context_window_size=8192,
    prefill_chunk_size=512,
    sliding_window_size=-1,
)
(draft / "mlc-chat-config.json").write_text(json.dumps(draft_config, indent=2) + "\n")
for name in base_config["tokenizer_files"]:
    path = original / name
    if path.exists() and not (draft / name).exists():
        (draft / name).symlink_to(path)
print("Prepared", target, "and", draft, flush=True)
