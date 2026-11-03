"""Extract LFM3 text weights; retain a manifest of the original local checkpoint."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--context", type=int, default=256, choices=[256, 512])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
c = json.loads((a.source / "config.json").read_text())
assert c["model_type"] == "lfm3_omni" and c["d1_attention"] == "set"
config = dict(
    model_type="d1_lfm3_text",
    text_config=c["text_config"],
    context_window_size=a.context,
    prefill_chunk_size=a.context,
    max_batch_size=1,
    vocab_size=c["text_config"]["vocab_size"],
)
(a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
with safe_open(a.source / "model.safetensors", framework="pt", device="cpu") as f:
    weights = {
        k.removeprefix("model."): f.get_tensor(k).contiguous()
        for k in f.keys()
        if k.startswith("model.language_model.")
    }
save_file(weights, a.output / "model.safetensors")
for name in ["tokenizer.json", "tokenizer_config.json", "calibration.json"]:
    shutil.copy2(a.source / name, a.output / name)
manifest = {}
for file in sorted(a.source.iterdir()):
    if file.is_file():
        with file.open("rb") as stream:
            manifest[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
(a.output / "source-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(dict(parameters=sum(w.numel() for w in weights.values()), tensors=len(weights))))
