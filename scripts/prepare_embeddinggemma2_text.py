"""Extract the text weights for the FP32, 128-token Xavier embedding PoC."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
config = json.loads((args.source / "config.json").read_text())
text = config["text_config"]
out_config = dict(
    model_type="embedding_gemma2_poc",
    architectures=["EmbeddingGemma2"],
    text_config=text,
    context_window_size=128,
    prefill_chunk_size=128,
    max_batch_size=1,
    vocab_size=text["vocab_size"],
)
(args.output / "config.json").write_text(json.dumps(out_config, indent=2) + "\n")
with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as reader:
    weights = {
        name: reader.get_tensor(name).contiguous()
        for name in reader.keys()
        if name.startswith("language_model.")
    }
save_file(weights, args.output / "model.safetensors")
for name in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "chat_template.jinja"]:
    if (args.source / name).exists():
        shutil.copy2(args.source / name, args.output / name)
print(json.dumps(dict(tensors=len(weights), parameters=sum(w.numel() for w in weights.values()))))
