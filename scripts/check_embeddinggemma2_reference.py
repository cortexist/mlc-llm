"""Save CPU FP32 reference vectors from the original checkpoint (no GPU work)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    EmbeddingGemma2TextConfig,
    EmbeddingGemma2TextModel,
)

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
torch.set_num_threads(2)
cfg = json.loads((args.source / "config.json").read_text())["text_config"]
model = EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(**cfg)).eval()
model.set_attn_implementation("eager")
with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as reader:
    state = {
        k.removeprefix("language_model."): reader.get_tensor(k).float()
        for k in reader.keys()
        if k.startswith("language_model.")
    }
model.load_state_dict(state, strict=True)
del state
tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
texts = [
    "task: search result | query: how do I reset my password?",
    "title: none | text: Reset your password using the account recovery link.",
    "title: none | text: Saturn is a planet with bright rings.",
    "task: classification | query: The delivery arrived two days late.",
]
encoded = tokenizer(
    texts, padding="max_length", max_length=128, return_tensors="pt", return_token_type_ids=False
)
vectors = []
with torch.no_grad():
    for i in range(len(texts)):
        output = model(
            input_ids=encoded.input_ids[i : i + 1], attention_mask=encoded.attention_mask[i : i + 1]
        ).last_hidden_state
        mask = encoded.attention_mask[i : i + 1, :, None]
        mean = (output * mask).sum(dim=1) / mask.sum(dim=1)
        vectors.append(torch.nn.functional.normalize(mean, dim=-1).numpy()[0])
np.savez(
    args.output,
    input_ids=encoded.input_ids.numpy().astype("int32"),
    attention_mask=encoded.attention_mask.numpy().astype("int32"),
    embeddings=np.array(vectors),
    texts=np.array(texts),
)
print(
    json.dumps(
        dict(
            texts=len(texts),
            shape=np.array(vectors).shape,
            query_relevant=float(np.dot(vectors[0], vectors[1])),
            query_unrelated=float(np.dot(vectors[0], vectors[2])),
        )
    ),
    flush=True,
)
