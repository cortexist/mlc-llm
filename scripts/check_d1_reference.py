"""Save original-checkpoint CPU FP32 hidden states and calibrated decisions."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from d1_inputs import DecisionInputs
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--context", type=int, default=256)
p.add_argument(
    "--fixtures", type=Path, help="JSON list of {state, questions}; synthetic inputs only"
)
a = p.parse_args()
torch.set_num_threads(2)
inputs = DecisionInputs(a.source, context=a.context)
c = json.loads((a.source / "config.json").read_text())["text_config"]
cfg = importlib.import_module(a.source.name + ".configuration_lfm3").Lfm3Config(**c)
cfg._attn_implementation = "d1"
cls = importlib.import_module(a.source.name + ".modeling_lfm3").Lfm3Model
with torch.device("meta"):
    model = cls(cfg)
with safe_open(a.source / "model.safetensors", framework="pt", device="cpu") as f:
    state = {
        k.removeprefix("model.language_model."): f.get_tensor(k).float()
        for k in f.keys()
        if k.startswith("model.language_model.")
    }
model.load_state_dict(state, strict=True, assign=True)
model.rotary_emb = importlib.import_module(a.source.name + ".modeling_lfm3").Lfm3RotaryEmbedding(
    cfg
)
model.eval()
fixtures = [
    (
        "The server cannot resolve DNS names. Its network cable is connected.",
        {"network": dict(type="noul", instructions="Is this a network issue?")},
    ),
    (
        "The build failed because the CUDA compiler was not found.",
        {
            "route": dict(
                type="choice",
                instructions="Choose the issue category.",
                criteria=dict(
                    network="Connectivity or DNS",
                    build="Compilation or build tools",
                    unknown="None of these",
                ),
            ),
            "enough": dict(
                type="noul", instructions="Does the text identify the missing compiler?"
            ),
        },
    ),
]
if a.fixtures:
    fixtures = [(f["state"], f["questions"]) for f in json.loads(a.fixtures.read_text())]
a.output.mkdir(parents=True, exist_ok=True)
results = []
for i, (text, qs) in enumerate(fixtures):
    packed = inputs.pack(text, qs)
    start = time.perf_counter()
    with (
        torch.no_grad(),
        inputs.attention.state_attention(
            model, torch.tensor([packed["state_len"]]), torch.tensor(packed["groups"])
        ),
    ):
        hidden = model(
            input_ids=torch.tensor(packed["tokens"]).long(),
            attention_mask=torch.tensor(packed["mask"]),
            use_cache=False,
        ).last_hidden_state.numpy()
    result = inputs.read(packed, hidden, lambda ids: state["embed_tokens.weight"][ids].numpy())
    # Independently verify selected-row scoring against the native full LM head.
    for name, q, slot in zip(packed["names"], packed["questions"], packed["slots"]):
        logits = torch.nn.functional.linear(
            torch.from_numpy(hidden[0, slot]), state["embed_tokens.weight"]
        )
        native = inputs.prompt.readout(
            inputs.reader.tokenizer, q, logits.log_softmax(-1), inputs.reader.calibration
        )
        answer = result["answers"][name]
        expected = (
            [answer["noul"], 1 - answer["noul"]]
            if answer["type"] == "noul"
            else list(answer["probabilities"].values())
        )
        np.testing.assert_allclose(native, expected, atol=1e-6)
    result.update(seconds=time.perf_counter() - start, state=text, questions=qs)
    results.append(result)
    np.savez(
        a.output / f"reference-{i}.npz",
        hidden=hidden,
        tokens=packed["tokens"],
        mask=packed["mask"],
        groups=packed["groups"],
    )
    print(json.dumps(result), flush=True)
(a.output / "reference.json").write_text(json.dumps(results, indent=2) + "\n")
