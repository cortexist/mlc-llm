"""Short actual-checkpoint GPU/reference comparison. Run with serving stopped."""

import argparse
import json
from pathlib import Path

import numpy as np
from d1_service import ResidentDecision

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("references", type=Path)
p.add_argument("--repeats", type=int, default=3)
a = p.parse_args()
worker = ResidentDecision(a.root.resolve())
results = []
try:
    for i, expected in enumerate(json.loads((a.references / "reference.json").read_text())):
        packed = worker.inputs.pack(expected["state"], expected["questions"])
        reference = np.load(a.references / f"reference-{i}.npz")
        for key, refkey in [("tokens", "tokens"), ("mask", "mask"), ("groups", "groups")]:
            np.testing.assert_array_equal(packed[key], reference[refkey])
        args = [worker.tensor(packed[k], worker.device) for k in ("tokens", "mask", "groups")]
        actual = worker.model.call("prefill", *args).numpy()
        n = packed["length"]
        x, y = (
            actual[:, :n].astype("float64").ravel(),
            reference["hidden"][:, :n].astype("float64").ravel(),
        )
        cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        assert np.isfinite(actual).all() and cosine > 0.99999
        if worker.precision == "f32":
            np.testing.assert_allclose(
                actual[:, :n], reference["hidden"][:, :n], atol=0.003, rtol=0.003
            )
        probability_tolerance = 0.001 if worker.precision == "f32" else 0.005
        observed = worker.inputs.read(packed, actual, worker.embedding_rows)
        for name, answer in observed["answers"].items():
            target = expected["answers"][name]
            if answer["type"] == "noul":
                assert abs(answer["noul"] - target["noul"]) < probability_tolerance
            else:
                assert answer["choice"] == target["choice"]
                np.testing.assert_allclose(
                    list(answer["probabilities"].values()),
                    list(target["probabilities"].values()),
                    atol=probability_tolerance,
                )
        times = []
        for _ in range(a.repeats):
            _, metrics = worker.decide(expected["state"], expected["questions"])
            times.append(metrics)
        result = dict(
            tokens=n,
            precision=worker.precision,
            hidden_cosine=cosine,
            max_abs_hidden=float(np.max(np.abs(actual[:, :n] - reference["hidden"][:, :n]))),
            answers=observed["answers"],
            warm_timings=times,
        )
        print(json.dumps(result), flush=True)
        results.append(result)
    (a.references / f"gpu-{worker.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
finally:
    worker.pool.shutdown(wait=True)
