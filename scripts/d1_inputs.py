"""Checkpoint-defined text packing and calibrated bool/choice readout.

Loads code only from an explicitly configured, trusted local checkpoint.
The reference scripts and service share formatting, but use independent backbones.
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class DecisionInputs:
    def __init__(self, source, context=256):
        from transformers import PreTrainedTokenizerFast

        self.source, self.context = Path(source).resolve(), context
        sys.path.insert(0, str(self.source.parent))

        def module(name):
            return importlib.import_module(self.source.name + "." + name)

        self.prompt, self.attention = module("prompt"), module("attention")
        runner = module("runner")
        self.reader = runner.SlotReader()
        r = self.reader
        r.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.source, local_files_only=True)
        r.bos = r.tokenizer.bos_token or ""
        r.lead = "<think></think>"
        r.state_style, r.system = self.prompt.DEFAULT_STATE_STYLE, self.prompt.DEFAULT_SYSTEM
        r.option_style, r.processor = "desc", None
        r.calibration = module("calibration").Calibration.load(self.source / "calibration.json")
        r._serve(SimpleNamespace(**json.loads((self.source / "config.json").read_text())))

    def pack(self, state, questions):
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 4:
            raise ValueError("Expected 1..4 named questions")
        # Pass JSON state through the checkpoint's native formatter, preserving
        # object order. Never truncate fields (headers can be decisive evidence).
        if (
            len(json.dumps(state, allow_nan=False)) > 65536
            or len(json.dumps(questions, allow_nan=False)) > 16384
        ):
            raise ValueError("Request text exceeds bounded input limit")
        qs = []
        for name, q in questions.items():
            if not isinstance(name, str) or not name or not isinstance(q, dict):
                raise ValueError("Questions must be named objects")
            if q.get("type") not in ("noul", "choice") or q.get("multiple", False):
                raise ValueError("Supported types: noul and single choice")
            if not isinstance(q.get("instructions"), str):
                raise ValueError("Question instructions must be text")
            if set(q) - {"type", "instructions", "criteria", "multiple"}:
                raise ValueError("Unsupported question fields")
            if "multiple" in q and q["multiple"] is not False:
                raise ValueError("Only single choice is supported")
            criteria = q.get("criteria")
            if q["type"] == "noul" and criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError("noul criteria must contain true and false descriptions")
                if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                    raise ValueError("noul criteria descriptions must be nonempty text")
            if q["type"] == "choice":
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
                    raise ValueError("choice requires 2..26 named options")
                if not all(
                    isinstance(k, str) and k and (v is None or isinstance(v, str))
                    for k, v in criteria.items()
                ):
                    raise ValueError("choice options must have text names and descriptions")
            # Validate the raw mapping: check_question trusts parsed dataclasses.
            self.prompt.check_question(q)
            parsed = self.prompt.as_question(q)
            qs.append(parsed)
        seq = self.reader._sequences([(state, qs, [])])[0]
        _, _, ids, state_len, groups, ends, _ = seq
        if len(ids) > self.context:
            raise ValueError(
                f"Packed request has {len(ids)} tokens; capacity {self.context}; no truncation"
            )
        tokens = np.zeros((1, self.context), dtype="int32")
        mask = np.zeros_like(tokens)
        qids = np.full_like(tokens, -1)
        tokens[0, : len(ids)], mask[0, : len(ids)], qids[0, : len(ids)] = ids, 1, groups
        return dict(
            tokens=tokens,
            mask=mask,
            groups=qids,
            length=len(ids),
            state_len=state_len,
            slots=[e[0] for e in ends],
            questions=qs,
            names=list(questions),
        )

    def read(self, packed, hidden, embedding_rows):
        answers = {}
        for name, q, slot in zip(packed["names"], packed["questions"], packed["slots"]):
            groups = self.prompt.readout_ids(self.reader.tokenizer, q)
            ids = sorted({t for group in groups for t in group})
            rows = embedding_rows(ids)
            logits = rows @ hidden[0, slot].astype("float32")
            # A common vocabulary log-softmax normalizer cancels in option softmax.
            probs = self.prompt.readout(
                self.reader.tokenizer, q, dict(zip(ids, logits)), self.reader.calibration
            )
            if isinstance(q, self.prompt.Noul):
                answers[name] = dict(type="noul", noul=probs[0])
            else:
                names = list(q.criteria)
                answers[name] = dict(
                    type="choice",
                    choice=names[int(np.argmax(probs))],
                    confidence=float(max(probs)),
                    probabilities=dict(zip(names, probs)),
                )
        return dict(answers=answers, usage=dict(input_tokens=packed["length"], output_tokens=0))
