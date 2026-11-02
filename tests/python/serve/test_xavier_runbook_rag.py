"""CPU checks for input budgets, evidence provenance and retrieval scoring."""

import sys
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from xavier_runbook_rag import (
    BM25,
    chunk_source,
    expand_context,
    fuse,
    gold_ids,
    rank,
    retrieval_metrics,
)


class RunbookTests(unittest.TestCase):
    def test_windows_preserve_text_lines_and_token_budget(self):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        text = "\n".join(" ".join(f"word{i * 20 + j}" for j in range(20)) for i in range(20))
        chunks = chunk_source("example.md", "Short title", text, tokenizer)
        self.assertGreater(len(chunks), 3)
        covered = set()
        for chunk in chunks:
            self.assertEqual(chunk["text"], text[chunk["start"] : chunk["end"]])
            self.assertEqual(chunk["tokens"], len(tokenizer.encode(chunk["input"]).ids) + 2)
            self.assertLessEqual(chunk["tokens"], 128)
            self.assertEqual(chunk["start_line"], text[: chunk["start"]].count("\n") + 1)
            self.assertEqual(chunk["end_line"], text[: chunk["end"] - 1].count("\n") + 1)
            covered.update(range(chunk["start"], chunk["end"]))
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        self.assertEqual(chunks[-1]["end"], len(text))

    def test_evidence_uses_source_spans_and_missing_annotations_fail(self):
        index = dict(
            sources=[dict(path="a.md", text="A signed\ncertificate is needed.")],
            passages=[
                dict(id="p1", source="a.md", start=0, end=5),
                dict(id="p2", source="a.md", start=5, end=30),
                dict(id="p3", source="b.md", start=0, end=30),
            ],
        )
        self.assertEqual(gold_ids(index, dict(id="q1", evidence=["signed certificate"])), {"p2"})
        self.assertEqual(gold_ids(index, dict(id="q2", evidence=[])), set())
        with self.assertRaisesRegex(ValueError, "Evidence absent"):
            gold_ids(index, dict(id="q3", evidence=["unsupported fact"]))

    def test_lexical_baseline_and_metric_denominators(self):
        docs = [
            dict(id="p1", title="GPU", text="vector embedding inference"),
            dict(id="p2", title="TLS", text="signed certificate issuer trust"),
            dict(id="p3", title="Queue", text="shared GPU requests"),
        ]
        bm25 = BM25(docs)
        ranking = rank(docs, bm25.scores("CERTIFICATE issuer"))
        self.assertEqual(ranking[0][0]["id"], "p2")
        self.assertEqual(bm25.scores(""), [0, 0, 0])
        self.assertEqual(bm25.scores("notinthecorpus"), [0, 0, 0])
        metric = retrieval_metrics(ranking, {"p2"})
        self.assertTrue(metric["hit_at_1"])
        self.assertEqual(metric["reciprocal_rank"], 1)
        metric = retrieval_metrics(ranking, {"absent"})
        self.assertFalse(metric["hit_at_3"])
        self.assertEqual(metric["reciprocal_rank"], 0)
        # A gold item sorted into a zero-score lexical tail is not a retrieval hit.
        metric = retrieval_metrics(rank(docs, bm25.scores("notinthecorpus")), {"p1"})
        self.assertIsNone(metric["first_relevant_rank"])
        self.assertEqual(metric["reciprocal_rank"], 0)

    def test_fusion_uses_rank_agreement_without_mixing_score_scales(self):
        docs = [dict(id=f"p{i}") for i in range(3)]
        semantic = [(docs[0], 0.9), (docs[1], 0.8), (docs[2], 0.7)]
        lexical = [(docs[1], 900), (docs[2], 100), (docs[0], 0)]
        combined = fuse(docs, semantic, lexical)
        self.assertEqual(combined[0][0]["id"], "p1")
        self.assertAlmostEqual(combined[0][1], 1 / 62 + 1 / 61)

    def test_context_expansion_keeps_exact_source_lines_and_original_match(self):
        index = dict(sources=[dict(path="a.md", text="first\nsecond\nthird\nfourth\nfifth")])
        top = [
            dict(
                source="a.md",
                start_line=3,
                end_line=3,
                text="thi",
                url="https://example.com/a.md#L3",
            )
        ]
        expanded = expand_context(top, index, 1)[0]
        self.assertEqual(expanded["text"], "second\nthird\nfourth")
        self.assertEqual((expanded["start_line"], expanded["end_line"]), (2, 4))
        self.assertTrue(expanded["url"].endswith("#L2"))
        self.assertEqual(top[0]["text"], "thi")
        expanded = expand_context(top, index, 12)[0]
        self.assertEqual((expanded["start_line"], expanded["end_line"]), (1, 5))


if __name__ == "__main__":
    unittest.main()
