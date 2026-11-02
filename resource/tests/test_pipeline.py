"""Tests for the parts of the pipeline that don't need a JVM/pyterrier.

Run with: PYTHONPATH=src python3 -m unittest discover tests -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from pyterrier_retrieval.corpus import load_sra_bench_corpus, skill_text
from pyterrier_retrieval.prompts import build_prompt, list_datasets
from pyterrier_retrieval.retrievers.sparse_index import tokenize
from pyterrier_retrieval.schema import RetrievalRecord, RetrievalResults, as_lookup, load
from pyterrier_retrieval.units import interleave


class CorpusTests(unittest.TestCase):
    def test_load_corpus(self):
        skills = [{"skill_id": "a_1", "name": "N", "description": "D", "content": "C"}]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(skills, f)
            path = f.name
        self.assertEqual(load_sra_bench_corpus(path), skills)

    def test_skill_text_skips_empty_fields(self):
        self.assertEqual(
            skill_text({"name": "N", "description": "", "content": "C"}),
            "N\nC",
        )

    def test_skill_text_all_present(self):
        self.assertEqual(
            skill_text({"name": "N", "description": "D", "content": "C"}),
            "N\nD\nC",
        )


class TokenizeTests(unittest.TestCase):
    def test_lowercases_strips_punctuation_splits(self):
        self.assertEqual(
            tokenize("How many ways, exactly?"),
            ["how", "many", "ways", "exactly"],
        )

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])


class PromptTests(unittest.TestCase):
    def test_all_datasets_covered(self):
        self.assertEqual(
            sorted(list_datasets()),
            ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"],
        )

    def test_theoremqa_prompt(self):
        system, user = build_prompt({"dataset": "theoremqa", "question": "2+2?"})
        self.assertIn("2+2?", user)
        self.assertTrue(system)

    def test_logicbench_prompt_is_bare_question(self):
        system, user = build_prompt({"dataset": "logicbench", "question": "Q"})
        self.assertEqual((system, user), ("", "Q"))


class SchemaTests(unittest.TestCase):
    def test_dump_and_load_round_trip(self):
        records = [
            RetrievalRecord(
                instance_id="i1",
                gold_skill_ids=["s1"],
                retrieved=[{"skill_id": "s1", "score": 1.0}, {"skill_id": "s2", "score": 0.5}],
            )
        ]
        results = RetrievalResults(
            retriever="pt_bm25", top_k=10, corpus_size=2, records=records,
            metrics={"Recall@10": 1.0}, dataset="theoremqa",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            results.dump(path)
            data = load(path)

        self.assertEqual(data["metadata"]["retriever"], "pt_bm25")
        self.assertEqual(data["metadata"]["dataset"], "theoremqa")
        self.assertEqual(data["metrics"], {"Recall@10": 1.0})
        self.assertEqual(data["results"][0]["instance_id"], "i1")
        self.assertEqual(as_lookup(data)["i1"][0]["skill_id"], "s1")


class MergeTests(unittest.TestCase):
    """interleave() merges a unit's per-query ranked lists into one (round-robin by rank, dedup)."""

    LISTS = [
        ["A", "X0", "Q", "X1"],
        ["B", "X2", "X3", "Q"],
        ["C", "X4", "X5", "Q"],
        ["D", "X6", "Q", "X7"],
    ]

    def test_interleave_prefers_each_lists_own_top_pick(self):
        self.assertEqual(interleave(self.LISTS, top_k=3), ["A", "B", "C"])

    def test_dedup_and_truncate_to_top_k(self):
        lists = [["A", "B", "C"], ["B", "A", "D"]]
        self.assertEqual(len(interleave(lists, top_k=2)), 2)

    def test_single_query_unit_is_a_no_op(self):
        self.assertEqual(interleave([["X", "Y", "Z"]], top_k=3), ["X", "Y", "Z"])


if __name__ == "__main__":
    unittest.main()
