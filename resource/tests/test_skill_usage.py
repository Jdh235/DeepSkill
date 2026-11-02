import json
import unittest
from pathlib import Path

from pyterrier_retrieval.corpus import load_skill_usage_corpus
from pyterrier_retrieval.units import skill_usage_units_from_instructions

SKILL_USAGE_META = Path("data/skill_usage/corpus/skill_usage/skills_meta.jsonl")
SKILL_USAGE_TASKS = Path("data/skill_usage/tasks")
SKILL_USAGE_MAPPING = Path("data/skill_usage/task_skill_mapping.json")

_missing = not SKILL_USAGE_META.exists() or not SKILL_USAGE_TASKS.exists()


@unittest.skipIf(_missing, "data/skill_usage/ not downloaded — see README")
class CorpusConsistencyTests(unittest.TestCase):
    def test_corpus_has_no_duplicate_skill_ids(self):
        corpus = load_skill_usage_corpus(SKILL_USAGE_META)
        ids = [s["skill_id"] for s in corpus]
        self.assertEqual(len(ids), len(set(ids)), "duplicate skill_id in corpus")

    def test_corpus_size_matches_metadata_file(self):
        corpus = load_skill_usage_corpus(SKILL_USAGE_META)
        n_lines = sum(1 for line in SKILL_USAGE_META.read_text().splitlines() if line.strip())
        self.assertEqual(len(corpus), n_lines)

    def test_every_skill_has_a_skill_md_folder_on_disk(self):
        corpus = {s["skill_id"]: s for s in load_skill_usage_corpus(SKILL_USAGE_META)}
        gold_ids = [sid for sid in corpus if sid.startswith("benchflow-ai--")]
        self.assertTrue(gold_ids, "no benchflow-ai-- skills found at all")
        empty_content = [sid for sid in gold_ids if not corpus[sid]["content"]]
        self.assertEqual(empty_content, [], "gold skills with no SKILL.md content found")


@unittest.skipIf(_missing, "data/skill_usage/ not downloaded — see README")
class GoldSkillResolutionTests(unittest.TestCase):
    """The check that caught the "are gold skills actually in the corpus"
    question earlier in this project — kept as a permanent regression test."""

    def test_all_gold_skill_ids_resolve_in_corpus(self):
        corpus_ids = {s["skill_id"] for s in load_skill_usage_corpus(SKILL_USAGE_META)}
        mapping = json.loads(SKILL_USAGE_MAPPING.read_text())

        missing = [
            (task, f"benchflow-ai--{name}")
            for task, names in mapping.items()
            for name in names
            if f"benchflow-ai--{name}" not in corpus_ids
        ]
        self.assertEqual(missing, [], f"gold skills missing from corpus: {missing}")

    def test_every_task_has_at_least_one_gold_skill(self):
        mapping = json.loads(SKILL_USAGE_MAPPING.read_text())
        empty = [task for task, names in mapping.items() if not names]
        self.assertEqual(empty, [], f"tasks with no gold skills: {empty}")


@unittest.skipIf(_missing, "data/skill_usage/ not downloaded — see README")
class UnitLoaderTests(unittest.TestCase):
    """Skill-Usage units come from each task's own instruction.md."""

    def test_units_cover_all_87_tasks(self):
        units = skill_usage_units_from_instructions(SKILL_USAGE_TASKS, SKILL_USAGE_MAPPING)
        self.assertEqual(len({u["id"] for u in units}), 87)

    def test_every_unit_has_nonempty_queries_and_gold(self):
        for unit in skill_usage_units_from_instructions(SKILL_USAGE_TASKS, SKILL_USAGE_MAPPING):
            self.assertTrue(unit["queries"], f"{unit['id']} has no queries")
            self.assertTrue(all(q.strip() for q in unit["queries"]), f"{unit['id']} has a blank query")
            self.assertTrue(unit["gold_skill_ids"], f"{unit['id']} has no gold")


if __name__ == "__main__":
    unittest.main()
