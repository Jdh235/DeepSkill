import unittest
from pathlib import Path

from pyterrier_retrieval.corpus import load_sra_bench_corpus, load_skill_usage_corpus, skill_text
from pyterrier_retrieval.units import (
    skill_usage_units_from_instructions,
    sra_bench_units,
)

SRA_BENCH_CORPUS = Path("data/sra_bench/corpus/corpus.json")
SRA_BENCH_THEOREMQA = Path("data/sra_bench/instances/theoremqa.json")
SKILL_USAGE_META = Path("data/skill_usage/corpus/skill_usage/skills_meta.jsonl")
SKILL_USAGE_TASKS = Path("data/skill_usage/tasks")
SKILL_USAGE_MAPPING = Path("data/skill_usage/task_skill_mapping.json")

_missing_sra_bench = not SRA_BENCH_CORPUS.exists()
_missing_skill_usage = not SKILL_USAGE_META.exists() or not SKILL_USAGE_TASKS.exists()


@unittest.skipIf(_missing_sra_bench, "data/sra_bench/ not downloaded — see README")
class SraBenchQueryExampleTests(unittest.TestCase):

    def test_query_is_the_built_prompt_not_bare_question(self):
        units = sra_bench_units(SRA_BENCH_THEOREMQA)
        unit = next(u for u in units if u["id"] == "theoremqa_00000")

        self.assertEqual(len(unit["queries"]), 1, "sra_bench units are always single-query")
        query = unit["queries"][0]

        self.assertTrue(query.startswith("Problem:How many ways are there to divide "
                                          "a set of 8 elements into 5 non-empty ordered subsets?"))
        self.assertIn("Solution:", query)
        self.assertIn("You are a science teacher", query)  # theoremqa's system prompt
        self.assertIn('Therefore, the answer is', query)

        self.assertEqual(unit["gold_skill_ids"], ["theoremqa_000"])

    def test_skill_text_is_name_description_content(self):
        corpus = load_sra_bench_corpus(SRA_BENCH_CORPUS)
        skill = next(s for s in corpus if s["skill_id"] == "theoremqa_000")
        text = skill_text(skill)

        self.assertTrue(text.startswith("Lah Numbers\n"))
        self.assertIn("Computing Lah numbers for counting ordered partitions", text)
        self.assertIn("# Lah Numbers", text)  # SKILL.md-style content heading


@unittest.skipIf(_missing_skill_usage, "data/skill_usage/ not downloaded — see README")
class SkillUsageQueryExampleTests(unittest.TestCase):

    def test_default_query_is_the_raw_instruction(self):
        units = skill_usage_units_from_instructions(SKILL_USAGE_TASKS, SKILL_USAGE_MAPPING)
        unit = next(u for u in units if u["id"] == "3d-scan-calc")

        self.assertEqual(len(unit["queries"]), 1, "instruction mode is always single-query")
        self.assertTrue(unit["queries"][0].startswith(
            "You need to calculate the mass of a 3D printed part."
        ))
        # Gold names in task_skill_mapping.json are bare; the corpus prefixes them.
        self.assertEqual(unit["gold_skill_ids"], ["benchflow-ai--mesh-analysis"])

    def test_skill_text_reads_skill_md_content(self):
        corpus = load_skill_usage_corpus(SKILL_USAGE_META)
        skill = next(s for s in corpus if s["skill_id"] == "benchflow-ai--mesh-analysis")
        text = skill_text(skill)

        self.assertTrue(text.startswith("mesh-analysis\n"))
        self.assertIn("Analyzes 3D mesh files (STL) to calculate geometric properties", text)
        self.assertIn("# Mesh Analysis", text)  # from the actual SKILL.md file on disk
        # Frontmatter (--- name: ... ---) must be stripped, not indexed verbatim.
        self.assertNotIn("---\nname:", text)


if __name__ == "__main__":
    unittest.main()
