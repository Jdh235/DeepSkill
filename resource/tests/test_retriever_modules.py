import argparse
import importlib
import sys
import types
import unittest
from pathlib import Path

try:
    import pyterrier  # noqa: F401
except ImportError:
    sys.modules.setdefault("pyterrier", types.SimpleNamespace())

try:
    import ir_measures  # noqa: F401
except ImportError:
    _stub = types.SimpleNamespace(calc_aggregate=lambda *a, **k: {})
    _stub.R = object()
    _stub.nDCG = object()
    sys.modules.setdefault("ir_measures", _stub)

EXPECTED_SPARSE_WMODELS = {
    "bm25": "BM25",
    "hiemstra_lm": "Hiemstra_LM",
}


def _load(module_name: str):
    return importlib.import_module(f"pyterrier_retrieval.retrievers.{module_name}")


class SparseRetrieverModuleTests(unittest.TestCase):
    def test_wmodel_matches_expected_per_file(self):
        for name, expected_wmodel in EXPECTED_SPARSE_WMODELS.items():
            mod = _load(f"sparse.{name}")
            self.assertEqual(mod.WMODEL, expected_wmodel, f"sparse/{name}.py has the wrong WMODEL")

    def test_retriever_name_derived_correctly(self):
        for name, expected_wmodel in EXPECTED_SPARSE_WMODELS.items():
            mod = _load(f"sparse.{name}")
            self.assertEqual(mod.RETRIEVER_NAME, f"pt_{expected_wmodel.lower()}")

    def test_all_retriever_names_are_unique(self):
        names = [_load(f"sparse.{n}").RETRIEVER_NAME for n in EXPECTED_SPARSE_WMODELS]
        self.assertEqual(len(names), len(set(names)), f"duplicate RETRIEVER_NAME across files: {names}")

    def test_sra_bench_defaults_resolve_for_every_retriever(self):
        for name in EXPECTED_SPARSE_WMODELS:
            mod = _load(f"sparse.{name}")
            args = argparse.Namespace(retriever_name=None, 
                dataset="sra_bench",
                instances=Path("data/sra_bench/instances/theoremqa.json"),
                corpus=None, index_path=None, output=None,
            )
            mod.apply_defaults(args)
            self.assertEqual(args.corpus, Path("data/sra_bench/corpus/corpus.json"))
            self.assertEqual(
                args.output, Path(f"results/sra_bench/theoremqa-{mod.RETRIEVER_NAME}.json"),
                f"sparse/{name}.py produced the wrong default output path",
            )

    def test_skill_usage_defaults_resolve_for_every_retriever(self):
        for name in EXPECTED_SPARSE_WMODELS:
            mod = _load(f"sparse.{name}")

            args = argparse.Namespace(retriever_name=None, 
                dataset="skill_usage", instances=None,
                corpus=None, tasks_dir=None, mapping=None,
                index_path=None, output=None,
            )
            mod.apply_defaults(args)
            self.assertEqual(args.output, Path(f"results/skill_usage/{mod.RETRIEVER_NAME}.json"))
            self.assertEqual(args.tasks_dir, Path("data/skill_usage/tasks"))
            self.assertEqual(args.mapping, Path("data/skill_usage/task_skill_mapping.json"))

    def test_index_path_defaults_to_a_fresh_temp_dir(self):
        # Regression test for the "always rebuild, never share a fixed
        # default path" decision — a fixed default would risk two
        # concurrent jobs corrupting each other's index mid-rebuild.
        mod = _load("sparse.bm25")
        args1 = argparse.Namespace(retriever_name=None, dataset="sra_bench", instances=Path("data/sra_bench/instances/theoremqa.json"),
                                    corpus=None, index_path=None, output=None)
        args2 = argparse.Namespace(retriever_name=None, dataset="sra_bench", instances=Path("data/sra_bench/instances/theoremqa.json"),
                                    corpus=None, index_path=None, output=None)
        mod.apply_defaults(args1)
        mod.apply_defaults(args2)
        self.assertNotEqual(args1.index_path, args2.index_path)


class DenseRetrieverModuleTests(unittest.TestCase):
    def test_bge_defaults(self):
        mod = _load("dense.bge")
        self.assertEqual(mod.RETRIEVER_NAME, "dense_bge")
        args = argparse.Namespace(retriever_name=None, 
            dataset="sra_bench", instances=Path("data/sra_bench/instances/champ.json"),
            corpus=None, index_path=None, output=None,
        )
        mod.apply_defaults(args)
        self.assertEqual(args.output, Path("results/sra_bench/champ-dense_bge.json"))


if __name__ == "__main__":
    unittest.main()
