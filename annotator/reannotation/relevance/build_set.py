"""Build the relevance re-annotation set: every task-skill pair whose round-1
ratings (reannotation/round1/task_subset_N/, one per annotator) differ by 2+.

Writes data/task_subset/<dataset>-annotation.json with ratings null. Refuses
to overwrite an existing set (it holds the re-annotations) unless --force.

Usage (from the repo root):
    python3 reannotation/relevance/build_set.py [--min-spread 2] [--force]
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND1_DIR = HERE.parent / "round1"
OUT = HERE / "data" / "task_subset"
ANNOTATOR = "annotator_1"  # same key app.py writes


def load_ratings(src: Path) -> dict[tuple[str, str, str], int | None]:
    """(dataset, instance_id, skill_id) -> this annotator's rating."""
    ratings = {}
    for path in sorted(src.glob("*-annotation.json")):
        dataset = path.name.removesuffix("-annotation.json")
        for result in json.loads(path.read_text())["results"]:
            for cand in result["retrieved"]:
                ratings[(dataset, result["instance_id"], cand["skill_id"])] = next(iter(cand.get("annotations", {}).values()), None)
    return ratings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-spread", type=int, default=2, help="Re-annotate pairs where highest - lowest rating >= this (default: 2)")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing set, wiping any saved re-annotations")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} already exists and holds the re-annotations. Pass --force to rebuild it (wipes them).")
    sources = sorted(ROUND1_DIR.glob("task_subset_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if len(sources) < 2:
        raise SystemExit(f"Need at least two annotators' folders in {ROUND1_DIR}.")

    all_ratings = [load_ratings(src) for src in sources]
    keys = set(all_ratings[0])
    for src, ratings in zip(sources[1:], all_ratings[1:]):
        if set(ratings) != keys:
            raise SystemExit(f"{src} doesn't cover the same task-skill pairs as {sources[0]}.")
    disputed = set()
    for key in keys:
        values = [r[key] for r in all_ratings if r[key] is not None]
        if len(values) >= 2 and max(values) - min(values) >= args.min_spread:
            disputed.add(key)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for path in sorted(sources[0].glob("*-annotation.json")):
        dataset = path.name.removesuffix("-annotation.json")
        data = json.loads(path.read_text())
        results = []
        for result in data["results"]:
            kept = [
                {**copy.deepcopy(cand), "annotations": {ANNOTATOR: None}}
                for cand in result["retrieved"]
                if (dataset, result["instance_id"], cand["skill_id"]) in disputed
            ]
            if kept:
                results.append({**{k: v for k, v in result.items() if k != "gold_not_retrieved"}, "retrieved": kept})
        if not results:
            continue
        n_pairs = sum(len(r["retrieved"]) for r in results)
        data["results"] = results
        data["metadata"]["n_queries"] = len(results)
        data["metadata"].setdefault("extra", {})["reannotation"] = {
            "sources": [s.name for s in sources],
            "min_spread": args.min_spread,
        }
        data["metrics"] = {"n_candidates": n_pairs}
        (OUT / path.name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(f"{dataset}: {n_pairs} pairs across {len(results)} tasks")
    print(f"\n{len(disputed)} pairs to re-annotate (spread >= {args.min_spread}), written to {OUT}")


if __name__ == "__main__":
    main()
