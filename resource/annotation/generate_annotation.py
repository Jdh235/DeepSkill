from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
POOL_DIR = HERE.parent / "pool"
TASKS_FILE = HERE / "tasks.txt"
DEFAULT_DEPTH = 5
SUBSET_SIZE = 75


def read_task_list(path: Path) -> list[str]:
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tasks.append(line)
    return tasks


def build_annotation_payload(dataset: str, pool_data: dict, n_annotators: int) -> dict:
    """Turn one dataset's filtered pool records into annotation-ready
    records: split into retrieved/gold_not_retrieved, each candidate
    tagged with is_gold and an empty per-annotator judgment slot.
    """
    annotator_keys = [f"annotator_{i}" for i in range(1, n_annotators + 1)]

    records = []
    n_candidates = 0
    n_gold_not_retrieved = 0
    for entry in pool_data["results"]:
        gold = set(entry["gold_skill_ids"])
        retrieved_ids = {r["skill_id"] for r in entry["retrieved"]}

        retrieved = [
            {
                "skill_id": r["skill_id"],
                "score": r["score"],
                "is_gold": r["skill_id"] in gold,
                "annotations": {k: None for k in annotator_keys},
            }
            for r in entry["retrieved"]
        ]
        retrieved.sort(key=lambda c: -c["score"])

        gold_not_retrieved = [
            {
                "skill_id": g,
                "is_gold": True,
                "annotations": {k: None for k in annotator_keys},
            }
            for g in sorted(gold - retrieved_ids)
        ]

        n_candidates += len(retrieved) + len(gold_not_retrieved)
        n_gold_not_retrieved += len(gold_not_retrieved)
        records.append({
            "instance_id": entry["instance_id"],
            "gold_skill_ids": entry["gold_skill_ids"],
            "retrieved": retrieved,
            "gold_not_retrieved": gold_not_retrieved,
        })

    return {
        "metadata": {
            "dataset": dataset,
            "retriever": "annotation",
            "corpus_size": pool_data["metadata"]["corpus_size"],
            "n_queries": len(records),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extra": {
                "pool_members": pool_data["metadata"]["extra"]["pool_members"],
                "union_k": pool_data["metadata"]["extra"]["union_k"],
                "n_annotators": n_annotators,
            },
        },
        "metrics": {
            "n_candidates": n_candidates,
            "n_gold_not_retrieved": n_gold_not_retrieved,
        },
        "results": records,
    }


def filter_pool_by_tasks(pool_dir: Path, wanted: list[str]) -> dict[str, dict]:
    """{dataset: pool payload filtered to just `wanted`'s instance_ids}.

    Matches by instance_id membership against every {dataset}-pool.json
    in pool_dir, not by guessing the dataset from the id's naming
    convention (skill_usage's ids aren't numeric). Exits with an error
    listing anything in `wanted` that wasn't found in any pool file,
    rather than silently building a partial subset.
    """
    remaining = set(wanted)
    filtered_by_dataset: dict[str, dict] = {}

    for pool_path in sorted(pool_dir.glob("*-pool.json")):
        if not remaining:
            break
        data = json.loads(pool_path.read_text())
        dataset = data["metadata"]["dataset"]
        by_iid = {r["instance_id"]: r for r in data["results"]}
        hits = [iid for iid in wanted if iid in remaining and iid in by_iid]
        if not hits:
            continue
        filtered_by_dataset[dataset] = {**data, "results": [by_iid[iid] for iid in hits]}
        remaining -= set(hits)

    if remaining:
        print(f"{len(remaining)} task id(s) from {TASKS_FILE} not found in any {pool_dir} pool file:")
        for iid in sorted(remaining):
            print(f"  {iid}")
        sys.exit(1)

    return filtered_by_dataset


def write_annotation_files(filtered_by_dataset: dict[str, dict], out_dir: Path, n_annotators: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, data in filtered_by_dataset.items():
        payload = build_annotation_payload(dataset, data, n_annotators)
        payload["metadata"]["extra"]["task_list_file"] = str(TASKS_FILE.relative_to(HERE.parent))

        out = out_dir / f"{dataset}-annotation.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        m = payload["metrics"]
        print(f"{dataset}: {len(payload['results'])} tasks, {m['n_candidates']} candidates "
              f"({m['n_gold_not_retrieved']} gold, not in pool)")
        print(f"  Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                         help=f"pool depth to draw candidates from (default: {DEFAULT_DEPTH})")
    parser.add_argument("--annotators", type=int, default=1,
                         help="number of annotator_N keys per candidate (default: 1)")
    args = parser.parse_args()

    if not TASKS_FILE.exists():
        print(f"No task list at {TASKS_FILE} -- create it first (one instance_id per line).")
        sys.exit(1)

    wanted = read_task_list(TASKS_FILE)
    if len(wanted) != SUBSET_SIZE:
        print(f"{TASKS_FILE} has {len(wanted)} task id(s), expected {SUBSET_SIZE}. "
              "Refusing to generate a mislabeled subset -- update SUBSET_SIZE in this "
              "script if the subset was deliberately resized.")
        sys.exit(1)

    filtered_by_dataset = filter_pool_by_tasks(POOL_DIR / f"top{args.depth}", wanted)
    out_dir = HERE / f"top{args.depth}" / "task_subset"
    write_annotation_files(filtered_by_dataset, out_dir, args.annotators)


if __name__ == "__main__":
    main()
