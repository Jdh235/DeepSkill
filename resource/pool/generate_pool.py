from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from pyterrier_retrieval.schema import RetrievalRecord, RetrievalResults

POOL = [
    "pt_bm25",
    "pt_hiemstra_lm",
    "pt_splade",
    "dense_bge",
    "dense_colbertv2",
    "dense_contriever",
    "dense_qwen3_emb_06b",
    "dense_qwen3_emb_8b",
    "dense_skillret_06b",
    "dense_skillret_8b",
    "dense_skillrouter_06b",
    "dense_bge_m3",
]
DEPTHS = [20, 10, 5]

REPO_ROOT = Path(__file__).parent.parent

DATASETS: dict[str, Callable[[str], Path]] = {
    "skill_usage": lambda ret: REPO_ROOT / f"results/skill_usage/{ret}.json",
    **{
        ds: (lambda ret, ds=ds: REPO_ROOT / f"results/sra_bench/{ds}-{ret}.json")
        for ds in ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]
    },
}

OUT_DIR = Path(__file__).parent


def find_missing(datasets: dict[str, Callable[[str], Path]]) -> dict[str, list[tuple[str, Path]]]:
    missing: dict[str, list[tuple[str, Path]]] = {}
    for dataset, path_for in datasets.items():
        gaps = [(ret, path_for(ret)) for ret in POOL if not path_for(ret).exists()]
        if gaps:
            missing[dataset] = gaps
    return missing


def load_ranked_by_iid(
    pool_members: list[str], path_for: Callable[[str], Path],
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], int]:
    """Load every pool member's raw results file for one dataset, once.

    Returns (gold_by_iid, ranked_by_iid, corpus_size):
      gold_by_iid[iid]              gold_skill_ids (every member agrees, take the first)
      ranked_by_iid[iid][retriever] that retriever's full ranked skill_id list
      corpus_size                   from the first member's metadata
    """
    per_member = {ret: json.loads(path_for(ret).read_text()) for ret in pool_members}
    corpus_size = next(iter(per_member.values()))["metadata"]["corpus_size"]

    gold_by_iid: dict[str, list[str]] = {}
    ranked_by_iid: dict[str, dict[str, list[str]]] = {}
    for ret, data in per_member.items():
        for entry in data["results"]:
            iid = entry["instance_id"]
            gold_by_iid.setdefault(iid, entry["gold_skill_ids"])
            ranked_by_iid.setdefault(iid, {})[ret] = [r["skill_id"] for r in entry["retrieved"]]

    return gold_by_iid, ranked_by_iid, corpus_size


def build_pool_at_depth(
    dataset: str,
    gold_by_iid: dict[str, list[str]],
    ranked_by_iid: dict[str, dict[str, list[str]]],
    corpus_size: int,
    pool_members: list[str],
    depth: int,
) -> RetrievalResults:
    """Union each pool member's own top-`depth` results per task, scored
    by consensus count -- how many members surfaced it -- at whatever
    depth is requested.
    """
    records = []
    pool_sizes = []
    covered, total_gold, fully_covered = 0, 0, 0
    for iid, by_ret in ranked_by_iid.items():
        counts = Counter(sid for top in by_ret.values() for sid in top[:depth])
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        retrieved = [{"skill_id": sid, "score": count} for sid, count in ranked]
        records.append(RetrievalRecord(
            instance_id=iid, gold_skill_ids=gold_by_iid[iid], retrieved=retrieved,
        ))

        pool_sizes.append(len(counts))
        gold = set(gold_by_iid[iid])
        hit = gold & counts.keys()
        covered += len(hit)
        total_gold += len(gold)
        fully_covered += int(hit == gold)

    metrics = {
        "avg_pool_size": sum(pool_sizes) / len(pool_sizes),
        "gold_coverage": covered / total_gold if total_gold else float("nan"),
        "tasks_fully_covered": fully_covered / len(records),
    }

    return RetrievalResults(
        retriever="pool",
        top_k=max(pool_sizes),
        corpus_size=corpus_size,
        records=records,
        metrics=metrics,
        dataset=dataset,
        extra={"pool_members": pool_members, "union_k": depth},
    )


def main() -> None:
    missing = find_missing(DATASETS)
    if missing:
        print("Missing retrieval results -- refusing to build a partial pool.\n")
        for dataset, gaps in missing.items():
            print(f"=== {dataset} ({len(gaps)} missing) ===")
            for ret, path in gaps:
                print(f"  {ret}: expected {path}")
        print("\nRun the corresponding retriever module(s) under "
              "src/pyterrier_retrieval/retrievers/ for the dataset(s) above "
              "(see README's \"Running a retriever\" section), then re-run this script.")
        sys.exit(1)

    for dataset, path_for in DATASETS.items():
        print(f"=== {dataset} ===")
        gold_by_iid, ranked_by_iid, corpus_size = load_ranked_by_iid(POOL, path_for)
        for depth in DEPTHS:
            result = build_pool_at_depth(dataset, gold_by_iid, ranked_by_iid, corpus_size, POOL, depth)
            out_dir = OUT_DIR / f"top{depth}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{dataset}-pool.json"
            result.dump(out)
            print(f"  top{depth}: {len(result.records)} tasks, avg_pool_size={result.metrics['avg_pool_size']:.1f}, "
                  f"gold_coverage={result.metrics['gold_coverage']:.4f}")
            print(f"    Saved: {out}")


if __name__ == "__main__":
    main()
