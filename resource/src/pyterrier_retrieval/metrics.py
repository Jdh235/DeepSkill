from __future__ import annotations

import ir_measures
from ir_measures import R, RPrec, nDCG


def compute_retrieval_metrics(records: list[dict], top_k: int = 10) -> dict[str, float]:
    ks = [k for k in (1, 5, 10, 50) if k <= top_k]
    measures = [nDCG @ k for k in ks] + [R @ k for k in ks] + [RPrec]

    qrels = {r["instance_id"]: {sid: 1 for sid in r["gold_skill_ids"]} for r in records}
    run = {
        r["instance_id"]: {e["skill_id"]: -i for i, e in enumerate(r["retrieved"])}
        for r in records
    }

    scores = ir_measures.calc_aggregate(measures, qrels, run)
    metrics = {str(measure): value for measure, value in scores.items()}
    for k in ks:
        metrics[f"Recall@{k}"] = metrics.pop(f"R@{k}")
    metrics["R-Precision"] = metrics.pop("Rprec")
    return metrics
