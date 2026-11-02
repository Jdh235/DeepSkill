from __future__ import annotations

import json
from pathlib import Path

from pyterrier_retrieval.metrics import compute_retrieval_metrics


def write_subset(path: Path) -> Path | None:
    path = Path(path).resolve()
    results_dir = next((p for p in path.parents if p.name == "results"), None)
    if results_dir is None:
        return None
    rel = path.relative_to(results_dir)
    if rel.parts[0] == "subset":
        return None

    tasks_path = results_dir.parent / "annotation" / "tasks.txt"
    if not tasks_path.exists():
        print(f"  [warn] {tasks_path} not found -- no subset results for {rel}")
        return None
    lines = (line.strip() for line in tasks_path.read_text().splitlines())
    task_ids = {t for t in lines if t and not t.startswith("#")}

    data = json.loads(path.read_text())
    records = [r for r in data.get("results", []) if r["instance_id"] in task_ids]
    if not records:
        return None

    metadata = {**data["metadata"], "n_queries": len(records)}
    metrics = compute_retrieval_metrics(records, top_k=metadata["top_k"])

    out_path = results_dir / "subset" / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metadata": metadata, "metrics": metrics, "results": records},
                                   ensure_ascii=False, indent=2))
    return out_path
