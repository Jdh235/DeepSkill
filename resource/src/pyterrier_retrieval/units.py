from __future__ import annotations

import json
from pathlib import Path


def sra_bench_units(instances_path: str | Path) -> list[dict]:
    """SRA-Bench instances -> units (one query per unit, from build_prompt())."""
    from pyterrier_retrieval.prompts import build_prompt

    instances = json.loads(Path(instances_path).read_text())
    units = []
    for inst in instances:
        gold_ids = inst.get("skill_annotations", [])
        if not gold_ids:
            continue
        system, user = build_prompt(inst)
        query = f"{user}\n{system}" if system else user
        units.append({"id": inst["instance_id"], "queries": [query], "gold_skill_ids": gold_ids})
    return units


def _skill_usage_gold(mapping: dict, task: str) -> list[str]:
    return [f"benchflow-ai--{name}" for name in mapping.get(task, [])]


def skill_usage_units_from_instructions(tasks_dir: str | Path, mapping_path: str | Path) -> list[dict]:
    tasks_dir = Path(tasks_dir)
    mapping = json.loads(Path(mapping_path).read_text())
    units = []
    for task_dir in sorted(tasks_dir.iterdir()):
        instruction_path = task_dir / "instruction.md"
        if not task_dir.is_dir() or not instruction_path.exists():
            continue
        task = task_dir.name
        query = instruction_path.read_text().strip()
        units.append({"id": task, "queries": [query], "gold_skill_ids": _skill_usage_gold(mapping, task)})
    return units


def interleave(ranked_lists: list[list[str]], top_k: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    max_len = max((len(r) for r in ranked_lists), default=0)
    for rank in range(max_len):
        for r_list in ranked_lists:
            if rank < len(r_list):
                sid = r_list[rank]
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
                    if len(result) >= top_k:
                        return result
    return result
