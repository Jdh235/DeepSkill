from __future__ import annotations

import json
from pathlib import Path

from pyterrier_retrieval.units import skill_usage_units_from_instructions, sra_bench_units

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent
DATA = REPO_ROOT / "data"
SRA = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]


def main() -> None:
    lines = (line.strip() for line in (REPO_ROOT / "annotation" / "tasks.txt").read_text().splitlines())
    tasks = {t for t in lines if t and not t.startswith("#")}
    units = {ds: sra_bench_units(DATA / "sra_bench" / "instances" / f"{ds}.json") for ds in SRA}
    units["skill_usage"] = skill_usage_units_from_instructions(DATA / "skill_usage" / "tasks",
                                                               DATA / "skill_usage" / "task_skill_mapping.json")
    written = 0
    for ds, ds_units in units.items():
        rows = sorted((u["id"], u["queries"][0]) for u in ds_units if u["id"] in tasks)
        (OUT_DIR / f"{ds}.jsonl").write_text("".join(
            json.dumps({"query_id": qid, "query": query}, ensure_ascii=False) + "\n" for qid, query in rows))
        written += len(rows)
        print(f"{ds:13} {len(rows)} queries")
    if written != len(tasks):
        raise SystemExit(f"wrote {written} queries for {len(tasks)} annotated tasks -- some tasks are missing from data/")
    print(f"Saved: {OUT_DIR}/{{dataset}}.jsonl")


if __name__ == "__main__":
    main()
