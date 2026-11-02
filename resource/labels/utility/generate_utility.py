from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[1]
POOL_DIR = REPO_ROOT / "annotation" / "top5" / "task_subset"

SRA_ROOT = REPO_ROOT / "agent_results" / "SRA-Bench" / "top5_subset50"
SU_ROOT = REPO_ROOT / "agent_results" / "Skill-Usage" / "25_task_set"
SRA_SWEEP = "candidate_sweep"
SRA = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]

Runs = "dict[str, dict[tuple[str, str | None], list[float]]]"


def sra_runs(model: str) -> Runs:
    out: Runs = defaultdict(lambda: defaultdict(list))
    for run in sorted((SRA_ROOT / SRA_SWEEP / model).glob("run*")):
        for ev in sorted(run.glob("eval/*/*/pos*.json")):
            ds = ev.relative_to(run / "eval").parts[0]
            skill_of = {}
            for line in (run / "inference" / ev.relative_to(run / "eval")).with_suffix(".jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    used = r["skill_ids_used"]
                    skill_of[r["instance_id"]] = (ast.literal_eval(used) if isinstance(used, str) else used)[0]
            for det in json.loads(ev.read_text())["details"]:
                out[ds][(det["instance_id"], skill_of[det["instance_id"]])].append(float(det["correct"]))
    for run in sorted((SRA_ROOT / "baselines" / model).glob("run*")):
        for ev in sorted(run.glob("eval/*/*/llm_direct.json")):
            ds = ev.relative_to(run / "eval").parts[0]
            for det in json.loads(ev.read_text())["details"]:
                out[ds][(det["instance_id"], None)].append(float(det["correct"]))
    return out


def su_runs(model: str) -> Runs:
    out: Runs = {"skill_usage": defaultdict(list)}
    for folder, with_skill in (("candidate_sweep", True), ("baselines", False)):
        for path in sorted((SU_ROOT / folder / model).glob("*/result.json")):
            result = json.loads(path.read_text())
            reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            task, _, skill = result["task_name"].partition("__")
            out["skill_usage"][(task, skill if with_skill else None)].append(float(reward) if reward is not None else 0.0)
    return out


def summary(rewards: list[float] | None) -> dict:
    if not rewards:
        return {"rewards": None, "mean": None}
    return {"rewards": [int(r) if float(r).is_integer() else r for r in rewards],
            "mean": round(sum(rewards) / len(rewards), 4)}


def write(model: str, ds: str, runs: dict, source: Path) -> tuple[int, int, set[int]]:
    pool = json.loads((POOL_DIR / f"{ds}-annotation.json").read_text())["results"]
    results, n_run, n_missing, n_runs = [], 0, 0, set()
    for task in pool:
        tid = task["instance_id"]
        candidates = []
        for c in task["retrieved"]:
            rewards = runs.get((tid, c["skill_id"]))
            n_run += bool(rewards); n_missing += not rewards
            if rewards:
                n_runs.add(len(rewards))
            candidates.append({"skill_id": c["skill_id"], "is_reference": c["skill_id"] in task["gold_skill_ids"],
                               **summary(rewards)})
        results.append({"instance_id": tid, "baseline": summary(runs.get((tid, None))), "candidates": candidates,
                        "not_run": [c["skill_id"] for c in task["gold_not_retrieved"]]})
    out = OUT_DIR / ("sra_bench" if ds in SRA else "skill_usage") / model
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ds}-utility.json").write_text(json.dumps({
        "metadata": {
            "dataset": ds, "model": model, "source": display_path(source),
            "reward": "1 if the run's answer is correct, else 0" if ds in SRA
                      else "task verifier reward in [0, 1]; no reward = 0",
            "baseline": "the model on the task with no skill",
            "candidates": "the model on the task with exactly that one skill injected",
            "runs": "independent attempts; baseline and skill runs are not paired",
            "pool": str((POOL_DIR / f"{ds}-annotation.json").relative_to(REPO_ROOT)),
        },
        "results": results}, indent=1))
    return n_run, n_missing, n_runs


def display_path(path: Path) -> str:
    """Repo-relative when inside the repo (as recorded so far), otherwise absolute."""
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def main() -> None:
    global SRA_ROOT, SU_ROOT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sra-runs", type=Path, default=SRA_ROOT, help="SRA-Bench agent runs (baselines/, candidate_sweep/)")
    parser.add_argument("--skill-usage-runs", type=Path, default=SU_ROOT,
                        help="Skill-Usage agent runs (baselines/, candidate_sweep/)")
    args = parser.parse_args()
    SRA_ROOT, SU_ROOT = args.sra_runs, args.skill_usage_runs

    sra_models = sorted(p.name for p in (SRA_ROOT / SRA_SWEEP).iterdir() if p.is_dir() and not p.name.startswith("_"))
    su_models = sorted(p.name for p in (SU_ROOT / "candidate_sweep").iterdir() if p.is_dir())
    jobs = [(m, sra_runs, SRA, SRA_ROOT / SRA_SWEEP) for m in sra_models] + \
           [(m, su_runs, ["skill_usage"], SU_ROOT / "candidate_sweep") for m in su_models]
    for model, load, datasets, source in jobs:
        runs = load(model)
        for ds in datasets:
            n_run, n_missing, n_runs = write(model, ds, runs.get(ds, {}), source)
            baselines = sum(1 for (t, s) in runs.get(ds, {}) if s is None)
            print(f"{model:24} {ds:13} candidates run: {n_run:4}  not run: {n_missing:2}  "
                  f"runs per candidate: {sorted(n_runs)}  tasks with baseline: {baselines}")
    print(f"Wrote {OUT_DIR}/{{sra_bench,skill_usage}}/<model>/<dataset>-utility.json")


if __name__ == "__main__":
    main()
