from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
TASKS_FILE = HERE / "tasks.txt"
SRA_BASELINES = REPO_ROOT / "agent_results" / "SRA-Bench" / "all_tasks" / "baselines"
SRA_INSTANCES = REPO_ROOT / "data" / "sra_bench" / "instances"
SKILL_USAGE_POPULATION = HERE / "skill_usage_population.txt"
SKILL_USAGE_TASKS = REPO_ROOT / "data" / "skill_usage" / "tasks"
POOL_DIR = REPO_ROOT / "pool" / "top5"

SRA_DATASETS = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]
BASELINE_METHOD = "llm_direct"
EVAL_TYPE_FIELD = {"logicbench": "task_type", "theoremqa": "answer_type", "medcalcbench": "output_type"}


def discover_models(baselines_root: Path) -> list[str]:
    return sorted(p.name for p in baselines_root.iterdir() if p.is_dir())


def discover_runs(model_dir: Path) -> list[int]:
    runs = []
    for p in model_dir.iterdir():
        if p.is_dir() and p.name.startswith("run"):
            try:
                runs.append(int(p.name[len("run"):]))
            except ValueError:
                continue
    return sorted(runs)


def load_baseline_correctness(baselines_root: Path) -> dict[str, dict[int, dict[str, dict[str, bool]]]]:
    """{model: {run: {dataset: {instance_id: correct}}}}, from eval/{dataset}/{model}/{BASELINE_METHOD}.json
    under each model/run in baselines_root. Reads BASELINE_METHOD only, NOT every *.json in the eval dir: an
    oracle_skill.json alongside it uses a different definition of "correct" and would overwrite the real
    no-skill result for every overlapping instance_id."""
    result: dict[str, dict[int, dict[str, dict[str, bool]]]] = {}
    for model in discover_models(baselines_root):
        model_dir = baselines_root / model
        result[model] = {}
        for run in discover_runs(model_dir):
            result[model][run] = {}
            for dataset in SRA_DATASETS:
                eval_path = model_dir / f"run{run}" / "eval" / dataset / model / f"{BASELINE_METHOD}.json"
                correctness: dict[str, bool] = {}
                if eval_path.exists():
                    for d in json.loads(eval_path.read_text())["details"]:
                        correctness[d["instance_id"]] = d["correct"]
                result[model][run][dataset] = correctness
    return result


def eval_types(dataset: str) -> dict[str, str]:
    """{instance_id: eval-type value}, or {} for a dataset with no type field."""
    field = EVAL_TYPE_FIELD.get(dataset)
    if field is None:
        return {}
    instances = json.loads((SRA_INSTANCES / f"{dataset}.json").read_text())
    return {item["instance_id"]: str(item["eval_data"].get(field, "unknown")) for item in instances}


def parse_task_toml_metadata(task_id: str) -> dict[str, str]:
    """{"difficulty": ..., "category": ...} from a Skill-Usage task.toml's [metadata] section (single-line
    key = "value" pairs; a regex rather than a TOML parser so it runs on Python 3.9, which has no tomllib)."""
    text = (SKILL_USAGE_TASKS / task_id / "task.toml").read_text()
    section = re.search(r"\[metadata\](.*?)(?=\n\[|\Z)", text, re.S)
    body = section.group(1) if section else ""
    result = {}
    for field in ("difficulty", "category"):
        m = re.search(rf'^{field}\s*=\s*"([^"]*)"', body, re.M)
        if m:
            result[field] = m.group(1)
    return result


def reference_skills_by_iid(dataset: str) -> dict[str, list[str]]:
    """{instance_id: reference skill ids}, from the dataset's judgment pool (every task, not just the sample)."""
    data = json.loads((POOL_DIR / f"{dataset}-pool.json").read_text())
    return {r["instance_id"]: r["gold_skill_ids"] for r in data["results"]}


N_PER_SRA_DATASET = 10
N_SKILL_USAGE = 25
DEFAULT_ATTEMPTS = 20_000
SEED = 42 


def task_accuracy_by_model(dataset: str, baseline: dict, models: list[str]) -> dict[str, dict[str, float]]:
    """{instance_id: {model: accuracy}} -- only instances with a defined
    accuracy (at least one available run) for EVERY model, so every
    candidate task has a complete accuracy vector to rank/average with.

    Iterates instance_ids in sorted (not set-native) order deliberately:
    a set's iteration order for strings depends on Python's per-process
    hash randomization (PYTHONHASHSEED), not just insertion order, so an
    unsorted set here would make dict/group ordering downstream (and
    ultimately the sequence of rng.sample() calls in sample_dataset)
    vary between separate runs even with the identical SEED. Sorting
    once here fixes that for everything built from this function's
    output.
    """
    all_iids: set[str] = set()
    for m in models:
        for run in baseline[m]:
            all_iids |= set(baseline[m][run].get(dataset, {}).keys())

    per_task: dict[str, dict[str, float]] = {}
    for iid in sorted(all_iids):
        accs = {}
        for m in models:
            passed, available = 0, 0
            for run in baseline[m]:
                correct = baseline[m][run].get(dataset, {}).get(iid)
                if correct is not None:
                    available += 1
                    passed += int(correct)
            if available == 0:
                break
            accs[m] = passed / available
        else:
            per_task[iid] = accs
    return per_task


def ordered_sum(values) -> float:
    """Sum floats strictly left to right. Python 3.12 changed the built-in sum() for floats to a compensated
    summation, which shifts the last bits of these means and deviations and so changes which near-tied draw wins;
    adding in order gives the same result, and the same tasks.txt, on every Python version."""
    total = 0
    for v in values:
        total += v
    return total


def population_stats(per_task: dict[str, dict[str, float]], models: list[str]) -> tuple[tuple[str, ...], dict[str, float]]:
    means = {m: ordered_sum(a[m] for a in per_task.values()) / len(per_task) for m in models}
    ranking = tuple(sorted(models, key=lambda m: means[m], reverse=True))
    return ranking, means


def gold_complexity_by_iid(dataset: str) -> dict[str, str]:
    """{instance_id: "simple" (1 gold skill) or "multi" (2+)}."""
    gold_by_iid = reference_skills_by_iid(dataset)
    return {iid: ("simple" if len(gold) == 1 else "multi") for iid, gold in gold_by_iid.items()}


def stratum_groups(dataset: str, per_task: dict[str, dict[str, float]]) -> dict[tuple[str, str], list[str]]:
    """{(eval_type, gold_complexity): [instance_id, ...]} restricted to
    per_task's own iids. eval_type collapses to "ALL" for datasets with
    no type field (bigcodebench/champ); gold_complexity is degenerate
    (always the same value) for every dataset except champ -- see this
    module's docstring.
    """
    types = eval_types(dataset)
    complexity = gold_complexity_by_iid(dataset)
    groups: dict[tuple[str, str], list[str]] = {}
    for iid in per_task:
        key = (types.get(iid, "ALL"), complexity.get(iid, "unknown"))
        groups.setdefault(key, []).append(iid)
    return groups


def allocate_with_floor(sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation of `total` across
    `sizes` (population count per key), with a floor of 1 per key when
    total >= number of keys -- so a rare type isn't silently missed by
    chance in a small draw. Guarantees sum(result.values()) ==
    min(total, sum(sizes.values())): a naive version that just clamps
    each key to its own population size AFTER distributing (as an
    earlier version of this function did) can lose units to that clamp
    without redistributing them -- found live via skill_usage's
    (difficulty, gold_complexity) stratification, where a 1-task stratum
    got bumped to a quota of 2 by the leftover-remainder round, got
    silently clamped back to 1, and nobody else got that lost slot: 25
    requested came out as a 24-task draw. This version redistributes any
    shortfall a cap causes to keys that still have room, looping until
    the total actually matches (or every key is fully saturated).
    """
    sizes = {k: v for k, v in sizes.items() if v > 0}
    if not sizes or total <= 0:
        return {k: 0 for k in sizes}
    total = min(total, sum(sizes.values()))  # can't allocate more than exists

    base = {k: min(1, sizes[k]) for k in sizes} if total >= len(sizes) else {k: 0 for k in sizes}
    remaining = total - sum(base.values())

    while remaining > 0:
        room = {k: sizes[k] - base[k] for k in sizes if sizes[k] > base[k]}
        if not room:
            break
        grand = sum(room.values())
        shares = {k: remaining * room[k] / grand for k in room}
        add = {k: min(int(shares[k]), room[k]) for k in room}
        for k in room:
            base[k] += add[k]
        remaining -= sum(add.values())
        if remaining <= 0:
            break
        order = sorted(room, key=lambda k: shares[k] - add[k], reverse=True)
        for k in order:
            if remaining <= 0:
                break
            if base[k] < sizes[k]:
                base[k] += 1
                remaining -= 1

    return base


def rejection_sample_within_quota(
    label: str,
    groups: dict,
    per_task: dict[str, dict[str, float]],
    models: list[str],
    ranking: tuple[str, ...],
    pop_means: dict[str, float],
    n: int,
    attempts: int,
    rng: random.Random,
) -> list[str]:
    """Shared core for both SRA-Bench and skill_usage draws: `groups` is
    a fixed stratification (already computed by the caller -- eval_type
    x gold_complexity for SRA-Bench, difficulty for skill_usage), and
    within the quota it implies, search --attempts random draws for the
    one whose per-model mean (accuracy or reward) ranking matches the
    population's, minimizing total deviation among those. Falls back to
    the closest-deviation draw if no attempt matches the ranking.
    """
    quota = allocate_with_floor({k: len(ids) for k, ids in groups.items()}, n)

    best_matching: tuple[float, list[str]] | None = None
    best_any: tuple[float, list[str]] | None = None
    for _ in range(attempts):
        sample: list[str] = []
        for k, q in quota.items():
            if q:
                sample.extend(rng.sample(groups[k], q))

        sample_means = {m: ordered_sum(per_task[i][m] for i in sample) / n for m in models}
        sample_ranking = tuple(sorted(models, key=lambda m: sample_means[m], reverse=True))
        deviation = ordered_sum(abs(sample_means[m] - pop_means[m]) for m in models)

        if best_any is None or deviation < best_any[0]:
            best_any = (deviation, sample)
        if sample_ranking == ranking and (best_matching is None or deviation < best_matching[0]):
            best_matching = (deviation, sample)

    if best_matching is None:
        print(f"  [warn] {label}: no draw in {attempts} attempts matched the population "
              f"model ranking {ranking} -- using the closest draw instead.")
        return sorted(best_any[1])
    return sorted(best_matching[1])


def sample_dataset(
    dataset: str,
    per_task: dict[str, dict[str, float]],
    models: list[str],
    ranking: tuple[str, ...],
    pop_means: dict[str, float],
    n: int,
    attempts: int,
    rng: random.Random,
) -> list[str]:
    groups = stratum_groups(dataset, per_task)
    return rejection_sample_within_quota(dataset, groups, per_task, models, ranking, pop_means, n, attempts, rng)


def skill_usage_population() -> list[str]:
    """The 57 Skill-Usage tasks whose reference solution earns a consistent reward (comment lines skipped)."""
    return sorted(line.strip() for line in SKILL_USAGE_POPULATION.read_text().splitlines()
                  if line.strip() and not line.startswith("#"))


def skill_usage_stratum_groups(iids: list[str]) -> dict[tuple[str, str], list[str]]:
    """{(difficulty, gold_complexity): [instance_id, ...]}, mirroring
    SRA-Bench's (eval_type, gold_complexity) stratification: difficulty
    (from task.toml) stands in for eval_type here, since skill_usage has
    no such field; gold_complexity is computed the same way as
    gold_complexity_by_iid does for SRA-Bench.
    """
    gold_by_iid = reference_skills_by_iid("skill_usage")
    groups: dict[tuple[str, str], list[str]] = {}
    for iid in sorted(iids):
        difficulty = parse_task_toml_metadata(iid).get("difficulty", "unknown")
        complexity = "simple" if len(gold_by_iid.get(iid, [])) == 1 else "multi"
        groups.setdefault((difficulty, complexity), []).append(iid)
    return groups


def sample_within_groups(groups: dict, n: int, rng: random.Random) -> list[str]:
    """Fixed-quota stratified sample, no search objective: allocate `n`
    across `groups` proportionally (largest-remainder, floor of 1 per
    group present -- allocate_with_floor), then draw randomly within
    each group.
    """
    quota = allocate_with_floor({k: len(ids) for k, ids in groups.items()}, n)
    chosen: list[str] = []
    for k, q in quota.items():
        if q:
            chosen.extend(rng.sample(groups[k], q))
    return sorted(chosen)


def main() -> None:
    global SRA_INSTANCES, SKILL_USAGE_POPULATION, SKILL_USAGE_TASKS, POOL_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS,
                         help=f"rejection-sampling draws to try per SRA-Bench dataset (default: {DEFAULT_ATTEMPTS})")
    parser.add_argument("--sra-baselines", type=Path, default=SRA_BASELINES)
    parser.add_argument("--sra-instances", type=Path, default=SRA_INSTANCES)
    parser.add_argument("--skill-usage-population", type=Path, default=SKILL_USAGE_POPULATION)
    parser.add_argument("--skill-usage-tasks", type=Path, default=SKILL_USAGE_TASKS)
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--output", type=Path, default=TASKS_FILE, help="where to write the task list")
    args = parser.parse_args()
    SRA_INSTANCES, SKILL_USAGE_POPULATION = args.sra_instances, args.skill_usage_population
    SKILL_USAGE_TASKS, POOL_DIR = args.skill_usage_tasks, args.pool_dir

    rng = random.Random(SEED)

    if not args.sra_baselines.exists():
        print(f"{args.sra_baselines} not found -- baseline-stratified sampling needs it.")
        sys.exit(1)
    baseline = load_baseline_correctness(args.sra_baselines)
    models = sorted(baseline.keys())
    print(f"Models: {models}\n")

    chosen: list[str] = []
    for dataset in SRA_DATASETS:
        per_task = task_accuracy_by_model(dataset, baseline, models)
        ranking, pop_means = population_stats(per_task, models)
        picked = sample_dataset(dataset, per_task, models, ranking, pop_means,
                                 N_PER_SRA_DATASET, args.attempts, rng)

        sample_means = {m: ordered_sum(per_task[i][m] for i in picked) / len(picked) for m in models}
        sample_ranking = tuple(sorted(models, key=lambda m: sample_means[m], reverse=True))
        ranking_note = "" if sample_ranking == ranking else f"  [ranking MISMATCH: sample={sample_ranking}]"

        print(f"{dataset}: population ranking {ranking}{ranking_note}")
        for m in models:
            print(f"    {m:25s} pop={pop_means[m]:.3f}  sample={sample_means[m]:.3f}")
        chosen.extend(picked)

    su_groups = skill_usage_stratum_groups(skill_usage_population())
    chosen_skill_usage = sample_within_groups(su_groups, N_SKILL_USAGE, rng)
    print(f"\nskill_usage: stratified on (difficulty, gold_complexity), {len(su_groups)} strata, "
          f"{N_SKILL_USAGE} of {sum(len(g) for g in su_groups.values())} tasks")
    chosen.extend(chosen_skill_usage)

    args.output.write_text("\n".join(chosen) + "\n")
    print(f"\n{len(chosen)} tasks written to {args.output}")


if __name__ == "__main__":
    main()
