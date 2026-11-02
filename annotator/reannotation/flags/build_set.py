"""Build the flag re-annotation set: every (dataset, skill, flag) the round-1
annotators didn't all agree on (reannotation/round1/task_subset_N/flags.json,
one per annotator)."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND1_DIR = HERE.parent / "round1"
UNITS_PATH = HERE / "data" / "units.json"
FLAGS_PATH = HERE / "data" / "flags.json"

sys.path.insert(0, str(HERE.parents[1]))  # repo root, for common.py
from common import FLAG_KEYS  # noqa: E402


def load_flags(src: Path) -> dict[tuple[str, str], dict[str, bool]]:
    """(dataset, skill_id) -> {flag: bool} for one annotator."""
    flags = json.loads((src / "flags.json").read_text())
    return {
        (dataset, skill_id): {k: bool(next(iter(ann.values()), {}).get(k)) for k in FLAG_KEYS}
        for dataset, skills in flags.items()
        for skill_id, ann in skills.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Rebuild an existing set, wiping any re-judged flags")
    args = parser.parse_args()

    if UNITS_PATH.exists() and not args.force:
        raise SystemExit(f"{UNITS_PATH} already exists and data/flags.json holds the re-annotations. Pass --force to rebuild (wipes them).")
    sources = sorted(ROUND1_DIR.glob("task_subset_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if len(sources) < 2:
        raise SystemExit(f"Need at least two annotators' folders in {ROUND1_DIR}.")

    all_flags = [load_flags(src) for src in sources]
    keys = set(all_flags[0])
    for src, flags in zip(sources[1:], all_flags[1:]):
        if set(flags) != keys:
            raise SystemExit(f"{src} doesn't flag the same skills as {sources[0]}.")

    units = [
        {"dataset": dataset, "skill_id": skill_id, "flag": k}
        for dataset, skill_id in sorted(keys)
        for k in FLAG_KEYS
        if len({f[(dataset, skill_id)][k] for f in all_flags}) > 1
    ]
    UNITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNITS_PATH.write_text(json.dumps(units, indent=2) + "\n")
    FLAGS_PATH.write_text("{}\n")

    by_dataset: dict[str, int] = {}
    by_flag = {k: 0 for k in FLAG_KEYS}
    for u in units:
        by_dataset[u["dataset"]] = by_dataset.get(u["dataset"], 0) + 1
        by_flag[u["flag"]] += 1
    for dataset, n in by_dataset.items():
        print(f"{dataset}: {n} flags to re-judge")
    print("by flag: " + ", ".join(f"{k}={v}" for k, v in by_flag.items()))
    n_skills = len({(u["dataset"], u["skill_id"]) for u in units})
    print(f"\n{len(units)} flags to re-judge across {n_skills} skills, written to {UNITS_PATH.parent}")


if __name__ == "__main__":
    main()
