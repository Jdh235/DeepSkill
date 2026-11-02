"""Build the no-skills task set.

Copies each task from the downloaded SkillsBench tasks to --output-dir
unchanged. The base tasks never set environment.skills_dir, so terminus-2
shows the agent no skills.

Copying gives this setting its own folder inside the repository, like every
other setting, so its Harbor configs work wherever the tasks were downloaded.

Usage:
    python3 scripts/prepare_no_skills_tasks.py --output-dir tasks-no-skills
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-names", nargs="+",
                        help="Tasks to copy (default: every task in the annotation file)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=Path(os.environ.get(
            "SKILL_USAGE_ANNOTATION",
            Path(__file__).resolve().parents[3] / "resource" / "annotation" / "top5" / "task_subset"
            / "skill_usage-annotation.json",
        )),
        help="Candidate pool per task (default: $SKILL_USAGE_ANNOTATION or resource/annotation/top5/task_subset/skill_usage-annotation.json)",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(os.environ.get(
            "SKILL_USAGE_TASKS_DIR",
            Path(__file__).resolve().parents[3] / "resource" / "data" / "skill_usage" / "tasks",
        )),
        help="Downloaded SkillsBench tasks (default: $SKILL_USAGE_TASKS_DIR or resource/data/skill_usage/tasks/)",
    )
    args = parser.parse_args()
    if not args.task_names:
        args.task_names = [entry["instance_id"] for entry in json.loads(args.annotation.read_text())["results"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0

    for name in args.task_names:
        src = args.tasks_dir / name
        if not src.is_dir():
            print(f"Skipping {name}: not found at {src}", file=sys.stderr)
            continue
        dest = args.output_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        n_ok += 1
        print(f"{name}: copied")

    print(f"\nDone. {n_ok} of {len(args.task_names)} tasks copied to {args.output_dir}.")


if __name__ == "__main__":
    main()
