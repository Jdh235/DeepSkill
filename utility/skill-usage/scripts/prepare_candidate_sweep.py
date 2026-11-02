from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_PATH = Path(os.environ.get("SKILL_USAGE_ANNOTATION", REPO_ROOT.parents[1] / "resource" / "annotation" / "top5" / "task_subset" / "skill_usage-annotation.json"))
CORPUS_DIR = Path(os.environ.get("SKILL_USAGE_CORPUS_DIR", REPO_ROOT.parents[1] / "resource" / "data" / "skill_usage" / "corpus" / "skill_usage"))
TASKS_DIR = Path(os.environ.get("SKILL_USAGE_TASKS_DIR", REPO_ROOT.parents[1] / "resource" / "data" / "skill_usage" / "tasks"))

BASE_IMAGE_PREFIX = "skill-usage-sweep-base"
CANDIDATE_IMAGE_PREFIX = "skill-usage-sweep"

_SKILL_COPY_RE = re.compile(r'^\s*COPY\s+skills/?\s+(\S+)\s*$', re.MULTILINE)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def image_exists(tag: str) -> bool:
    result = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    return result.returncode == 0


def build_task_base_image(task_name: str, force: bool = False) -> str:
    """Build (or reuse) the task's normal environment image, tagged for reuse
    across all of that task's candidate-skill trials."""
    tag = f"{BASE_IMAGE_PREFIX}:{task_name}"
    if not force and image_exists(tag):
        return tag
    env_dir = TASKS_DIR / task_name / "environment"
    print(f"  building base image {tag} from {env_dir} ...")
    run(["docker", "build", "--network=host", "-t", tag, str(env_dir)])
    return tag


def skill_copy_targets(task_name: str) -> list[str]:
    """Every path this task's Dockerfile copies its skills/ folder to (one
    per agent convention -- usually 6-9, e.g. /root/.claude/skills,
    /root/.codex/skills, ... -- detected per task rather than assumed,
    since some tasks use /app/... instead of /root/...)."""
    dockerfile = TASKS_DIR / task_name / "environment" / "Dockerfile"
    text = dockerfile.read_text()
    targets = _SKILL_COPY_RE.findall(text)
    return [t for t in targets if t and t != "skills"]


def stage_candidate_skill(skill_id: str, staging_root: Path) -> tuple[str, bool]:
    """Copy <corpus>/<skill_id>/ into a staging dir minus SKILL.md.
    Returns (skill_md_content, has_sibling_files)."""
    src = CORPUS_DIR / skill_id
    skill_md = src / "SKILL.md"
    content = skill_md.read_text().strip()

    dest = staging_root / skill_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    (dest / "SKILL.md").unlink()

    has_siblings = any(dest.rglob("*"))
    if not has_siblings:
        shutil.rmtree(dest)
    return content, has_siblings


def build_candidate_image(task_name: str, base_tag: str, skill_id: str, targets: list[str], staged_skill_dir: Path | None, staging_dir: Path) -> str:
    safe_skill_tag = re.sub(r"[^a-zA-Z0-9_.-]", "-", skill_id).lower()
    tag = f"{CANDIDATE_IMAGE_PREFIX}:{task_name}__{safe_skill_tag}"

    rm_targets = " ".join(targets)
    lines = [f"FROM {base_tag}", f"RUN rm -rf {rm_targets}"]

    if staged_skill_dir is not None:
        build_context = staged_skill_dir.parent
        for target in targets:
            lines.append(f"COPY {staged_skill_dir.name} {target}/{skill_id}/")
    else:
        build_context = staging_dir

    dockerfile_path = build_context / f".Dockerfile.sweep.{safe_skill_tag}"
    dockerfile_path.write_text("\n".join(lines) + "\n")
    try:
        run(["docker", "build", "--network=host", "-f", str(dockerfile_path), "-t", tag, str(build_context)])
    finally:
        dockerfile_path.unlink(missing_ok=True)

    return tag


def write_trial_dir(output_dir: Path, task_name: str, skill_id: str, image_tag: str, skill_md_content: str, deployed_path: str | None) -> Path:
    safe_skill_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", skill_id)
    trial_dir = output_dir / f"{task_name}__{safe_skill_name}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True)

    src = TASKS_DIR / task_name

    (trial_dir / "environment").mkdir()
    (trial_dir / "environment" / "Dockerfile").symlink_to(src / "environment" / "Dockerfile")
    (trial_dir / "tests").symlink_to(src / "tests")
    (trial_dir / "solution").symlink_to(src / "solution")

    task_toml = (src / "task.toml").read_text()
    task_toml = task_toml.replace(
        "[environment]\n",
        f'[environment]\ndocker_image = "{image_tag}"\n',
        1,
    )
    (trial_dir / "task.toml").write_text(task_toml)

    original_instruction = (src / "instruction.md").read_text()
    if deployed_path:
        header = f"Relevant Skill (files located at {deployed_path}):"
    else:
        header = "Relevant Skill:"
    injected = (
        original_instruction.rstrip("\n") + "\n\n" + header + "\n" + skill_md_content + "\n"
    )
    (trial_dir / "instruction.md").write_text(injected)

    return trial_dir


def main() -> None:
    global CORPUS_DIR, TASKS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-names", nargs="*", default=None, help="Restrict to these tasks (default: all 25 in the annotation file)")
    parser.add_argument("--max-candidates", type=int, default=None, help="Cap candidates per task (for smoke testing)")
    parser.add_argument("--staging-dir", type=Path, default=REPO_ROOT / ".local-workspace" / "candidate-sweep-staging")
    parser.add_argument("--annotation", type=Path, default=ANNOTATION_PATH,
                        help="Candidate pool per task (default: $SKILL_USAGE_ANNOTATION or resource/annotation/top5/task_subset/skill_usage-annotation.json)")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR,
                        help="Unzipped skill corpus (default: $SKILL_USAGE_CORPUS_DIR or resource/data/skill_usage/corpus/skill_usage/)")
    parser.add_argument("--tasks-dir", type=Path, default=TASKS_DIR,
                        help="Downloaded SkillsBench tasks (default: $SKILL_USAGE_TASKS_DIR or resource/data/skill_usage/tasks/)")
    args = parser.parse_args()

    CORPUS_DIR = args.corpus_dir.resolve()
    TASKS_DIR = args.tasks_dir.resolve()
    if not CORPUS_DIR.is_dir():
        sys.exit(f"Skill corpus not found at {CORPUS_DIR} -- download it (README step 2) or pass --corpus-dir")
    if not TASKS_DIR.is_dir():
        sys.exit(f"Tasks not found at {TASKS_DIR} -- download them (README step 2) or pass --tasks-dir")

    annotation = json.loads(args.annotation.read_text())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    n_trials = 0
    n_images_built = 0

    for entry in annotation["results"]:
        task_name = entry["instance_id"]
        if args.task_names and task_name not in args.task_names:
            continue
        if not (TASKS_DIR / task_name).is_dir():
            print(f"Skipping {task_name}: not found in {TASKS_DIR}", file=sys.stderr)
            continue

        print(f"=== {task_name} ===")
        base_tag = build_task_base_image(task_name)
        n_images_built += 1
        targets = skill_copy_targets(task_name)
        if not targets:
            print(f"  warning: no skill COPY targets found in {task_name}'s Dockerfile, skipping task", file=sys.stderr)
            continue

        candidates = entry["retrieved"]
        if args.max_candidates:
            candidates = candidates[: args.max_candidates]

        for candidate in candidates:
            skill_id = candidate["skill_id"]
            skill_md_content, has_siblings = stage_candidate_skill(skill_id, args.staging_dir)
            staged_dir = (args.staging_dir / skill_id) if has_siblings else None

            image_tag = build_candidate_image(task_name, base_tag, skill_id, targets, staged_dir, args.staging_dir)
            n_images_built += 1

            deployed_path = f"{targets[0].rstrip('/')}/{skill_id}/" if has_siblings else None
            trial_dir = write_trial_dir(output_dir, task_name, skill_id, image_tag, skill_md_content, deployed_path)
            n_trials += 1
            print(f"  {skill_id}: {'with sibling files' if has_siblings else 'bare SKILL.md'} -> {trial_dir.name}")

    print(f"\nDone. {n_trials} trials generated in {output_dir}, {n_images_built} docker builds.")


if __name__ == "__main__":
    main()
