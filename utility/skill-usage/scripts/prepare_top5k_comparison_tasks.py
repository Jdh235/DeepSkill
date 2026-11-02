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

BASE_IMAGE_PREFIX = "skill-usage-top5k-base"
IMAGE_PREFIX = "skill-usage-top5k"

_SKILL_COPY_RE = re.compile(r'^\s*COPY\s+skills/?\s+(\S+)\s*$', re.MULTILINE)
_CLAUDE_SKILLS_COPY_RE = re.compile(r'^\s*COPY\s+skills/?\s+(\S+/\.claude/skills)\s*$', re.MULTILINE)

TOP_K = 5


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def build_task_base_image(task_name: str) -> str:
    tag = f"{BASE_IMAGE_PREFIX}:{task_name}"
    if image_exists(tag):
        return tag
    env_dir = TASKS_DIR / task_name / "environment"
    print(f"  building base image {tag} from {env_dir} ...")
    run(["docker", "build", "--network=host", "-t", tag, str(env_dir)])
    return tag


def skill_copy_targets(task_name: str) -> list[str]:
    dockerfile = TASKS_DIR / task_name / "environment" / "Dockerfile"
    text = dockerfile.read_text()
    return [t for t in _SKILL_COPY_RE.findall(text) if t and t != "skills"]


def detect_skills_dir(task_name: str) -> str | None:
    dockerfile = TASKS_DIR / task_name / "environment" / "Dockerfile"
    match = _CLAUDE_SKILLS_COPY_RE.search(dockerfile.read_text())
    return match.group(1) if match else None


def top5_candidates(entry: dict) -> list[dict]:
    return sorted(entry["retrieved"], key=lambda c: -c["score"])[:TOP_K]


def stage_skill(skill_id: str, staging_root: Path, strip_skill_md: bool) -> tuple[str, bool]:
    """Copy <corpus>/<skill_id>/ into staging. If strip_skill_md, SKILL.md is
    removed after reading its content (full-injection setting); otherwise
    it's left in place (progressive-disclosure setting).
    Returns (skill_md_content, has_files_to_copy)."""
    src = CORPUS_DIR / skill_id
    skill_md_content = (src / "SKILL.md").read_text().strip()

    dest = staging_root / skill_id
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    if strip_skill_md:
        (dest / "SKILL.md").unlink()

    has_files = any(dest.rglob("*"))
    if not has_files:
        shutil.rmtree(dest)
    return skill_md_content, has_files


def build_setting_image(task_name: str, base_tag: str, targets: list[str], staged_dirs: list[tuple[str, Path | None]], setting: str) -> str:
    """FROM <base_tag> + rm the original oracle skill from every agent path +
    COPY each of the 5 candidates' staged folders in at <target>/<skill_id>/."""
    tag = f"{IMAGE_PREFIX}:{task_name}__{setting}"
    rm_targets = " ".join(targets)
    lines = [f"FROM {base_tag}", f"RUN rm -rf {rm_targets}"]

    build_context = REPO_ROOT / ".local-workspace" / "top5k-staging" / f"{task_name}__{setting}"
    if build_context.exists():
        shutil.rmtree(build_context)
    build_context.mkdir(parents=True)

    for skill_id, staged_dir in staged_dirs:
        if staged_dir is None:
            continue
        dest = build_context / skill_id
        shutil.copytree(staged_dir, dest)
        for target in targets:
            lines.append(f"COPY {skill_id} {target}/{skill_id}/")

    dockerfile_path = build_context / "Dockerfile"
    dockerfile_path.write_text("\n".join(lines) + "\n")
    run(["docker", "build", "--network=host", "-f", str(dockerfile_path), "-t", tag, str(build_context)])
    shutil.rmtree(build_context)

    return tag


def write_task_dir(output_dir: Path, task_name: str, image_tag: str, skills_dir_path: str | None,
                    inject_content: str | None) -> Path:
    task_dir = output_dir / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    src = TASKS_DIR / task_name
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").symlink_to(src / "environment" / "Dockerfile")
    (task_dir / "tests").symlink_to(src / "tests")
    (task_dir / "solution").symlink_to(src / "solution")

    task_toml = (src / "task.toml").read_text()
    insert = f'[environment]\ndocker_image = "{image_tag}"\n'
    if skills_dir_path:
        insert += f'skills_dir = "{skills_dir_path}"\n'
    task_toml = task_toml.replace("[environment]\n", insert, 1)
    (task_dir / "task.toml").write_text(task_toml)

    original_instruction = (src / "instruction.md").read_text()
    if inject_content:
        final = original_instruction.rstrip("\n") + "\n\n" + inject_content + "\n"
    else:
        final = original_instruction
    (task_dir / "instruction.md").write_text(final)

    return task_dir


def main() -> None:
    global CORPUS_DIR, TASKS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--progdisc-output-dir", type=Path, required=True)
    parser.add_argument("--fullinject-output-dir", type=Path, required=True)
    parser.add_argument("--task-names", nargs="*", default=None)
    parser.add_argument("--staging-dir", type=Path, default=REPO_ROOT / ".local-workspace" / "top5k-skill-staging")
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
    progdisc_dir = args.progdisc_output_dir.resolve()
    fullinject_dir = args.fullinject_output_dir.resolve()
    progdisc_dir.mkdir(parents=True, exist_ok=True)
    fullinject_dir.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = 0
    for entry in annotation["results"]:
        task_name = entry["instance_id"]
        if args.task_names and task_name not in args.task_names:
            continue
        if not (TASKS_DIR / task_name).is_dir():
            print(f"Skipping {task_name}: not found", file=sys.stderr)
            continue

        print(f"=== {task_name} ===")
        base_tag = build_task_base_image(task_name)
        targets = skill_copy_targets(task_name)
        if not targets:
            print(f"  warning: no skill COPY targets found, skipping", file=sys.stderr)
            continue
        claude_skills_dir = detect_skills_dir(task_name)

        candidates = top5_candidates(entry)
        skill_ids = [c["skill_id"] for c in candidates]
        print(f"  top-{TOP_K} candidates: {skill_ids}")

        # --- progressive-disclosure setting: full folders (SKILL.md kept) ---
        pd_staged = [
            (sid, args.staging_dir / "progdisc" / sid if stage_skill(sid, args.staging_dir / "progdisc", strip_skill_md=False)[1] else None)
            for sid in skill_ids
        ]
        pd_tag = build_setting_image(task_name, base_tag, targets, pd_staged, "progdisc")
        write_task_dir(progdisc_dir, task_name, pd_tag, claude_skills_dir, inject_content=None)

        # --- full-injection setting: SKILL.md content injected, siblings only copied ---
        fi_contents = []
        fi_staged = []
        for sid in skill_ids:
            content, has_siblings = stage_skill(sid, args.staging_dir / "fullinject", strip_skill_md=True)
            fi_staged.append((sid, (args.staging_dir / "fullinject" / sid) if has_siblings else None))
            deployed_path = f"{targets[0].rstrip('/')}/{sid}/" if has_siblings else None
            header = f"Relevant Skill (files located at {deployed_path}):" if deployed_path else "Relevant Skill:"
            fi_contents.append(f"{header}\n{content}")
        fi_tag = build_setting_image(task_name, base_tag, targets, fi_staged, "fullinject")
        inject_text = "\n---\n".join(fi_contents)
        write_task_dir(fullinject_dir, task_name, fi_tag, claude_skills_dir, inject_content=inject_text)

        n_tasks += 1
        print(f"  progdisc -> {pd_tag}, fullinject -> {fi_tag}")

    print(f"\nDone. {n_tasks} tasks x 2 settings.")
    print(f"  progressive-disclosure tasks: {progdisc_dir}")
    print(f"  full-injection tasks: {fullinject_dir}")


if __name__ == "__main__":
    main()
