from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # repo root, for common.py
from common import (  # noqa: E402
    CORPUS_PATH,
    DATASET_CAPABILITIES,
    DATASET_DESCRIPTIONS,
    FLAG_DEFINITIONS,
    INSTANCES_DIR,
    RUBRIC,
    SKILL_USAGE_CORPUS_DIR,
    SKILL_USAGE_TASKS_DIR,
)
ANNOTATION_DIR = Path(os.environ.get("ANNOTATION_DIR", str(BASE_DIR / "task_subset"))).resolve()

JUDGE_ROLE = "You are judging whether a candidate skill is relevant to a task. Read the task question and the candidate skill below, assign a relevance score using the rubric, and flag the skill if it's malicious, shows incorrect information, or isn't actionable in this environment."

OUTPUT_FORMAT = """Respond with a single JSON object and nothing else:
{"reasoning": "<one sentence explaining the relevance rating>","rating": <0, 1, 2 or 3>, "malicious": <true|false>, "incorrect_info": <true|false>, "not_actionable": <true|false>}"""

MAX_FILE_READS = 5
MAX_FILE_CHARS = 20_000


def file_access_note(has_task_files: bool, has_skill_files: bool) -> str:
    """Tells the judge what the bundled files are and when to open them.
    Earlier wording only said it *could* open them (or should "when
    necessary"), and judges almost never did (Qwen3.6 on 6 of 750
    SkillsBench pairs, gpt-oss-120b on none), so it now asks for the
    relevant skill files to be opened before judging. gpt-oss also took
    read_file for part of the task environment when judging "Not
    actionable", hence the last paragraph."""
    lines = []
    if has_task_files:
        lines.append(
            "The task comes with the files listed under \"Task files\": the input data and "
            "documents an agent would work with when solving it."
        )
    if has_skill_files:
        lines.append(
            "The skill is bundled with the files listed under \"Skill files\". SKILL.md often only points to them, "
            "so some of what the skill provides may be in these files rather than in SKILL.md."
        )
    how_to_open = []
    if has_skill_files:
        how_to_open.append(
            "Before judging, open the skill files most relevant to the task with the read_file "
            "tool."
        )
    if has_task_files:
        how_to_open.append(
            "Open task files with the read_file tool when you need them to understand what the "
            "task requires."
        )
    how_to_open.append(
        f"You can open at most {MAX_FILE_READS} files per judgment; files longer than "
        f"{MAX_FILE_CHARS:,} characters are truncated, and binary files can't be shown."
    )
    lines.append(" ".join(how_to_open))
    lines.append(
        "The read_file tool is only for you, the judge, to inspect these files. It is not part of "
        "the task's environment."
    )
    return "\n\n".join(lines)


def load_annotation_file(dataset: str) -> dict:
    return json.loads((ANNOTATION_DIR / f"{dataset}-annotation.json").read_text())


@lru_cache(maxsize=None)
def _instance_questions(dataset: str) -> dict[str, str]:
    items = json.loads((INSTANCES_DIR / f"{dataset}.json").read_text())
    return {item["instance_id"]: item.get("question", "") for item in items}


@lru_cache(maxsize=1)
def _corpus() -> dict[str, dict]:
    """skill_id -> {name, description, content}. corpus.json is ~200MB, so
    it's loaded once and only when an sra_bench skill is actually asked for."""
    return {s["skill_id"]: s for s in json.loads(CORPUS_PATH.read_text())}


def load_question(dataset: str, instance_id: str) -> str:
    if dataset == "skill_usage":
        path = SKILL_USAGE_TASKS_DIR / instance_id / "instruction.md"
        return path.read_text() if path.is_file() else "(question not found)"
    return _instance_questions(dataset).get(instance_id, "(question not found)")


def load_skill_markdown(dataset: str, skill_id: str) -> str:
    """Same raw markdown the annotator sees in the Candidate Skill box."""
    if dataset == "skill_usage":
        path = SKILL_USAGE_CORPUS_DIR / skill_id / "SKILL.md"
        return path.read_text() if path.is_file() else f"(SKILL.md not found for {skill_id})"
    skill = _corpus().get(skill_id, {})
    name = skill.get("name", skill_id)
    description = skill.get("description", "")
    content = skill.get("content", "(content not found)")
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{content}"


def _is_docker_file(name: str) -> bool:
    return name == "Dockerfile" or name.startswith("Dockerfile.") or name.startswith("docker-compose.")


def list_bundled_files(base: Path, exclude_top: set[str] = frozenset(), skip_docker: bool = False) -> list[str]:
    """Same file listing app.py shows under "Task files" / "Skill files"."""
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if skip_docker and _is_docker_file(path.name):
            continue
        rel = path.relative_to(base)
        if rel.parts[0] in exclude_top:
            continue
        out.append(rel.as_posix())
    return out


def load_bundled_files(dataset: str, instance_id: str, skill_id: str) -> tuple[list[str], list[str]]:
    """(task_files, skill_files) -- only SkillsBench has these. The task's
    environment/skills/ is excluded, as in the app: it's a copy of the
    gold skill(s), which the judge must stay blind to."""
    if dataset != "skill_usage":
        return [], []
    task_files = list_bundled_files(SKILL_USAGE_TASKS_DIR / instance_id / "environment", exclude_top={"skills"}, skip_docker=True)
    skill_files = list_bundled_files(SKILL_USAGE_CORPUS_DIR / skill_id, exclude_top={"SKILL.md"})
    return task_files, skill_files


def read_bundled_file(dataset: str, instance_id: str, skill_id: str, kind: str, rel_path: str) -> str:
    """Contents of one bundled file, for the judge's read_file tool. Only
    paths in the pair's own file listing are allowed, which also keeps the
    gold skills (environment/skills/) and Dockerfiles out of reach, as in
    the app."""
    task_files, skill_files = load_bundled_files(dataset, instance_id, skill_id)
    if kind == "task":
        listed, base = task_files, SKILL_USAGE_TASKS_DIR / instance_id / "environment"
    elif kind == "skill":
        listed, base = skill_files, SKILL_USAGE_CORPUS_DIR / skill_id
    else:
        return f"Error: kind must be \"task\" or \"skill\", not {kind!r}."
    if rel_path not in listed:
        return f"Error: {rel_path!r} is not one of the listed {kind} files."

    path = base / rel_path
    size = path.stat().st_size
    with path.open("rb") as f:
        # Some task files are 100MB+, so never read more than needed.
        raw = f.read(MAX_FILE_CHARS * 4)
    if b"\x00" in raw:
        return f"(binary file, {size:,} bytes -- contents can't be shown)"
    text = raw.decode("utf-8", errors="replace")
    if text.count("�") > len(text) // 100:
        return f"(binary file, {size:,} bytes -- contents can't be shown)"
    if len(text) > MAX_FILE_CHARS or len(raw) < size:
        return text[:MAX_FILE_CHARS] + f"\n\n[... truncated -- showing the first {MAX_FILE_CHARS:,} characters of a {size:,}-byte file]"
    return text


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(
    dataset: str,
    question: str,
    skill_markdown: str,
    task_files: list[str] = (),
    skill_files: list[str] = (),
    print_prompt: bool = False,
) -> str:
    """Lay out the prompt in the same order as the annotator's task page:
    role, dataset description, task question (+ files), candidate skill
    (+ files), flags, then the rubric -- with each rubric level's full
    criteria, which the annotator learns from the intro page."""
    parts = [JUDGE_ROLE]
    if dataset_description := DATASET_DESCRIPTIONS.get(dataset, ""):
        parts.append(dataset_description)

    task = question.strip()
    if task_files:
        task += f"\n\nTask files ({len(task_files)}):\n" + "\n".join(f"- {f}" for f in task_files)
    parts.append(f"## Task Question\n\n<task>\n{task}\n</task>")

    skill = skill_markdown.strip()
    if skill_files:
        skill += f"\n\nSkill files ({len(skill_files)}):\n" + "\n".join(f"- {f}" for f in skill_files)
    parts.append(f"## Candidate Skill\n\n<skill>\n{skill}\n</skill>")
    if task_files or skill_files:
        parts.append(file_access_note(bool(task_files), bool(skill_files)))

    capability = DATASET_CAPABILITIES.get(dataset, "")
    flag_lines = []
    for key, label, description in FLAG_DEFINITIONS:
        line = f"- {label}: {description}"
        if key == "not_actionable" and capability:
            line += f" {capability}"
        flag_lines.append(line)
    parts.append("## Flags\n\n" + "\n".join(flag_lines))

    parts.append("## Rubric\n\n" + "\n".join(
        f"- {value}: {label}. {criteria}" for value, label, criteria in RUBRIC
    ))

    parts.append(f"## Output Format\n\n{OUTPUT_FORMAT}")

    prompt = "\n\n".join(parts)
    if print_prompt:
        print(prompt)
    return prompt


def build_prompt_for(dataset: str, instance_id: str, skill_id: str, print_prompt: bool = False) -> str:
    """Load everything for one (task, skill) pair from disk and build its prompt."""
    task_files, skill_files = load_bundled_files(dataset, instance_id, skill_id)
    return build_prompt(
        dataset,
        load_question(dataset, instance_id),
        load_skill_markdown(dataset, skill_id),
        task_files,
        skill_files,
        print_prompt=print_prompt,
    )


def build_messages(dataset: str, instance_id: str, skill_id: str, print_prompt: bool = False) -> list[dict]:
    """OpenAI-style chat messages, for any OpenAI-compatible /v1/chat/completions API."""
    return [{"role": "user", "content": build_prompt_for(dataset, instance_id, skill_id, print_prompt)}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. theoremqa, skill_usage")
    parser.add_argument("--instance-id", help="defaults to the first task in the annotation file")
    parser.add_argument("--skill-id", help="defaults to the first retrieved skill for that task")
    args = parser.parse_args()

    results = load_annotation_file(args.dataset)["results"]
    if args.instance_id:
        result = next((r for r in results if r["instance_id"] == args.instance_id), None)
        if result is None:
            parser.error(f"instance {args.instance_id!r} not in {args.dataset}-annotation.json")
    else:
        result = results[0]
    skill_id = args.skill_id or result["retrieved"][0]["skill_id"]

    build_prompt_for(args.dataset, result["instance_id"], skill_id, print_prompt=True)


if __name__ == "__main__":
    main()
