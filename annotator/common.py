"""Shared by app.py and the re-annotation apps (reannotation/), so every round
shows the same wording: the rubric, flag definitions and dataset notes, plus
loaders for task questions and skill content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR.parent / "resource" / "data"))).resolve()
INSTANCES_DIR = Path(os.environ.get("INSTANCES_DIR", str(DATA_DIR / "sra_bench" / "instances"))).resolve()
CORPUS_PATH = Path(os.environ.get("CORPUS_PATH", str(DATA_DIR / "sra_bench" / "corpus" / "corpus.json"))).resolve()
SKILL_USAGE_TASKS_DIR = Path(os.environ.get("SKILL_USAGE_TASKS_DIR", str(DATA_DIR / "skill_usage" / "tasks"))).resolve()
SKILL_USAGE_CORPUS_DIR = Path(os.environ.get("SKILL_USAGE_CORPUS_DIR", str(DATA_DIR / "skill_usage" / "corpus" / "skill_usage"))).resolve()

RUBRIC = [
    (0, "Not relevant", "The skill has no meaningful connection to the task. Applying it would not help solve the task in any way."),
    (1, "Marginally relevant", "The skill touches on a related topic, tool, or domain, but doesn't address what the task actually requires. It might be useful background, but wouldn't meaningfully help produce a solution."),
    (2, "Relevant", "The skill is applicable to the task and provides genuinely useful guidance toward a solution, even if it doesn't cover every part of what's needed."),
    (3, "Highly relevant", "The skill directly and substantially addresses the task. Following it would solve the task, or get very close to a complete solution."),
]
FLAG_DEFINITIONS = [
    ("malicious", "Malicious", "The skill's content is designed to cause harm. For example, exfiltrating data, installing backdoors, or other destructive actions. This is judged on the skill itself, independent of whether it's relevant to this task."),
    ("incorrect_info", "Incorrect information", "The skill's content shows factually incorrect information. It should be flagged regardless of its relevance to the task."),
    ("not_actionable", "Not actionable in this environment", "The skill would need a capability (e.g. code execution, web search) that isn't available in this dataset's environment to actually be used, even if it's conceptually relevant to the task. Judged on the skill itself, independent of its relevance rating."),
]
FLAG_KEYS = [key for key, _, _ in FLAG_DEFINITIONS]
DATASET_DESCRIPTIONS: dict[str, str] = {
    "theoremqa": "Tasks in this dataset require applying a mathematical theorem to answer a question.",
    "logicbench": "Tasks in this dataset test a specific logical reasoning pattern.",
    "champ": "Tasks in this dataset require applying a mathematical concept to solve a problem.",
    "medcalcbench": "Tasks in this dataset require computing a medical calculator score or value.",
    "bigcodebench": "Tasks in this dataset require writing code that uses a specific software library.",
    "skill_usage": "Tasks in this dataset are broad, real-world agentic tasks (DevOps, security, data analysis, scientific computing, and more).",
}
DATASET_CAPABILITIES: dict[str, str] = {
    "theoremqa": "This dataset's environment has no web search and no code execution.",
    "logicbench": "This dataset's environment has no web search and no code execution.",
    "champ": "This dataset's environment has no web search and no code execution.",
    "medcalcbench": "This dataset's environment has no web search and no general code execution, but calculator functions are enabled.",
    "bigcodebench": "This dataset's environment expects Python code as its final textual answer. However, it does not support web search and code execution as intermittent steps.",
    "skill_usage": "This dataset's environment supports both web search and code execution, in a command line interface.",
}


def flag_definitions_for(dataset: str) -> list[tuple[str, str, str, str]]:
    """FLAG_DEFINITIONS as (key, label, description, capability) tuples --
    capability is the current dataset's environment note, rendered as its
    own bold line under the "not_actionable" flag's description. Empty for
    the other two flags, and for every flag when there's no specific
    dataset (e.g. the intro page)."""
    capability = DATASET_CAPABILITIES.get(dataset, "")
    return [
        (key, label, description, capability if key == "not_actionable" else "")
        for key, label, description in FLAG_DEFINITIONS
    ]
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "skill_usage": "SkillsBench",
}


def dataset_display_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


def load_instance_text() -> dict[str, dict[str, str]]:
    """dataset name -> {instance_id: question}."""
    out: dict[str, dict[str, str]] = {}
    for path in INSTANCES_DIR.glob("*.json"):
        dataset = path.stem
        items = json.loads(path.read_text())
        out[dataset] = {item["instance_id"]: item.get("question", "") for item in items}
    return out


def load_skill_usage_instance_text() -> dict[str, str]:
    """task name -> instruction.md contents. SkillsBench instances live as
    SKILL_USAGE_TASKS_DIR/<name>/instruction.md rather than in a single
    JSON like the sra_bench datasets."""
    out: dict[str, str] = {}
    if not SKILL_USAGE_TASKS_DIR.is_dir():
        return out
    for task_dir in SKILL_USAGE_TASKS_DIR.iterdir():
        instruction = task_dir / "instruction.md"
        if instruction.is_file():
            out[task_dir.name] = instruction.read_text()
    return out


def load_skill_usage_skill_md(skill_id: str) -> str | None:
    """SkillsBench skills are plain SKILL.md files (already have their own
    YAML frontmatter) under data/skill_usage/corpus/skill_usage/<skill_id>/,
    read directly rather than pre-loaded like the sra_bench CORPUS."""
    path = SKILL_USAGE_CORPUS_DIR / skill_id / "SKILL.md"
    return path.read_text() if path.is_file() else None


def _is_docker_file(name: str) -> bool:
    return name == "Dockerfile" or name.startswith("Dockerfile.") or name.startswith("docker-compose.")


def list_bundled_files(base: Path, exclude_top: set[str] = frozenset(), skip_docker: bool = False) -> list[str]:
    """Relative (POSIX-style) paths of every file under base, skipping any
    top-level entry named in exclude_top. Used to surface a skill's or
    task's bundled files (references, scripts, data, ...) alongside the
    main SKILL.md / instruction.md text, which only link to them by name.
    skip_docker (task files only -- every task environment has one, and
    it's environment boilerplate rather than task-relevant content) drops
    Dockerfile/docker-compose.* at any depth."""
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


def _resolve_skill_usage_file(kind: str, item_id: str, rel_path: str) -> Path:
    """Resolve a bundled-file request to a path on disk, guarding against
    path traversal (".."  in rel_path) and, for task files, against
    reaching into environment/skills/ -- that subfolder is a copy of the
    task's gold skill(s), which annotators must stay blind to (see
    list_bundled_files call site in show_task)."""
    if kind == "task":
        base = (SKILL_USAGE_TASKS_DIR / item_id / "environment").resolve()
    elif kind == "skill":
        base = (SKILL_USAGE_CORPUS_DIR / item_id).resolve()
    else:
        raise HTTPException(status_code=400)
    target = (base / rel_path).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=400)
    if kind == "task" and Path(rel_path).parts[0] == "skills":
        raise HTTPException(status_code=403)
    if not target.is_file():
        raise HTTPException(status_code=404)
    return target


def load_corpus(needed_skill_ids: set[str]) -> dict[str, dict]:
    """skill_id -> {name, description, content}, filtered down to only the
    skill_ids referenced by loaded annotation files (corpus.json itself is
    ~26k skills / 200+MB, most of it irrelevant to this pilot subset)."""
    all_skills = json.loads(CORPUS_PATH.read_text())
    return {
        s["skill_id"]: s
        for s in all_skills
        if s["skill_id"] in needed_skill_ids
    }


def google_translate_url(text: str, max_chars: int = 1800) -> str:
    """Link that opens Google Translate pre-filled with (a prefix of) the
    given text, auto-detecting the source language."""
    return f"https://translate.google.com/?sl=auto&tl=en&text={quote(text[:max_chars])}&op=translate"


# --- Used by the re-annotation apps ---

def skill_markdown(dataset: str, skill_id: str, corpus: dict[str, dict]) -> str:
    """The skill as shown to annotators: SkillsBench SKILL.md as-is, SRA-Bench
    skills rebuilt with the same frontmatter app.py's show_task uses."""
    if dataset == "skill_usage":
        return load_skill_usage_skill_md(skill_id) or f"(SKILL.md not found for {skill_id})"
    skill = corpus.get(skill_id, {})
    name = skill.get("name", skill_id)
    description = skill.get("description", "")
    content = skill.get("content", "(content not found)")
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{content}"


def skill_files(skill_id: str) -> list[tuple[str, str]]:
    """(label, url) for a SkillsBench skill's bundled files, served by /skill-usage-file."""
    return [
        (rel, f"/skill-usage-file?kind=skill&id={quote(skill_id)}&path={quote(rel)}")
        for rel in list_bundled_files(SKILL_USAGE_CORPUS_DIR / skill_id, exclude_top={"SKILL.md"})
    ]


def task_files(instance_id: str) -> list[tuple[str, str]]:
    """(label, url) for a SkillsBench task's environment files, excluding
    environment/skills/ (a copy of the task's gold skill(s), which annotators
    must stay blind to) and Docker boilerplate."""
    return [
        (rel, f"/skill-usage-file?kind=task&id={quote(instance_id)}&path={quote(rel)}")
        for rel in list_bundled_files(SKILL_USAGE_TASKS_DIR / instance_id / "environment", exclude_top={"skills"}, skip_docker=True)
    ]


def load_seed(seed_file: Path, app_name: str) -> int:
    """Seed fixing an app's walk order: ANNOTATION_SEED env var if set (local
    testing only), else seed_file, written once per checkout by set_seed.py."""
    if "ANNOTATION_SEED" in os.environ:
        return int(os.environ["ANNOTATION_SEED"])
    if seed_file.exists():
        return int(seed_file.read_text().strip())
    raise RuntimeError(
        f"No seed set. Run `python3 set_seed.py --app {app_name}` once to create "
        f"{seed_file.relative_to(BASE_DIR)} before starting the app."
    )
