from __future__ import annotations

import json
import re
from pathlib import Path

_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def load_sra_bench_corpus(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _read_skill_md(skills_dir: Path, skill_id: str) -> str:
    skill_md = skills_dir / skill_id / "SKILL.md"
    if not skill_md.exists():
        return ""
    text = skill_md.read_text(errors="replace")
    return _FRONTMATTER.sub("", text, count=1).strip()


def load_skill_usage_corpus(meta_path: str | Path) -> list[dict]:
    meta_path = Path(meta_path)
    skills_dir = meta_path.parent
    skills = []
    for line in meta_path.read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        skills.append({
            "skill_id": rec["skill_id"],
            "name": rec.get("name", ""),
            "description": rec.get("description", ""),
            "content": _read_skill_md(skills_dir, rec["skill_id"]),
        })
    return skills


def skill_text(skill: dict) -> str:
    """Build retrieval text for a skill: name + description + content."""
    parts = []
    for key in ("name", "description", "content"):
        if skill.get(key):
            parts.append(skill[key])
    return "\n".join(parts)
