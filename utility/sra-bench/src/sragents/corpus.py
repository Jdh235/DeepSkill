"""Skill corpus loading.

A skill is a dict with (at minimum) ``skill_id``, ``name``, ``description``,
``content``. Some skills additionally expose ``tools`` — executable Python
functions an inference engine may invoke on behalf of the model.
"""

import json
import threading
from pathlib import Path

from sragents.config import CORPUS_PATH

# One parsed corpus per file, so callers that pass an explicit path (e.g. the
# --corpus flag) don't re-read the file for every task.
_cache: dict[Path, dict[str, dict]] = {}
_cache_lock = threading.Lock()


def load_corpus(path: Path | None = None) -> list[dict]:
    """Load the skill corpus as a list.

    Args:
        path: Optional override (default: :data:`sragents.config.CORPUS_PATH`).
    """
    p = Path(path) if path else CORPUS_PATH
    if not p.exists():
        zip_path = p.with_suffix(".json.zip")
        hint = (
            f"\nRun: unzip {zip_path} -d {p.parent}"
            if zip_path.exists() else ""
        )
        raise FileNotFoundError(f"Skill corpus not found: {p}{hint}")
    return json.loads(p.read_text())


def load_corpus_dict(path: Path | None = None) -> dict[str, dict]:
    """Load the corpus keyed by ``skill_id``; each file is read once per process."""
    key = (Path(path) if path else CORPUS_PATH).resolve()
    with _cache_lock:
        if key not in _cache:
            _cache[key] = {s["skill_id"]: s for s in load_corpus(key)}
        return _cache[key]


def skill_text(skill: dict) -> str:
    """Build retrieval text for a skill: name + description + content."""
    parts = []
    for key in ("name", "description", "content"):
        if skill.get(key):
            parts.append(skill[key])
    return "\n".join(parts)


def display_name(skill: dict, index: int | None = None) -> str:
    """Human-readable skill name for prompts shown to the model.

    Falls back to ``"Skill #<index>"`` or ``"Unnamed skill"`` — **never to
    ``skill_id``**, which carries a dataset-name prefix (e.g.
    ``theoremqa_000``) and would leak benchmark identity to the model.
    """
    name = (skill.get("name") or "").strip()
    if name:
        return name
    return f"Skill #{index}" if index is not None else "Unnamed skill"
