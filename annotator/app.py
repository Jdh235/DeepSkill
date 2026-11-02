from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common import (
    DATASET_DESCRIPTIONS,
    FLAG_DEFINITIONS,
    RUBRIC,
    SKILL_USAGE_CORPUS_DIR,
    SKILL_USAGE_TASKS_DIR,
    _resolve_skill_usage_file,
    dataset_display_name,
    flag_definitions_for,
    google_translate_url,
    list_bundled_files,
    load_corpus,
    load_instance_text,
    load_skill_usage_instance_text,
    load_skill_usage_skill_md,
)

BASE_DIR = Path(__file__).parent
ANNOTATION_DIR = Path(os.environ.get("ANNOTATION_DIR", str(BASE_DIR / "task_subset"))).resolve()

ANNOTATOR = "annotator_1"

_SEED_FILE = BASE_DIR / "annotation_seed.txt"
if "ANNOTATION_SEED" in os.environ:
    ANNOTATION_SEED = int(os.environ["ANNOTATION_SEED"])
elif _SEED_FILE.exists():
    ANNOTATION_SEED = int(_SEED_FILE.read_text().strip())
else:
    raise RuntimeError(
        f"No seed set. Run `python3 set_seed.py` once to create {_SEED_FILE.name} "
        f"before starting the app."
    )

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def _dataset_name(path: Path) -> str:
    return path.name.removesuffix("-annotation.json")


def load_annotation_files() -> dict[str, dict]:
    """dataset name -> full parsed annotation JSON (kept in memory, mutated
    and re-saved on each POST)."""
    files = sorted(ANNOTATION_DIR.glob("*-annotation.json"))
    return {_dataset_name(p): json.loads(p.read_text()) for p in files}


def save_dataset(dataset: str) -> None:
    for path in ANNOTATION_DIR.glob("*-annotation.json"):
        if _dataset_name(path) == dataset:
            path.write_text(json.dumps(ANNOTATIONS[dataset], ensure_ascii=False, indent=2) + "\n")
            return
    raise HTTPException(status_code=404)

FLAGS_PATH = ANNOTATION_DIR / "flags.json"


def load_flags() -> dict:
    if FLAGS_PATH.exists():
        return json.loads(FLAGS_PATH.read_text())
    return {}


def save_flags() -> None:
    FLAGS_PATH.write_text(json.dumps(FLAGS, ensure_ascii=False, indent=2) + "\n")


def get_flags(dataset: str, skill_id: str, annotator: str) -> dict:
    return FLAGS.get(dataset, {}).get(skill_id, {}).get(annotator, {})


def set_flags(dataset: str, skill_id: str, annotator: str, flags: dict) -> None:
    FLAGS.setdefault(dataset, {}).setdefault(skill_id, {})[annotator] = flags
    save_flags()


def add_missing_gold(data: dict) -> int:
    """Some gold_skill_ids never made it into the pooled "retrieved" list
    (dataset "metrics.n_gold_not_retrieved" tracks this). Append a
    candidate entry for each so it still gets shown and rated -- otherwise
    a gold skill could go completely unjudged. Idempotent: skips ids
    already present, so re-running (e.g. on every startup) is a no-op once
    added."""
    added = 0
    for result in data["results"]:
        retrieved_ids = {c["skill_id"] for c in result["retrieved"]}
        for skill_id in result["gold_skill_ids"]:
            if skill_id not in retrieved_ids:
                result["retrieved"].append({
                    "skill_id": skill_id,
                    "score": None,
                    "is_gold": True,
                    "annotations": {ANNOTATOR: None},
                })
                retrieved_ids.add(skill_id)
                added += 1
    return added


ANNOTATIONS = load_annotation_files()
FLAGS = load_flags()
INSTANCE_TEXT = load_instance_text()
if "skill_usage" in ANNOTATIONS:
    INSTANCE_TEXT["skill_usage"] = load_skill_usage_instance_text()

for _dataset, _data in ANNOTATIONS.items():
    if add_missing_gold(_data):
        save_dataset(_dataset)

_needed_ids = {
    cand["skill_id"]
    for data in ANNOTATIONS.values()
    for result in data["results"]
    for cand in result["retrieved"]
}
CORPUS = load_corpus(_needed_ids)
DATASET_ORDER: list[str] = sorted(ANNOTATIONS)
random.Random(ANNOTATION_SEED).shuffle(DATASET_ORDER)


@dataclass
class SkillPosition:
    dataset: str
    result_index: int
    candidate: dict 
    task_ordinal: int
    task_total: int
    skill_ordinal: int
    skill_total: int


def _build_skill_positions() -> list[SkillPosition]:
    rng = random.Random(ANNOTATION_SEED)

    tasks: list[tuple[str, int]] = []
    for dataset in DATASET_ORDER:
        indices = list(range(len(ANNOTATIONS[dataset]["results"])))
        rng.shuffle(indices)
        tasks.extend((dataset, i) for i in indices)
    task_total = len(tasks)

    positions: list[SkillPosition] = []
    for task_ordinal, (dataset, i) in enumerate(tasks, start=1):
        result = ANNOTATIONS[dataset]["results"][i]
        retrieved = result["retrieved"]
        order = list(range(len(retrieved)))
        rng.shuffle(order)
        skill_total = len(order)
        for skill_ordinal, idx in enumerate(order, start=1):
            positions.append(SkillPosition(
                dataset=dataset,
                result_index=i,
                candidate=retrieved[idx],
                task_ordinal=task_ordinal,
                task_total=task_total,
                skill_ordinal=skill_ordinal,
                skill_total=skill_total,
            ))
    return positions

POSITIONS: list[SkillPosition] = _build_skill_positions()
DATASET_START: dict[str, int] = {}
for _i, _sp in enumerate(POSITIONS, start=1):
    DATASET_START.setdefault(_sp.dataset, _i)


def position_at(position: int) -> SkillPosition:
    if position < 1 or position > len(POSITIONS):
        raise HTTPException(status_code=404)
    return POSITIONS[position - 1]


def _start_position() -> int:
    """First not-yet-rated position, or 1 if everything's rated."""
    for position, sp in enumerate(POSITIONS, start=1):
        if sp.candidate["annotations"].get(ANNOTATOR) is None:
            return position
    return 1


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("intro.html", {
        "request": request,
        "start_position": _start_position(),
        "rubric": RUBRIC,
        "flag_definitions": flag_definitions_for(""),
    })


@app.get("/skill-usage-file")
def skill_usage_file(kind: str, id: str, path: str):
    """Serves one bundled file (task environment or skill reference/script)
    for opening/downloading in a new tab -- see list_bundled_files."""
    target = _resolve_skill_usage_file(kind, id, path)
    return FileResponse(target, filename=target.name, content_disposition_type="inline")


@app.get("/task/{position}")
def show_task(request: Request, position: int, saved: bool = False):
    sp = position_at(position)
    result = ANNOTATIONS[sp.dataset]["results"][sp.result_index]
    instance_id = result["instance_id"]
    question = INSTANCE_TEXT.get(sp.dataset, {}).get(instance_id, "(question not found)")

    task_files: list[tuple[str, str]] = []
    skill_files: list[tuple[str, str]] = []
    if sp.dataset == "skill_usage":
        raw_markdown = load_skill_usage_skill_md(sp.candidate["skill_id"]) or f"(SKILL.md not found for {sp.candidate['skill_id']})"
        task_files = [
            (rel, f"/skill-usage-file?kind=task&id={quote(instance_id)}&path={quote(rel)}")
            for rel in list_bundled_files(SKILL_USAGE_TASKS_DIR / instance_id / "environment", exclude_top={"skills"}, skip_docker=True)
        ]
        skill_files = [
            (rel, f"/skill-usage-file?kind=skill&id={quote(sp.candidate['skill_id'])}&path={quote(rel)}")
            for rel in list_bundled_files(SKILL_USAGE_CORPUS_DIR / sp.candidate["skill_id"], exclude_top={"SKILL.md"})
        ]
    else:
        skill = CORPUS.get(sp.candidate["skill_id"], {})
        name = skill.get("name", sp.candidate["skill_id"])
        description = skill.get("description", "")
        content = skill.get("content", "(content not found)")
        raw_markdown = f"---\nname: {name}\ndescription: {description}\n---\n\n{content}"
    flags = get_flags(sp.dataset, sp.candidate["skill_id"], ANNOTATOR)
    candidate = {
        "skill_id": sp.candidate["skill_id"],
        "raw_markdown": raw_markdown,
        "rating": sp.candidate["annotations"].get(ANNOTATOR),
        "translate_url": google_translate_url(raw_markdown),
        "flags": {key: flags.get(key, False) for key, _, _ in FLAG_DEFINITIONS},
    }

    done = sum(1 for p in POSITIONS if p.candidate["annotations"].get(ANNOTATOR) is not None)

    return templates.TemplateResponse("task.html", {
        "request": request,
        "dataset": sp.dataset,
        "dataset_label": dataset_display_name(sp.dataset),
        "dataset_description": DATASET_DESCRIPTIONS.get(sp.dataset, ""),
        "datasets": [(d, dataset_display_name(d), DATASET_START[d]) for d in DATASET_ORDER],
        "task_ordinal": sp.task_ordinal,
        "task_total": sp.task_total,
        "skill_ordinal": sp.skill_ordinal,
        "skill_total": sp.skill_total,
        "question": question,
        "task_files": task_files,
        "skill_files": skill_files,
        "candidate": candidate,
        "flag_definitions": flag_definitions_for(sp.dataset),
        "rubric": RUBRIC,
        "position": position,
        "total": len(POSITIONS),
        "done": done,
        "prev_position": position - 1 if position > 1 else None,
        "next_position": position + 1 if position < len(POSITIONS) else None,
        "saved": saved,
    })


@app.post("/task/{position}")
async def save_task(request: Request, position: int):
    sp = position_at(position)
    form = await request.form()
    value = form.get("rating", "")
    sp.candidate["annotations"][ANNOTATOR] = int(value) if value != "" else None
    save_dataset(sp.dataset)

    set_flags(sp.dataset, sp.candidate["skill_id"], ANNOTATOR, {
        key: (key in form) for key, _, _ in FLAG_DEFINITIONS
    })

    next_position = position + 1 if position < len(POSITIONS) else position
    return RedirectResponse(f"/task/{next_position}?saved=true", status_code=303)
