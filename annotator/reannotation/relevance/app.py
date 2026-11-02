from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common import (
    BASE_DIR as REPO_ROOT,
    DATASET_DESCRIPTIONS,
    FLAG_KEYS,
    RUBRIC,
    _resolve_skill_usage_file,
    dataset_display_name,
    flag_definitions_for,
    google_translate_url,
    load_corpus,
    load_instance_text,
    load_seed,
    load_skill_usage_instance_text,
    skill_files,
    skill_markdown,
    task_files,
)

HERE = Path(__file__).resolve().parent
ANNOTATION_DIR = HERE / "data" / "task_subset"
ROUND1_DIR = HERE.parent / "round1"  
ROUND2_DIR = HERE.parent / "round2"  

if not ANNOTATION_DIR.is_dir():
    raise RuntimeError(f"{ANNOTATION_DIR} not found. Run `python3 reannotation/relevance/build_set.py` first.")

ANNOTATOR = "annotator_1"
REANNOTATION_SEED = load_seed(HERE / "annotation_seed.txt", "relevance")

app = FastAPI()
app.mount("/static", StaticFiles(directory=REPO_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def _dataset_name(path: Path) -> str:
    return path.name.removesuffix("-annotation.json")


def load_annotation_files() -> dict[str, dict]:
    return {_dataset_name(p): json.loads(p.read_text()) for p in sorted(ANNOTATION_DIR.glob("*-annotation.json"))}


def save_dataset(dataset: str) -> None:
    path = ANNOTATION_DIR / f"{dataset}-annotation.json"
    if not path.exists():
        raise HTTPException(status_code=404)
    path.write_text(json.dumps(ANNOTATIONS[dataset], ensure_ascii=False, indent=2) + "\n")


def load_previous_ratings() -> dict[tuple[str, str, str], dict[str, int | None]]:
    """(dataset, instance_id, skill_id) -> {"annotator_N": rating}, from round1/."""
    out: dict[tuple[str, str, str], dict[str, int | None]] = {}
    for subset in sorted(ROUND1_DIR.glob("task_subset_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
        annotator = "annotator_" + subset.name.rsplit("_", 1)[1]
        for path in subset.glob("*-annotation.json"):
            dataset = _dataset_name(path)
            for result in json.loads(path.read_text())["results"]:
                for cand in result["retrieved"]:
                    rating = next(iter(cand.get("annotations", {}).values()), None)
                    out.setdefault((dataset, result["instance_id"], cand["skill_id"]), {})[annotator] = rating
    return out


def load_flag_dir(folder: Path, fill_missing: bool) -> dict[tuple[str, str], dict[str, dict[str, bool]]]:
    """(dataset, skill_id) -> {"annotator_N": {flag: bool}}, from each
    folder/task_subset_N/flags.json that exists. fill_missing treats an unset
    flag as False (round 1); otherwise only flags actually saved are kept, so
    round 2 overrides just the flags that were re-judged."""
    out: dict[tuple[str, str], dict[str, dict[str, bool]]] = {}
    for subset in sorted(folder.glob("task_subset_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
        path = subset / "flags.json"
        if not path.is_file():
            continue
        annotator = "annotator_" + subset.name.rsplit("_", 1)[1]
        for dataset, skills in json.loads(path.read_text()).items():
            for skill_id, ann in skills.items():
                values = next(iter(ann.values()), {})
                out.setdefault((dataset, skill_id), {})[annotator] = {
                    k: bool(values.get(k)) for k in FLAG_KEYS if fill_missing or k in values
                }
    return out


ANNOTATIONS = load_annotation_files()
PREVIOUS_RATINGS = load_previous_ratings()
PREVIOUS_FLAGS = load_flag_dir(ROUND1_DIR, fill_missing=True)
ROUND1_ANNOTATORS = sorted({a for flags in PREVIOUS_FLAGS.values() for a in flags}, key=lambda a: int(a.removeprefix("annotator_")))
INSTANCE_TEXT = load_instance_text()
if "skill_usage" in ANNOTATIONS:
    INSTANCE_TEXT["skill_usage"] = load_skill_usage_instance_text()
CORPUS = load_corpus({
    cand["skill_id"]
    for dataset, data in ANNOTATIONS.items() if dataset != "skill_usage"
    for result in data["results"]
    for cand in result["retrieved"]
})

DATASET_ORDER: list[str] = sorted(ANNOTATIONS)
random.Random(REANNOTATION_SEED).shuffle(DATASET_ORDER)


@dataclass
class SkillPosition:
    dataset: str
    result_index: int
    candidate: dict  # reference into ANNOTATIONS -- mutating this mutates the source
    task_ordinal: int
    task_total: int
    skill_ordinal: int
    skill_total: int


def _build_skill_positions() -> list[SkillPosition]:
    rng = random.Random(REANNOTATION_SEED)
    tasks: list[tuple[str, int]] = []
    for dataset in DATASET_ORDER:
        indices = list(range(len(ANNOTATIONS[dataset]["results"])))
        rng.shuffle(indices)
        tasks.extend((dataset, i) for i in indices)

    positions: list[SkillPosition] = []
    for task_ordinal, (dataset, i) in enumerate(tasks, start=1):
        retrieved = ANNOTATIONS[dataset]["results"][i]["retrieved"]
        order = list(range(len(retrieved)))
        rng.shuffle(order)
        for skill_ordinal, idx in enumerate(order, start=1):
            positions.append(SkillPosition(dataset, i, retrieved[idx], task_ordinal, len(tasks), skill_ordinal, len(order)))
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
        "total": len(POSITIONS),
        "rubric": RUBRIC,
    })


@app.get("/skill-usage-file")
def skill_usage_file(kind: str, id: str, path: str):
    target = _resolve_skill_usage_file(kind, id, path)
    return FileResponse(target, filename=target.name, content_disposition_type="inline")


@app.get("/task/{position}")
def show_task(request: Request, position: int, saved: bool = False):
    sp = position_at(position)
    result = ANNOTATIONS[sp.dataset]["results"][sp.result_index]
    instance_id = result["instance_id"]
    skill_id = sp.candidate["skill_id"]
    question = INSTANCE_TEXT.get(sp.dataset, {}).get(instance_id, "(question not found)")
    raw_markdown = skill_markdown(sp.dataset, skill_id, CORPUS)
    is_skill_usage = sp.dataset == "skill_usage"

    round1_flags = PREVIOUS_FLAGS.get((sp.dataset, skill_id), {})
    round2_flags = load_flag_dir(ROUND2_DIR, fill_missing=False).get((sp.dataset, skill_id), {})
    rubric_labels = {value: label for value, label, _ in RUBRIC}
    candidate = {
        "skill_id": skill_id,
        "raw_markdown": raw_markdown,
        "rating": sp.candidate["annotations"].get(ANNOTATOR),
        "translate_url": google_translate_url(raw_markdown),
        "previous_ratings": [
            ("Annotator " + key.removeprefix("annotator_"), value, rubric_labels.get(value, "not rated"))
            for key, value in sorted(
                PREVIOUS_RATINGS.get((sp.dataset, instance_id, skill_id), {}).items(),
                key=lambda kv: int(kv[0].removeprefix("annotator_")),
            )
        ],
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
        "task_files": task_files(instance_id) if is_skill_usage else [],
        "skill_files": skill_files(skill_id) if is_skill_usage else [],
        "candidate": candidate,
        "annotator_labels": ["Annotator " + a.removeprefix("annotator_") for a in ROUND1_ANNOTATORS],
        "flag_rows": [
            (label, [
                (round2_flags[a][key], True) if key in round2_flags.get(a, {})
                else (round1_flags.get(a, {}).get(key), False)
                for a in ROUND1_ANNOTATORS
            ])
            for key, label, _, _ in flag_definitions_for(sp.dataset)
        ],
        "flag_capabilities": [cap for _, _, _, cap in flag_definitions_for(sp.dataset) if cap],
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

    next_position = position + 1 if position < len(POSITIONS) else position
    return RedirectResponse(f"/task/{next_position}?saved=true", status_code=303)
