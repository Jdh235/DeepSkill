from __future__ import annotations

import json
import random
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from common import (
    BASE_DIR as REPO_ROOT,
    DATASET_DESCRIPTIONS,
    FLAG_DEFINITIONS,
    FLAG_KEYS,
    _resolve_skill_usage_file,
    dataset_display_name,
    flag_definitions_for,
    google_translate_url,
    load_corpus,
    load_seed,
    skill_files,
    skill_markdown,
)

HERE = Path(__file__).resolve().parent
UNITS_PATH = HERE / "data" / "units.json"
FLAGS_PATH = HERE / "data" / "flags.json"
ROUND1_DIR = HERE.parent / "round1" 

if not UNITS_PATH.is_file():
    raise RuntimeError(f"{UNITS_PATH} not found. Run `python3 reannotation/flags/build_set.py` first.")

ANNOTATOR = "annotator_1"
REANNOTATION_SEED = load_seed(HERE / "annotation_seed.txt", "flags")

app = FastAPI()
app.mount("/static", StaticFiles(directory=REPO_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def load_previous_flags() -> dict[tuple[str, str], dict[str, dict[str, bool]]]:
    """(dataset, skill_id) -> {"annotator_N": {flag: bool}}, from round1/."""
    out: dict[tuple[str, str], dict[str, dict[str, bool]]] = {}
    for subset in sorted(ROUND1_DIR.glob("task_subset_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])):
        annotator = "annotator_" + subset.name.rsplit("_", 1)[1]
        for dataset, skills in json.loads((subset / "flags.json").read_text()).items():
            for skill_id, ann in skills.items():
                values = next(iter(ann.values()), {})
                out.setdefault((dataset, skill_id), {})[annotator] = {k: bool(values.get(k)) for k in FLAG_KEYS}
    return out


def save_flags() -> None:
    FLAGS_PATH.write_text(json.dumps(FLAGS, ensure_ascii=False, indent=2) + "\n")


def get_new_flag(unit: dict) -> bool | None:
    """This unit's re-judged flag, or None if not re-judged yet."""
    return FLAGS.get(unit["dataset"], {}).get(unit["skill_id"], {}).get(ANNOTATOR, {}).get(unit["flag"])


UNITS: list[dict] = json.loads(UNITS_PATH.read_text())
FLAGS: dict = json.loads(FLAGS_PATH.read_text()) if FLAGS_PATH.exists() else {}
PREVIOUS_FLAGS = load_previous_flags()
DISPUTED: dict[tuple[str, str], set[str]] = {}  # (dataset, skill_id) -> its disputed flags
for _u in UNITS:
    DISPUTED.setdefault((_u["dataset"], _u["skill_id"]), set()).add(_u["flag"])
CORPUS = load_corpus({u["skill_id"] for u in UNITS if u["dataset"] != "skill_usage"})

DATASET_ORDER: list[str] = sorted({u["dataset"] for u in UNITS})
random.Random(REANNOTATION_SEED).shuffle(DATASET_ORDER)


def _build_order() -> list[dict]:
    """Datasets (DATASET_ORDER) -> flag types within each dataset -> skills
    within each flag type, every level shuffled from the seed."""
    rng = random.Random(REANNOTATION_SEED)
    order = []
    for dataset in DATASET_ORDER:
        flag_types = sorted({u["flag"] for u in UNITS if u["dataset"] == dataset})
        rng.shuffle(flag_types)
        for flag in flag_types:
            group = [u for u in UNITS if u["dataset"] == dataset and u["flag"] == flag]
            rng.shuffle(group)
            order.extend(group)
    return order


ORDER: list[dict] = _build_order()
DATASET_START: dict[str, int] = {}
for _i, _u in enumerate(ORDER, start=1):
    DATASET_START.setdefault(_u["dataset"], _i)


def unit_at(position: int) -> dict:
    if position < 1 or position > len(ORDER):
        raise HTTPException(status_code=404)
    return ORDER[position - 1]


def _start_position() -> int:
    """First not-yet-re-judged flag, or 1 if everything's done."""
    for position, unit in enumerate(ORDER, start=1):
        if get_new_flag(unit) is None:
            return position
    return 1


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("intro.html", {
        "request": request,
        "start_position": _start_position(),
        "total": len(ORDER),
        "n_skills": len({(u["dataset"], u["skill_id"]) for u in ORDER}),
        "flag_definitions": flag_definitions_for(""),
    })


@app.get("/skill-usage-file")
def skill_usage_file(kind: str, id: str, path: str):
    target = _resolve_skill_usage_file(kind, id, path)
    return FileResponse(target, filename=target.name, content_disposition_type="inline")


@app.get("/skill/{position}")
def show_skill(request: Request, position: int, saved: bool = False):
    unit = unit_at(position)
    dataset, skill_id, flag = unit["dataset"], unit["skill_id"], unit["flag"]
    raw_markdown = skill_markdown(dataset, skill_id, CORPUS)
    previous = PREVIOUS_FLAGS.get((dataset, skill_id), {})
    annotators = sorted(previous, key=lambda a: int(a.removeprefix("annotator_")))

    done = sum(1 for u in ORDER if get_new_flag(u) is not None)

    return templates.TemplateResponse("skill.html", {
        "request": request,
        "dataset": dataset,
        "dataset_label": dataset_display_name(dataset),
        "dataset_description": DATASET_DESCRIPTIONS.get(dataset, ""),
        "datasets": [(d, dataset_display_name(d), DATASET_START[d]) for d in DATASET_ORDER],
        "skill_id": skill_id,
        "raw_markdown": raw_markdown,
        "translate_url": google_translate_url(raw_markdown),
        "skill_files": skill_files(skill_id) if dataset == "skill_usage" else [],
        "flag_definitions": [
            (key, label, description, dataset_capability)
            for key, label, description, _ in flag_definitions_for(dataset)
            if key == flag
            for dataset_capability in [next((cap for *_, cap in flag_definitions_for(dataset) if cap), "")]
        ],
        "flags": {flag: get_new_flag(unit)},
        "annotator_labels": ["Annotator " + a.removeprefix("annotator_") for a in annotators],
        "previous_rows": [
            (label, [previous[a][key] for a in annotators])
            for key, label, _ in FLAG_DEFINITIONS
            if key == flag
        ],
        "position": position,
        "total": len(ORDER),
        "done": done,
        "prev_position": position - 1 if position > 1 else None,
        "next_position": position + 1 if position < len(ORDER) else None,
        "saved": saved,
    })


@app.post("/skill/{position}")
async def save_skill(request: Request, position: int):
    unit = unit_at(position)
    form = await request.form()
    answer = form.get(unit["flag"])
    if answer not in ("yes", "no"):  # no default: an explicit Yes or No is required
        raise HTTPException(status_code=400, detail="Choose Yes or No before saving.")
    key = (unit["dataset"], unit["skill_id"])
    saved_flags = FLAGS.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(ANNOTATOR, {})
    saved_flags[unit["flag"]] = answer == "yes"
    for flag in FLAG_KEYS:
        if flag not in DISPUTED[key]:
            saved_flags[flag] = next(iter(PREVIOUS_FLAGS[key].values()))[flag]
    FLAGS[key[0]][key[1]][ANNOTATOR] = {flag: saved_flags[flag] for flag in FLAG_KEYS if flag in saved_flags}
    save_flags()
    next_position = position + 1 if position < len(ORDER) else position
    return RedirectResponse(f"/skill/{next_position}?saved=true", status_code=303)
