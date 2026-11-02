from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parent
FIRST_ROUND = LABELS_DIR / "first_round"
REANNOTATION = LABELS_DIR / "reannotation"
FLAGS_ROUND2 = REANNOTATION / "flags" / "round2"
FINAL_DIR = LABELS_DIR / "final"  # generated final labels, one subfolder per format:
FINAL_QRELS_DIR = FINAL_DIR / "qrels"  # standard TREC qrels, for evaluation tools
FINAL_FLAGS_DIR = FINAL_DIR / "flags"  # consensus flags per pair, with the relevance grade, for analysis
ANNOTATORS = ["annotator_1", "annotator_2", "annotator_3"]  # annotator N = first_round/task_subset_N
FLAGS = ["malicious", "incorrect_info", "not_actionable"]
USE_REANNOTATION = os.environ.get("LABELS_REANNOTATION", "1") != "0"

Key = tuple[str, str, str]


def read_grades(folder: Path) -> dict[Key, int | None]:
    """Grades in one export folder, in file order. Each export holds one annotator's grade per pair."""
    grades: dict[Key, int | None] = {}
    for path in sorted(folder.glob("*-annotation.json")):
        dataset = path.name.removesuffix("-annotation.json")
        for result in json.loads(path.read_text())["results"]:
            for cand in result["retrieved"]:
                grade = next(iter((cand.get("annotations") or {}).values()), None)
                grades[(dataset, result["instance_id"], cand["skill_id"])] = grade
    return grades


def lower_median(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


@lru_cache(maxsize=None)
def load_labels(reannotation: bool = USE_REANNOTATION) -> list[dict]:
    """One dict per judged task-skill pair: dataset, instance_id, skill_id, annotator_1..3 (round-1 grades,
    None if skipped), reannotation (round-2 grade or None) and final."""
    rounds = [read_grades(FIRST_ROUND / f"task_subset_{n}") for n in range(1, len(ANNOTATORS) + 1)]
    pairs = list(rounds[0])
    for name, grades in zip(ANNOTATORS, rounds):
        if set(grades) != set(pairs):
            raise SystemExit(f"{name} doesn't cover the same task-skill pairs as annotator_1.")
    second = read_grades(REANNOTATION) if reannotation else {}
    if reannotation and any(g is None for g in second.values()):
        raise SystemExit("Some re-annotated pairs have no grade yet.")

    rows = []
    for key in pairs:
        grades = [r[key] for r in rounds]
        rated = [g for g in grades if g is not None]
        if not rated:
            continue 
        new = second.get(key)
        dataset, instance_id, skill_id = key
        rows.append({"dataset": dataset, "instance_id": instance_id, "skill_id": skill_id,
                     **dict(zip(ANNOTATORS, grades)), "reannotation": new,
                     "final": new if new is not None else lower_median(rated)})
    return rows


def reannotated_pairs() -> set[Key]:
    """The pairs sent for re-annotation (whether or not LABELS_REANNOTATION is on)."""
    return set(read_grades(REANNOTATION))


def load_flags(round_dir: Path = FIRST_ROUND) -> dict[tuple[str, str], list[dict[str, bool]]]:
    """{(dataset, skill_id): [flags of annotator_1, _2, _3]}, for skills all annotators saw. round_dir is
    FIRST_ROUND (round 1, every skill) or FLAGS_ROUND2 (round 2: the skills with any round-1 flag).
    Each flags.json is keyed by its annotator inside the file; the folder decides who it is."""
    per_annotator = []
    for n in range(1, len(ANNOTATORS) + 1):
        data = json.loads((round_dir / f"task_subset_{n}" / "flags.json").read_text())
        per_annotator.append({(ds, skill): {f: bool(next(iter(by.values())).get(f)) for f in FLAGS}
                              for ds, skills in data.items() for skill, by in skills.items()})
    shared = set.intersection(*(set(a) for a in per_annotator))
    return {k: [a[k] for a in per_annotator] for k in shared}


def consensus_flags() -> dict[tuple[str, str], dict[str, bool]]:
    """{(dataset, skill_id): {flag: final decision}}. Only flags that at least one annotator marked in round 1
    were re-annotated (by all three); their final decision is the majority (2 of 3) of the round-2 answers.
    A flag no annotator marked in round 1 is final as not flagged."""
    first, second = load_flags(), load_flags(FLAGS_ROUND2)
    out = {}
    for key, round1 in first.items():
        marked = [f for f in FLAGS if any(a[f] for a in round1)]
        if marked and key not in second:
            raise SystemExit(f"{key} was flagged in round 1 but has no round-2 flags.")
        out[key] = {f: f in marked and sum(a[f] for a in second[key]) >= 2 for f in FLAGS}
    return out


def write_flags() -> None:
    """final/flags/{dataset}-flags.tsv: one row per judged task-skill pair, in the qrels' order, with the final
    relevance grade and the skill's consensus flags (1 = flagged). Flags are given once per skill, so a skill's
    flags repeat on each of its pairs."""
    rows, flags = load_labels(), consensus_flags()
    FINAL_FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        pairs = [r for r in rows if r["dataset"] == dataset]
        lines = ["query_id\tskill_id\trelevance\t" + "\t".join(FLAGS)]
        lines += [f"{r['instance_id']}\t{r['skill_id']}\t{r['final']}\t"
                  + "\t".join(str(int(flags[(dataset, r['skill_id'])][f])) for f in FLAGS) for r in pairs]
        (FINAL_FLAGS_DIR / f"{dataset}-flags.tsv").write_text("\n".join(lines) + "\n")
        skills = {r["skill_id"] for r in pairs}
        counts = ", ".join(f"{f} {sum(flags[(dataset, s)][f] for s in skills)}" for f in FLAGS)
        print(f"{dataset}: {len(pairs)} pairs, {len(skills)} skills, flagged skills: {counts}")
    print(f"Wrote relevance + consensus flags to {FINAL_FLAGS_DIR}")


def qrels(dataset: str, label: str = "final", reannotation: bool = USE_REANNOTATION) -> dict[str, dict[str, int]]:
    """{instance_id: {skill_id: grade}} for one dataset, under `label`: "final" or one of ANNOTATORS
    (that annotator's round-1 grades). Pairs without a grade are left out (unjudged)."""
    out: dict[str, dict[str, int]] = {}
    for row in load_labels(reannotation):
        if row["dataset"] == dataset and row[label] is not None:
            out.setdefault(row["instance_id"], {})[row["skill_id"]] = row[label]
    return out


def write_qrels() -> None:
    """final/qrels/{dataset}-relevance.qrels with the final label, in TREC format (query_id 0 doc_id relevance)."""
    rows = load_labels()
    FINAL_QRELS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        lines = [f"{r['instance_id']} 0 {r['skill_id']} {r['final']}" for r in rows if r["dataset"] == dataset]
        (FINAL_QRELS_DIR / f"{dataset}-relevance.qrels").write_text("\n".join(lines) + "\n")
        n_re = sum(r["dataset"] == dataset and r["reannotation"] is not None for r in rows)
        print(f"{dataset}: {len(lines)} pairs, {n_re} from re-annotation")
    print(f"Wrote final-label qrels to {FINAL_QRELS_DIR}" + ("" if USE_REANNOTATION else " (re-annotation ignored)"))
