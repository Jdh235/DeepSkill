"""RQ2 table: how single skills change the outcome on tasks the baseline never or always solves, by grade.

Tasks are those the model solves in none of its 3 no-skill trials (baseline 0/3) or in all 3 (baseline 3/3).
Every trial with one skill injected counts on its own, against the baseline's trials: on a 0/3 task a trial
that succeeds helps (otherwise it ties); on a 3/3 task a trial that fails hurts (otherwise it ties), so Tie is
100 minus Help or Hurt and isn't shown. A trial counts as its reward (Skill-Usage gives partial credit), so a reward
of 0.4 on a 0/3 task is 0.4 of a help. Cells are the share of trials that help (0/3) or hurt (3/3), by final
annotator grade (labels/human/human_labels.py), with the numbers of tasks and trials each model has at that baseline. Pairs
never run (reference skills no pool retriever returned) are left out, and so are cells drawn from fewer than
MIN_TASKS tasks (on Skill-Usage, Qwen3.6-27B solves only one task in all 3 trials).

Writes to figures/output/rq2/:

    table_help_tie_hurt.tex   tab:help-tie-hurt   rows: one per model under each benchmark;
                                                  columns: baseline 0/3 (tasks, trials, Help by grade 0-3),
                                                  baseline 3/3 (tasks, trials, Hurt by grade 0-3)

    python3 figures/scripts/generate_help_tie_hurt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "output" / "rq2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
UTILITY_DIR = REPO_ROOT / "labels" / "utility"

sys.path.insert(0, str(REPO_ROOT / "labels" / "human"))
from human_labels import load_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tex_table import TABLE_TAIL, table_head  # noqa: E402

BENCHMARKS = {"SRA-Bench": "sra_bench", "Skill-Usage": "skill_usage"}
MODELS = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "GLM-4.5-Air-FP8": "GLM-4.5-Air",
          "Qwen3.6-27B-FP8": "Qwen3.6-27B"}  # weakest first
GRADES = [0, 1, 2, 3]
# baseline trials solved -> the outcome shown (Help: the skill trial succeeds; Hurt: it fails)
SIDES = {0: "Help", 3: "Hurt"}
ARROWS = {"Help": r"$\uparrow$", "Hurt": r"$\downarrow$"}  # higher Help is better, lower Hurt is better
MIN_TASKS = 2  # a cell from one task says more about that task than about the grade


def outcomes(folder: str, model: str, grades: dict) -> tuple[dict, dict] | None:
    """(baseline, grade) -> {"trials", "tasks", "share"}, and baseline -> its tasks; None if the model wasn't run."""
    model_dir = UTILITY_DIR / folder / model
    if not model_dir.is_dir():
        return None
    counts = {(b, g): {"trials": 0, "tasks": set(), "share": 0.0} for b in SIDES for g in GRADES}
    base_tasks = {b: set() for b in SIDES}
    for path in sorted(model_dir.glob("*-utility.json")):
        for r in json.loads(path.read_text())["results"]:
            base = round(sum(r["baseline"]["rewards"]), 6)
            if base not in SIDES:
                continue
            base_tasks[base].add(r["instance_id"])
            for c in r["candidates"]:
                if c["rewards"] is None:
                    continue
                cell = counts[(base, grades[(r["instance_id"], c["skill_id"])])]
                cell["tasks"].add(r["instance_id"])
                for reward in c["rewards"]:  # every trial on its own
                    cell["trials"] += 1
                    cell["share"] += reward if base == 0 else 1 - reward
    return counts, base_tasks


def main() -> None:
    grades = {(r["instance_id"], r["skill_id"]): r["final"] for r in load_labels()}
    width = 2 + len(GRADES)  # columns per side: tasks, trials, one per grade
    n_cols = 1 + len(SIDES) * width
    lines, shown = [], []  # shown: (tasks, trials) of every cell with a share
    for b_i, (bench, folder) in enumerate(BENCHMARKS.items()):
        if b_i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{bench}}}}} \\")
        for model, short in MODELS.items():
            result = outcomes(folder, model, grades)
            if result is None:  # model not run on this benchmark
                continue
            counts, base_tasks = result
            cells, log = [], []
            for b, name in SIDES.items():
                cells += [str(len(base_tasks[b])), str(sum(counts[(b, g)]["trials"] for g in GRADES))]
                for g in GRADES:
                    c = counts[(b, g)]
                    tasks = len(c["tasks"])
                    share = f"{100 * c['share'] / c['trials']:.0f}" if c["trials"] else "--"
                    log.append(f"{name[0]}{g} {share} ({tasks}t/{c['trials']})")
                    if tasks < MIN_TASKS:
                        cells.append("--")
                        continue
                    shown.append((tasks, c["trials"]))
                    cells.append(share)
            print(f"{bench:12} {short:13} " + "  ".join(log))
            lines.append(rf"\quad {short} & " + " & ".join(cells) + r" \\")

    print(f"Each share shown is drawn from at least {min(t for t, _ in shown)} tasks "
          f"and {min(n for _, n in shown)} trials")
    (OUT_DIR / "table_help_tie_hurt.tex").write_text("\n".join([
        "% Generated by figures/scripts/generate_help_tie_hurt.py -- needs \\usepackage{booktabs}",
        *table_head("tab:help-tie-hurt"),
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l {' '.join('r' * width for _ in SIDES)}@{{}}}}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{{width}}}{{c}}{{Baseline {b}/3: {name} (\%) {ARROWS[name]}}}"
                           for b, name in SIDES.items()) + r" \\",
        rf"\cmidrule(lr){{2-{1 + width}}} \cmidrule(l){{{2 + width}-{n_cols}}}",
        "Model & " + " & ".join(" & ".join(["Tasks", "$n$", *map(str, GRADES)]) for _ in SIDES) + r" \\",
        r"\midrule", *lines, r"\bottomrule", r"\end{tabular}", TABLE_TAIL,
    ]) + "\n")
    print(f"Saved: {OUT_DIR / 'table_help_tie_hurt'}.tex")


if __name__ == "__main__":
    main()
