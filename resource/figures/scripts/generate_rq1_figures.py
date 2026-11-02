"""RQ1 tables: how complete are each benchmark's reference skills?

Compares the reference skills (the skills each benchmark marks as correct --
what we used to call gold) with our human judgments (graded 0-3, final
label) on the annotated subset. Every other judged skill is an "other skill".
Writes five booktabs tables to figures/output/ -- \\input them in Overleaf (needs
\\usepackage{booktabs}) or copy the rows:

    table1_setup.tex              benchmarks and annotation setup: domain
                                  (hand-filled, see DOMAIN), tasks, whether
                                  tasks have a single or multiple reference
                                  skills, library size (corpus_size in the
                                  annotation files), judgment-pool size per
                                  task (avg/min/max).
    table2_reference_skills.tex   do our judgments confirm the reference
                                  skills? One row per dataset, with column
                                  groups for single- and multi-reference
                                  tasks: tasks, and the share of their
                                  reference skills graded >= 2 and 3.
    table_disagreement.tex        first-round disagreement: share of pairs
                                  whose three grades differ by 0, 1 or >= 2
                                  (the >= 2 pairs were re-annotated).
    table_flags.tex               quality flags, first and second round: skills
                                  flagged by at least one annotator, split by how
                                  many of the three raised each flag (round 2
                                  re-flags the skills with any round-1 flag).
    table_flag_agreement.tex      Cohen's kappa on the not-actionable flag
                                  (SRA-Bench): annotator pairs, LLM judge vs
                                  annotators and vs the majority flag, in round 1
                                  and on the final flags. Also prints the
                                  re-annotation statistics quoted in the text.
    table_reannotation.tex        re-annotated pairs (first-round grades two
                                  or more apart): re-annotated grade below,
                                  equal to or above the first-round median.

Inputs:
    labels/human/human_labels.py   human grades: first-round grades from the
                                  annotation exports, final label (re-annotation,
                                  else median)
    ANNOTATION_DIR  annotation/top5/task_subset/{dataset}-annotation.json
                    (which skills are reference skills)

    .venv/bin/python figures/scripts/generate_rq1_figures.py   (needs scikit-learn)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "output"  # everything the paper includes
OUT_DIR.mkdir(exist_ok=True)
ANNOTATION_DIR = REPO_ROOT / "annotation" / "top5" / "task_subset"
sys.path.insert(0, str(REPO_ROOT / "labels" / "human"))
from human_labels import (  # noqa: E402
    ANNOTATORS as ASSESSORS, FLAGS, FLAGS_ROUND2, load_flags, load_labels, lower_median, reannotated_pairs,
)
sys.path.insert(0, str(Path(__file__).parent))
from tex_table import TABLE_TAIL, table_head  # noqa: E402

DATASETS = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa", "skill_usage"]
DISPLAY = {
    "bigcodebench": "BigCodeBench", "champ": "CHAMP", "logicbench": "LogicBench",
    "medcalcbench": "MedCalcBench", "theoremqa": "TheoremQA", "skill_usage": "Skill-Usage",
}
# Table 1 domain column -- not in the data, set by hand.
DOMAIN = {
    "bigcodebench": "Code", "champ": "Math", "logicbench": "Logic",
    "medcalcbench": "Medical", "theoremqa": "STEM", "skill_usage": "Mixed",
}


def load_reference() -> set[tuple[str, str, str]]:
    """(dataset, instance_id, skill_id) for every reference skill."""
    reference = set()
    for ds in DATASETS:
        for r in json.loads((ANNOTATION_DIR / f"{ds}-annotation.json").read_text())["results"]:
            reference |= {(ds, r["instance_id"], s) for s in r["gold_skill_ids"]}
    return reference


def load_pairs() -> dict[str, list[dict]]:
    reference = load_reference()
    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in load_labels():
        key = (row["dataset"], row["instance_id"], row["skill_id"])
        pairs[row["dataset"]].append({
            "task": row["instance_id"], "reference": key in reference, "grade": row["final"],
            # first-round rating of each assessor (None if they skipped the pair)
            "assessors": [row[a] for a in ASSESSORS],
        })
    return pairs


def ordinal_alpha(units: list[list[int | None]]) -> float:
    """Krippendorff's alpha, ordinal metric, over units (one list of assessor
    ratings per task-skill pair, None = missing). Same definition as the
    `krippendorff` package (checked to 4 decimal places)."""
    o: dict[tuple[int, int], float] = defaultdict(float)  # coincidence matrix
    for ratings in units:
        vals = [v for v in ratings if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, a in enumerate(vals):
            for j, b in enumerate(vals):
                if i != j:
                    o[(a, b)] += 1 / (m - 1)
    cats = sorted({c for c, _ in o})
    n_c = {c: sum(o[(c, k)] for k in cats) for c in cats}
    n = sum(n_c.values())

    def delta2(c: int, k: int) -> float:  # ordinal distance
        lo, hi = min(c, k), max(c, k)
        return (sum(n_c[g] for g in cats if lo <= g <= hi) - (n_c[c] + n_c[k]) / 2) ** 2

    observed = sum(o[(c, k)] * delta2(c, k) for c in cats for k in cats)
    expected = sum(n_c[c] * n_c[k] * delta2(c, k) for c in cats for k in cats) / (n - 1)
    return 1 - observed / expected


MIDRULE = r"\midrule"


def write_table(name: str, label: str, colspec: str, header: list[str],
                rows: list[tuple[str, list[str]] | str], tabcolsep: str = "5pt") -> None:
    """rows: (row label, cells) in display order; a plain string is written as-is
    (a rule, \\addlinespace, or a full group-heading row)."""
    lines = [
        r"% Generated by figures/scripts/generate_rq1_figures.py -- needs \usepackage{booktabs}",
        *table_head(label),
        r"\small",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header,
        r"\midrule",
    ]
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            row_label, cells = row
            lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", TABLE_TAIL]
    (OUT_DIR / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT_DIR / name}.tex")


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}"


def table2_reference_skills(pairs: dict[str, list[dict]]) -> None:
    """Human grades of reference skills, one row per dataset: all tasks first,
    then split into column groups by task type (Single: one reference skill;
    Multi: two or more). A dataset without tasks of a type shows -- there."""
    def cells(P: list[dict]) -> list[str]:
        n_reference = defaultdict(int)  # task ids are dataset-prefixed, so pooling datasets is safe
        for p in P:
            n_reference[p["task"]] += p["reference"]
        ref = [p for p in P if p["reference"]]  # all tasks, both types together
        out = [str(len(n_reference)), pct(sum(p["grade"] >= 2 for p in ref), len(ref)),
               pct(sum(p["grade"] == 3 for p in ref), len(ref))]
        for single in (True, False):
            tasks = [t for t, n in n_reference.items() if (n == 1) == single]
            ref = [p for p in P if p["reference"] and (n_reference[p["task"]] == 1) == single]
            if not tasks:
                out += ["--"] * 3
                continue
            out += [str(len(tasks)), pct(sum(p["grade"] >= 2 for p in ref), len(ref)),
                    pct(sum(p["grade"] == 3 for p in ref), len(ref))]
        return out

    sra_datasets = [ds for ds in DATASETS if ds != "skill_usage"]
    width = max(len(rf"\quad {DISPLAY[ds]}") for ds in sra_datasets)  # aligns the source, not the PDF
    sra = [p for ds in sra_datasets for p in pairs[ds]]
    rows = [r"\multicolumn{10}{@{}l}{\textit{SRA-Bench}} \\"]
    rows += [(rf"\quad {DISPLAY[ds]}".ljust(width), cells(pairs[ds])) for ds in sra_datasets]
    rows += [r"\addlinespace[2pt]", (r"\quad \textit{All}", cells(sra)), MIDRULE]
    rows += [(r"\textit{Skill-Usage}", cells(pairs["skill_usage"]))]
    write_table(
        "table2_reference_skills",
        "tab:rq1-reference",
        "@{}l r r r r r r r r r@{}",
        [r" & \multicolumn{3}{c}{All tasks} & \multicolumn{3}{c}{Single} & \multicolumn{3}{c}{Multi} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(l){8-10}",
         r"Dataset & Tasks & $\geq$2 & 3 & Tasks & $\geq$2 & 3 & Tasks & $\geq$2 & 3 \\"],
        rows,
        tabcolsep="6pt",
    )


def table1_setup(pairs: dict[str, list[dict]]) -> None:
    """Benchmarks and annotation setup: domain, tasks, whether tasks have a
    single or multiple reference skills, library size (skills available to
    retrieve from) and judgment-pool size per task. Agreement (alpha) is in
    figures/output/table_agreement_panel.tex."""
    metadata = {ds: json.loads((ANNOTATION_DIR / f"{ds}-annotation.json").read_text())["metadata"]
                for ds in DATASETS}

    def cells(P: list[dict], domain: str, library: int | None, show_type: bool = True) -> list[str]:
        depth, n_reference = defaultdict(int), defaultdict(int)
        for p in P:
            depth[p["task"]] += 1
            n_reference[p["task"]] += p["reference"]
        kinds = {"Single" if n == 1 else "Multi" for n in n_reference.values()}
        task_type = (kinds.pop() if len(kinds) == 1 else "Mixed") if show_type else ""
        return [
            domain, str(len(depth)), task_type, f"{library:,}" if library else "",
            f"{len(P) / len(depth):.1f}", str(min(depth.values())), str(max(depth.values())),
        ]

    sra_datasets = [ds for ds in DATASETS if ds != "skill_usage"]
    sra_library = {metadata[ds]["corpus_size"] for ds in sra_datasets}
    assert len(sra_library) == 1, f"SRA-Bench datasets should share one skill library, got {sra_library}"
    width = max(len(rf"\quad {DISPLAY[ds]}") for ds in sra_datasets)
    sra = [p for ds in sra_datasets for p in pairs[ds]]
    rows = [r"\multicolumn{8}{@{}l}{\textit{SRA-Bench}} \\"]
    # The SRA-Bench datasets share one library, so its size is shown once, on the "All" row.
    rows += [(rf"\quad {DISPLAY[ds]}".ljust(width), cells(pairs[ds], DOMAIN[ds], None)) for ds in sra_datasets]
    rows += [r"\addlinespace[2pt]",
             (r"\quad \textit{All}", cells(sra, "", sra_library.pop(), show_type=False)), MIDRULE]
    rows += [(r"\textit{Skill-Usage}", cells(pairs["skill_usage"], DOMAIN["skill_usage"],
                                             metadata["skill_usage"]["corpus_size"]))]
    write_table(
        "table1_setup",
        "tab:setup",
        "@{}l l r l r r r r@{}",
        [r" & & & & & \multicolumn{3}{c}{Pool per task} \\",
         r"\cmidrule(l){6-8}",
         r"\multirow{-2}{*}{Dataset} & \multirow{-2}{*}{Domain} & \multirow{-2}{*}{Tasks} & "
         r"\multirow{-2}{*}{Ref. skills} & \multirow{-2}{*}{Library} & Avg & Min & Max \\"],
        rows,
        tabcolsep="4.5pt",
    )


def table_disagreement() -> None:
    """First-round disagreement: how far apart the three annotators' grades are (highest - lowest), before
    re-annotation. Uses the first-round grades only; spread >= 2 is the set that was re-annotated."""
    spreads: dict[str, list[int]] = defaultdict(list)
    wide = set()
    for r in load_labels():
        first = [r[a] for a in ASSESSORS]
        if None in first:
            continue
        spreads[r["dataset"]].append(max(first) - min(first))
        if max(first) - min(first) >= 2:
            wide.add((r["dataset"], r["instance_id"], r["skill_id"]))
    assert wide == reannotated_pairs(), "re-annotated set is not the spread >= 2 set"

    def cells(values: list[int]) -> list[str]:
        n = len(values)
        counts = [sum(v == 0 for v in values), sum(v == 1 for v in values), sum(v >= 2 for v in values)]
        return [f"{n:,}"] + [f"{100 * c / n:.0f}\\,{{\\scriptsize({c:,})}}" for c in counts]

    sra_datasets = [ds for ds in DATASETS if ds != "skill_usage"]
    width = max(len(rf"\quad {DISPLAY[ds]}") for ds in sra_datasets)
    rows = [r"\multicolumn{5}{@{}l}{\textit{SRA-Bench}} \\"]
    rows += [(rf"\quad {DISPLAY[ds]}".ljust(width), cells(spreads[ds])) for ds in sra_datasets]
    rows += [r"\addlinespace[2pt]",
             (r"\quad \textit{All}", cells([v for ds in sra_datasets for v in spreads[ds]])), MIDRULE]
    rows += [(r"\textit{Skill-Usage}", cells(spreads["skill_usage"]))]
    write_table(
        "table_disagreement",
        "tab:disagreement",
        "@{}l r rrr@{}",
        [r" & & \multicolumn{3}{c}{Grade spread} \\",
         r"\cmidrule(l){3-5}",
         r"Dataset & Pairs & 0 & 1 & $\geq$2 \\"],
        rows,
    )


FLAG_NAMES = {"not_actionable": "Not actionable", "incorrect_info": "Incorrect information", "malicious": "Malicious"}
JUDGE_DIR = REPO_ROOT / "labels" / "llm_judge" / "gpt-oss-120b"  # the judge reported in the paper


def judge_flags() -> dict[tuple[str, str], list[dict[str, bool]]]:
    """{(dataset, skill_id): [the LLM judge's flags for each task-skill pair it rated]}. Unlike the annotators,
    the judge flags every pair, so a skill can be flagged for one task and not another."""
    out: dict[tuple[str, str], list[dict[str, bool]]] = defaultdict(list)
    for path in sorted(JUDGE_DIR.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                if j.get("rating") is not None:
                    out[(j["dataset"], j["skill_id"])].append({f: bool(j.get(f)) for f in FLAGS})
    return out


def table_flags() -> None:
    """Quality flags in both rounds, and from the LLM judge. Annotators flag a skill once per dataset, so the unit
    is a skill (not a task-skill pair). Round 1 covers every skill; in round 2 all three annotators flag again the
    skills that received any round-1 flag. Per flag and round: skills flagged by at least one annotator, split by
    how many of the three raised it. The judge flags each task-skill pair, so a skill counts as judge-flagged if
    any of its pairs is flagged (over the round-1 skills). "Any flag" counts a skill that received at least one of
    the three flags."""
    rounds = [load_flags(), load_flags(FLAGS_ROUND2)]
    round1_flagged = {k for k, v in rounds[0].items() if any(any(fl.values()) for fl in v)}
    assert set(rounds[1]) == round1_flagged, "round 2 is not the set of skills with a round-1 flag"
    judge = judge_flags()
    assert set(rounds[0]) <= set(judge), "the LLM judge did not rate every annotated skill"
    kinds = [*((FLAG_NAMES[f], lambda fl, f=f: fl[f]) for f in FLAG_NAMES), ("Any flag", lambda fl: any(fl.values()))]
    benchmarks = {"SRA-Bench": [ds for ds in DATASETS if ds != "skill_usage"], "Skill-Usage": ["skill_usage"]}
    rows = []
    for b, dsl in benchmarks.items():
        skills = [[v for (ds, _), v in flags.items() if ds in dsl] for flags in rounds]
        judged = [judge[k] for k in rounds[0] if k[0] in dsl]  # the round-1 skills, as the judge flagged them
        if rows:
            rows.append(MIDRULE)
        rows.append(f"{b} & Skills annotated & {len(skills[0]):,} & & & & {len(skills[1]):,} & & & & {len(judged):,} \\\\")
        for name, raised in kinds:
            cells = []
            for round_skills in skills:
                by = [sum(raised(fl) for fl in v) for v in round_skills]  # how many annotators raised it, per skill
                n = [sum(k == j for k in by) for j in (1, 2, 3)]
                cells += [str(sum(n)), *map(str, n)]
            cells.append(str(sum(any(raised(fl) for fl in v) for v in judged)))
            if name == "Any flag":
                rows.append(r"\addlinespace[2pt]")
            rows.append(f" & {name} & " + " & ".join(cells) + r" \\")
    write_table(
        "table_flags",
        "tab:flags",
        "@{}l l r rrr r rrr r@{}",
        [r" & & \multicolumn{4}{c}{First round} & \multicolumn{4}{c}{Second round} & LLM \\",
         r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
         r" & & & \multicolumn{3}{c}{Flagged by} & & \multicolumn{3}{c}{Flagged by} & judge \\",
         r"\cmidrule(lr){4-6} \cmidrule(lr){8-10}",
         r"Benchmark & Flag & Flagged & 1 & 2 & 3 & Flagged & 1 & 2 & 3 & Flagged \\"],
        rows,
        tabcolsep="4pt",
    )


def kappa_fmt(x: float) -> str:
    """Two decimals, no leading zero, LaTeX minus."""
    return f"{'$-$' if x < 0 else ''}{abs(x):.2f}".replace("0.", ".", 1)


def table_flag_agreement() -> None:
    """Agreement on the not-actionable flag on SRA-Bench, the only flag frequent enough for Cohen's kappa
    (malicious: at most 5 skills; incorrect information: never used by an annotator; Skill-Usage annotators flag
    almost nothing). Two like-for-like comparisons: round 1, where every judgment is independent, and the final
    flags (each annotator's round-2 answer where the skill was re-examined, else round 1), where annotators saw
    each other's round-1 answers. The LLM judge counts a skill as flagged if any of its task-skill pairs is
    (as in table_flags). Also prints the re-annotation statistics quoted in the text."""
    # imported here, not at the top: other scripts import this module's helpers under the system python3
    from sklearn.metrics import cohen_kappa_score

    rounds = [load_flags(), load_flags(FLAGS_ROUND2)]
    final = {k: rounds[1].get(k, v) for k, v in rounds[0].items()}
    judge = judge_flags()
    benchmarks = {"SRA-Bench": [ds for ds in DATASETS if ds != "skill_usage"], "Skill-Usage": ["skill_usage"]}

    for b, dsl in benchmarks.items():  # re-annotation statistics for the text
        keys = [k for k in rounds[1] if k[0] in dsl]
        unanimous = sum(all(len({a[f] for a in rounds[1][k]}) == 1 for f in FLAGS) for k in keys)
        print(f"{b}: {len(keys)} re-examined skills, all three agree on every flag for {unanimous} "
              f"({100 * unanimous / len(keys):.0f}%)")
        for f in FLAGS:
            by = [sum(a[f] for a in rounds[1][k]) for k in keys]
            print(f"  {f}: round-2 majority {sum(n >= 2 for n in by)}, unanimous {sum(n == 3 for n in by)}")

    keys = [k for k in rounds[0] if k[0] in benchmarks["SRA-Bench"]]
    judged = [int(any(fl["not_actionable"] for fl in judge[k])) for k in keys]
    rows = {"Annotator vs annotator": [], "LLM judge vs annotator": [], "LLM judge vs majority flag": []}
    for flags in (rounds[0], final):
        ann = [[int(flags[k][a]["not_actionable"]) for k in keys] for a in range(len(ASSESSORS))]
        majority = [int(sum(col) >= 2) for col in zip(*ann)]
        pairs = [cohen_kappa_score(ann[a], ann[b]) for a, b in combinations(range(len(ASSESSORS)), 2)]
        vs_judge = [cohen_kappa_score(a, judged) for a in ann]
        rows["Annotator vs annotator"].append(f"{kappa_fmt(min(pairs))}--{kappa_fmt(max(pairs))}")
        rows["LLM judge vs annotator"].append(f"{kappa_fmt(min(vs_judge))}--{kappa_fmt(max(vs_judge))}")
        rows["LLM judge vs majority flag"].append(kappa_fmt(cohen_kappa_score(majority, judged)))
        print("SRA-Bench not actionable kappa, annotator pairs " + " ".join(f"{x:.2f}" for x in pairs)
              + " | judge vs annotators " + " ".join(f"{x:.2f}" for x in vs_judge))
    write_table(
        "table_flag_agreement",
        "tab:flag-agreement",
        "@{}l rr@{}",
        [r"Comparison & Round 1 & Final \\"],
        [(name, cells) for name, cells in rows.items()],
    )


def table_reannotation() -> None:
    """Re-annotated pairs (first-round grades two or more apart): was the re-annotated grade below, equal to or
    above the median of the three first-round grades?"""
    rows_by_ds: dict[str, list[tuple[int, int]]] = defaultdict(list)  # dataset -> [(median, re-annotated grade)]
    for r in load_labels(reannotation=True):  # always: this table is about the re-annotation
        if r["reannotation"] is not None:
            first = [r[a] for a in ASSESSORS if r[a] is not None]
            rows_by_ds[r["dataset"]].append((lower_median(first), r["reannotation"]))

    def cells(pairs: list[tuple[int, int]]) -> list[str]:
        n = len(pairs)
        counts = [sum(new < med for med, new in pairs), sum(new == med for med, new in pairs),
                  sum(new > med for med, new in pairs)]
        return [str(n)] + [f"{100 * c / n:.0f}\\,{{\\scriptsize({c})}}" for c in counts]

    sra_datasets = [ds for ds in DATASETS if ds != "skill_usage"]
    width = max(len(rf"\quad {DISPLAY[ds]}") for ds in sra_datasets)
    rows = [r"\multicolumn{5}{@{}l}{\textit{SRA-Bench}} \\"]
    rows += [(rf"\quad {DISPLAY[ds]}".ljust(width), cells(rows_by_ds[ds])) for ds in sra_datasets]
    rows += [r"\addlinespace[2pt]",
             (r"\quad \textit{All}", cells([x for ds in sra_datasets for x in rows_by_ds[ds]])), MIDRULE]
    rows += [(r"\textit{Skill-Usage}", cells(rows_by_ds["skill_usage"]))]
    write_table(
        "table_reannotation",
        "tab:reannotation",
        "@{}l r rrr@{}",
        [r" & & \multicolumn{3}{c}{vs.\ first-round median} \\",
         r"\cmidrule(l){3-5}",
         r"Dataset & Pairs & Below & Equal & Above \\"],
        rows,
    )


def main() -> None:
    pairs = load_pairs()
    table1_setup(pairs)
    table2_reference_skills(pairs)
    table_disagreement()
    table_flags()
    table_flag_agreement()
    table_reannotation()


if __name__ == "__main__":
    main()
