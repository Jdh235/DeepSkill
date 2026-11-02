"""Retriever table: how the 12 pool retrievers rank with the benchmarks' reference labels and with our human
relevance grades, kept as separate label sources.

Per benchmark (never pooled), nDCG@5 and rank of each retriever under three conditions:

    Ref. all       reference labels (binary) on every benchmark task with a reference skill
    Ref. sample    reference labels on the annotated sample (annotation/tasks.txt)
    Human sample   our final human grades (0-3, labels/human/human_labels.py) on the sample

Ref. all vs Ref. sample isolates the effect of sampling; Ref. sample vs Human sample, on the same tasks,
isolates the effect of the labels. Kendall's tau_b compares each ranking with the one to its left. Runs are
the released top-50 runs in results/.

Writes to figures/output/:

    table_retriever_ranking.tex   tab:retriever-ranking   rows: retrievers (table order of
                                                          RETRIEVERS); columns: the three
                                                          conditions per benchmark; then Kendall's tau

    python3 figures/scripts/generate_retriever_ranking.py   (needs ir_measures)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import ir_measures
from ir_measures import nDCG

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "output"  # everything the paper includes
OUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "labels" / "human"))
from human_labels import load_labels  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from tex_table import TABLE_TAIL, table_head  # noqa: E402

DEPTH = 5
CONDITIONS = ["Ref. all", "Ref. sample", "Human sample"]

SRA = ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]
BENCHMARKS = {"SRA-Bench": SRA, "Skill-Usage": ["skill_usage"]}

# Pool retrievers, in table order: display name, type, model (Hugging Face id or PyTerrier weighting model).
RETRIEVERS = {
    "pt_bm25": ("BM25", "Lexical", "PyTerrier BM25"),
    "pt_hiemstra_lm": ("Hiemstra LM", "Lexical", "PyTerrier Hiemstra_LM"),
    "pt_splade": ("SPLADE", "Learned sparse", "naver/splade-cocondenser-ensembledistil"),
    "dense_bge": ("BGE-base", "Dense", "BAAI/bge-base-en-v1.5"),
    "dense_contriever": ("Contriever", "Dense", "facebook/contriever-msmarco"),
    "dense_bge_m3": ("BGE-M3", "Dense", "BAAI/bge-m3"),
    "dense_colbertv2": ("ColBERTv2", "Late interaction", "colbert-ir/colbertv2.0"),
    "dense_qwen3_emb_06b": ("Qwen3-Emb-0.6B", "LLM embedder", "Qwen/Qwen3-Embedding-0.6B"),
    "dense_qwen3_emb_8b": ("Qwen3-Emb-8B", "LLM embedder", "Qwen/Qwen3-Embedding-8B"),
    "dense_skillret_06b": ("SkillRet-0.6B", "Skill fine-tuned", "ThakiCloud/SKILLRET-Embedding-0.6B"),
    "dense_skillret_8b": ("SkillRet-8B", "Skill fine-tuned", "ThakiCloud/SKILLRET-Embedding-8B"),
    "dense_skillrouter_06b": ("SkillRouter-0.6B", "Skill fine-tuned", "pipizhao/SkillRouter-Embedding-0.6B"),
}


def load_grades() -> dict[tuple[str, str], int]:
    return {(r["instance_id"], r["skill_id"]): r["final"] for r in load_labels()}


def tau_b(a: list[float], b: list[float]) -> float:
    c = d = ta = tb = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        x, y = a[i] - a[j], b[i] - b[j]
        if x == 0 and y == 0:
            continue
        if x == 0:
            ta += 1
        elif y == 0:
            tb += 1
        elif x * y > 0:
            c += 1
        else:
            d += 1
    den = ((c + d + ta) * (c + d + tb)) ** 0.5
    return (c - d) / den if den else float("nan")


def mean_ndcg(qrels: dict, run: dict) -> float:
    """Mean nDCG@DEPTH over every task in qrels; a task the run doesn't score counts 0."""
    per_task = {m.query_id: m.value for m in ir_measures.iter_calc([nDCG @ DEPTH], qrels, run)}
    return sum(per_task.get(t, 0.0) for t in qrels) / len(qrels)


def load_run(ds: str, ret: str) -> list[dict]:
    path = (REPO_ROOT / "results" / "skill_usage" / f"{ret}.json" if ds == "skill_usage"
            else REPO_ROOT / "results" / "sra_bench" / f"{ds}-{ret}.json")
    return json.loads(path.read_text())["results"]


def scores(dsl: list[str], tasks: set[str], grades: dict) -> tuple[dict, list[int]]:
    """{retriever: [nDCG@DEPTH per condition]} for one benchmark, and the task counts (all, sample)."""
    human = {}
    for (task, skill), g in grades.items():
        human.setdefault(task, {})[skill] = g
    out, counts = {}, []
    for ret in RETRIEVERS:
        run, reference = {}, {}
        for ds in dsl:
            for r in load_run(ds, ret):
                if r["gold_skill_ids"]:  # tasks without a reference skill have no reference labels
                    run[r["instance_id"]] = {e["skill_id"]: -i for i, e in enumerate(r["retrieved"][:50])}
                    reference[r["instance_id"]] = {s: 1 for s in r["gold_skill_ids"]}
        sample = {t: v for t, v in reference.items() if t in tasks}
        sample_human = {t: v for t, v in human.items() if t in sample}
        assert set(sample_human) == set(sample), "a sampled task has no human grades"
        out[ret] = [mean_ndcg(reference, run), mean_ndcg(sample, run), mean_ndcg(sample_human, run)]
        counts = [len(reference), len(sample)]
    return out, counts


def main() -> None:
    lines = (line.strip() for line in (REPO_ROOT / "annotation" / "tasks.txt").read_text().splitlines())
    tasks = {t for t in lines if t and not t.startswith("#")}
    grades = load_grades()
    by_bench, counts, taus = {}, {}, {}
    for bench, dsl in BENCHMARKS.items():
        by_bench[bench], counts[bench] = scores(dsl, tasks, grades)
        cols = [[by_bench[bench][r][k] for r in RETRIEVERS] for k in range(len(CONDITIONS))]
        taus[bench] = [None] + [tau_b(cols[k - 1], cols[k]) for k in range(1, len(CONDITIONS))]
        print(f"{bench}: {counts[bench][0]} tasks, sample {counts[bench][1]}; Kendall's tau vs the column to the "
              f"left: " + ", ".join(f"{c} {t:.2f}" for c, t in zip(CONDITIONS[1:], taus[bench][1:])))

    def cell(bench: str, ret: str, k: int) -> str:
        values = {r: by_bench[bench][r][k] for r in RETRIEVERS}
        rank = sorted(values, key=values.get, reverse=True).index(ret) + 1
        return f"{values[ret]:.3f}".lstrip("0") + f" ({rank})"

    rows = [f"{RETRIEVERS[ret][0]} & " + " & ".join(cell(b, ret, k) for b in BENCHMARKS for k in range(len(CONDITIONS)))
            + r" \\" for ret in RETRIEVERS]
    tau_cells = [f"{t:.2f}".replace("0.", ".", 1) if t is not None else "" for b in BENCHMARKS for t in taus[b]]
    n = len(CONDITIONS)
    heads = {"Ref. all": r"\makecell[r]{Ref.\\all}", "Ref. sample": r"\makecell[r]{Ref.\\sample}",
             "Human sample": r"\makecell[r]{Human\\sample}"}
    (OUT_DIR / "table_retriever_ranking.tex").write_text("\n".join([
        "% Generated by figures/scripts/generate_retriever_ranking.py -- needs \\usepackage{booktabs,makecell}",
        *table_head("tab:retriever-ranking"),
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l {'r' * n} {'r' * n}@{{}}}}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{b}}}" for b in BENCHMARKS) + r" \\",
        rf"\cmidrule(lr){{2-{1 + n}}} \cmidrule(l){{{2 + n}-{1 + 2 * n}}}",
        "Retriever & " + " & ".join(heads[c] for _ in BENCHMARKS for c in CONDITIONS) + r" \\",
        r"\midrule", *rows, r"\midrule",
        "Kendall's $\\tau$ & " + " & ".join(tau_cells) + r" \\",
        r"\bottomrule", r"\end{tabular}", TABLE_TAIL,
    ]) + "\n")
    print(f"Saved: {OUT_DIR / 'table_retriever_ranking'}.tex")


if __name__ == "__main__":
    main()
