"""facebook/contriever-msmarco dense retrieval over SRA-Bench or Skill-Usage tasks.

Token limit: 512 (verified via SentenceTransformer's own max_seq_length,
which exactly matches this BERT-base model's architectural
max_position_embeddings -- pyterrier_dr.SBertBiEncoder doesn't override it,
so it's a hard truncation with no headroom, same as BGE).

Same shape as dense/bge.py -- only --dataset (and --instances, for
SRA-Bench) are required, everything else defaults from --dataset:

    python -m pyterrier_retrieval.retrievers.dense.contriever \\
        --dataset sra_bench --instances data/sra_bench/instances/theoremqa.json
    # -> corpus=data/sra_bench/corpus/corpus.json, index-path=<fresh temp dir>,
    #    output=results/sra_bench/theoremqa-dense_contriever.json

    python -m pyterrier_retrieval.retrievers.dense.contriever --dataset skill_usage
    # -> corpus/tasks-dir/mapping from data/skill_usage/, index-path=<fresh temp dir>,
    #    output=results/skill_usage/dense_contriever.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from pyterrier_retrieval.corpus import load_sra_bench_corpus, load_skill_usage_corpus, skill_text
from pyterrier_retrieval.metrics import compute_retrieval_metrics
from pyterrier_retrieval.schema import RetrievalRecord, RetrievalResults
from pyterrier_retrieval.retrievers.dense_index import build_dense_index
from pyterrier_retrieval.units import (
    interleave,
    skill_usage_units_from_instructions,
    sra_bench_units,
)

MODEL_NAME = "facebook/contriever-msmarco"
RETRIEVER_NAME = "dense_contriever"
# Contriever has no instruction-tuned query format -- unlike BGE/Qwen3, the
# sragents repo's own contriever factory (sragents.retrieve.dense) uses no
# query prefix at all.
QUERY_PREFIX = ""
# Matches the sragents repo's own contriever factory. Contriever is a plain
# BERT-base encoder (like BGE), with a hard 512-token architectural limit,
# so unlike Qwen3-Embedding there's no OOM risk from very long documents at
# this batch size.
BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["sra_bench", "skill_usage"], required=True)
    p.add_argument("--instances", type=Path,
                    help="sra_bench: which instances file (required — no single default)")
    p.add_argument("--corpus", type=Path, default=None, help="default: derived from --dataset")
    p.add_argument("--tasks-dir", type=Path, default=None, help="skill_usage default: data/skill_usage/tasks")
    p.add_argument("--mapping", type=Path, default=None,
                    help="skill_usage default: data/skill_usage/task_skill_mapping.json")
    p.add_argument("--index-path", type=Path, default=None,
                    help="default: a fresh temp dir (rebuilt by default; pass "
                         "--no-rebuild-index to reuse an existing index here instead)")
    p.add_argument("--output", type=Path, default=None, help="default: results/<dataset>/...")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"skills encoded per batch (default: {BATCH_SIZE}) -- lower it if the GPU runs "
                         "out of memory; the embeddings only change by floating-point noise, since "
                         "padding is masked out")
    p.add_argument("--rebuild-index", action=argparse.BooleanOptionalAction, default=True,
                    help="default: always rebuild. Pass --no-rebuild-index to reuse an "
                         "already-built index at --index-path instead -- useful when running "
                         "multiple SRA-Bench datasets back-to-back against the same corpus, so "
                         "the index is only built once rather than once per dataset.")
    args = p.parse_args()
    apply_defaults(args)
    return args


def apply_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "sra_bench":
        if not args.instances:
            raise ValueError("--instances is required for --dataset sra_bench (which SRA-Bench file to run)")
        args.corpus = args.corpus or Path("data/sra_bench/corpus/corpus.json")
        args.index_path = args.index_path or Path(tempfile.mkdtemp(prefix="dense_contriever_index_"))
        args.output = args.output or Path(f"results/sra_bench/{args.instances.stem}-{RETRIEVER_NAME}.json")
    else:
        args.corpus = args.corpus or Path("data/skill_usage/corpus/skill_usage/skills_meta.jsonl")
        args.tasks_dir = args.tasks_dir or Path("data/skill_usage/tasks")
        args.mapping = args.mapping or Path("data/skill_usage/task_skill_mapping.json")
        args.index_path = args.index_path or Path(tempfile.mkdtemp(prefix="dense_contriever_index_"))
        args.output = args.output or Path(f"results/skill_usage/{RETRIEVER_NAME}.json")


def load_corpus_for(args: argparse.Namespace) -> list[dict]:
    if args.dataset == "sra_bench":
        return load_sra_bench_corpus(args.corpus)
    return load_skill_usage_corpus(args.corpus)


def load_units_for(args: argparse.Namespace) -> list[dict]:
    if args.dataset == "sra_bench":
        return sra_bench_units(args.instances)
    return skill_usage_units_from_instructions(args.tasks_dir, args.mapping)


def dataset_label(args: argparse.Namespace) -> str:
    """More specific than args.dataset for sra_bench (e.g. "theoremqa"), which
    matches what the un-unified job file used to record."""
    if args.dataset == "sra_bench":
        instances = json.loads(args.instances.read_text())
        return instances[0]["dataset"] if instances else "sra_bench"
    return "skill_usage"


def main() -> None:
    args = parse_args()

    # ── Corpus + index ───────────────────────────────────────────────────
    corpus = load_corpus_for(args)
    corpus_ids = [s["skill_id"] for s in corpus]
    corpus_texts = [skill_text(s) for s in corpus]
    print(f"Skill library: {len(corpus)} skills")

    model, index = build_dense_index(
        corpus_ids, corpus_texts, args.index_path, model_name=MODEL_NAME, batch_size=args.batch_size,
        rebuild=args.rebuild_index,
    )
    pipeline = model.query_encoder() >> index.retriever()

    # ── Units (queries + gold) ───────────────────────────────────────────
    units = load_units_for(args)
    print(f"Units: {len(units)}")

    # One qid per (unit, query) pair so multi-query units retrieve
    # independently before being interleaved back into one ranked list.
    # "|||" as a separator is safe: no task/instance id in either dataset
    # contains it.
    query_rows = [
        {"qid": f"{unit['id']}|||{i}", "query": QUERY_PREFIX + q}
        for unit in units
        for i, q in enumerate(unit["queries"])
    ]
    topics = pd.DataFrame(query_rows)
    print(f"Queries: {len(topics)}")

    # ── Retrieve ──────────────────────────────────────────────────────────
    run = pipeline.transform(topics)
    by_qid = {
        qid: group.sort_values("score", ascending=False).head(args.top_k * 2)
        for qid, group in run.groupby("qid")
    }

    records = []
    for unit in units:
        ranked_lists, score_by_id = [], {}
        for i in range(len(unit["queries"])):
            group = by_qid.get(f"{unit['id']}|||{i}")
            if group is None:
                ranked_lists.append([])
                continue
            ranked_lists.append(list(group["docno"]))
            for sid, score in zip(group["docno"], group["score"]):
                score_by_id.setdefault(sid, float(score))
        merged = interleave(ranked_lists, args.top_k)
        retrieved = [{"skill_id": sid, "score": score_by_id[sid]} for sid in merged]
        records.append(RetrievalRecord(
            instance_id=unit["id"], gold_skill_ids=unit["gold_skill_ids"], retrieved=retrieved,
        ))

    # ── Save + score ─────────────────────────────────────────────────────
    metrics = compute_retrieval_metrics(
        [{"instance_id": r.instance_id, "gold_skill_ids": r.gold_skill_ids, "retrieved": r.retrieved}
         for r in records],
        top_k=args.top_k,
    )
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    RetrievalResults(
        retriever=RETRIEVER_NAME,
        top_k=args.top_k,
        corpus_size=len(corpus),
        records=records,
        metrics=metrics,
        dataset=dataset_label(args),
    ).dump(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
