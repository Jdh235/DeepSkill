"""Score a retriever on the annotated tasks against the graded relevance labels.

    pip install python-terrier huggingface_hub
    python score_retriever.py sra_bench
    python score_retriever.py skill_usage

The retriever is BM25 as run for the judgment pool, which scores nDCG@5 0.276 on SRA-Bench and 0.578 on
Skill-Usage.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyterrier as pt
from huggingface_hub import hf_hub_download
from ir_measures import P, R, Judged, nDCG

LABELS = "jdh235/DeepSkill"
DATASETS = {"sra_bench": ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"],
            "skill_usage": ["skill_usage"]}
# A skill counts as relevant when graded 2 or 3. Only the pool's top 5 was judged, so measure at depth 5
# and report Judged@5, the share of the top 5 with a label (unjudged skills count as non-relevant).
MEASURES = [nDCG @ 5, P(rel=2) @ 5, R(rel=2) @ 5, Judged @ 5]
FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

NAME = "BM25"  # shown in the results


def tokenize(text: str) -> str:
    """Lowercase, punctuation to spaces, split on whitespace: the pool's sparse retrievers' tokens. Tokens over
    100 characters (base64 strings and the like) are dropped: Terrier reserves max.term.length for every term,
    so allowing them makes a 15 GB index, and dropping them leaves every task's top 5 unchanged."""
    return " ".join(w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if len(w) <= 100)


def retriever(skills: list[dict], index_dir: str) -> pt.Transformer:
    """The retriever to score -- swap in your own; index_dir is a scratch folder, deleted afterwards. BM25 as in
    the judgment pool: no stemming or stopwords."""
    index = pt.IterDictIndexer(
        index_dir, meta={"docno": max(len(s["skill_id"]) for s in skills)},
        tokeniser="WhitespaceTokeniser", stemmer=None, stopwords=None, properties={"max.term.length": "150"},
    ).index({"docno": s["skill_id"], "text": tokenize(s["text"])} for s in skills)
    return (pt.apply.query(lambda q: tokenize(q["query"]))
            >> pt.terrier.Retriever(index, wmodel="BM25", num_results=50, controls={"termpipelines": ""}))


def download(repo: str, path: str) -> str:
    return hf_hub_download(repo, path, repo_type="dataset")


def load_labels(benchmark: str, labels_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(qrels, queries) for the benchmark's annotated tasks, in PyTerrier's columns."""
    def path(name: str) -> str:
        return str(labels_dir / name) if labels_dir else download(LABELS, name)
    datasets = DATASETS[benchmark]
    qrels = pd.concat(pt.io.read_qrels(path(f"qrels/{ds}-relevance.qrels")) for ds in datasets)
    queries = pd.concat(pd.read_json(path(f"queries/{ds}.jsonl"), lines=True) for ds in datasets)
    return qrels, queries.rename(columns={"query_id": "qid"})


def load_skills(benchmark: str) -> list[dict]:
    """Every skill in the benchmark's corpus as {"skill_id", "text"}, the text being its name, description
    and content, one per line. Skill-Usage's content is the skill's SKILL.md without its frontmatter."""
    if benchmark == "sra_bench":
        skills = json.loads(Path(download("WeihangSu/SRA-Bench", "corpus/corpus.json")).read_text())
    else:
        archive = zipfile.ZipFile(download("Shiyu-Lab/Skill-Usage", "skills-34k/skills.zip"))
        files = set(archive.namelist())
        skills = []
        for line in Path(download("Shiyu-Lab/Skill-Usage", "skills-34k/skills_meta.jsonl")).read_text().splitlines():
            if not line:
                continue
            meta = json.loads(line)
            name = f"{meta['skill_id']}/SKILL.md"
            content = ""
            if name in files:
                text = io.TextIOWrapper(archive.open(name), encoding="utf-8", errors="replace").read()
                content = FRONTMATTER.sub("", text, count=1).strip()
            skills.append({"skill_id": meta["skill_id"], "name": meta.get("name", ""),
                           "description": meta.get("description", ""), "content": content})
    return [{"skill_id": s["skill_id"],
             "text": "\n".join(s[k] for k in ("name", "description", "content") if s.get(k))} for s in skills]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=DATASETS)
    p.add_argument("--labels-dir", type=Path, help="a local copy of the labels dataset (default: download it)")
    args = p.parse_args()

    qrels, queries = load_labels(args.benchmark, args.labels_dir)
    skills = load_skills(args.benchmark)
    print(f"{len(queries)} tasks, {len(qrels)} judged pairs, {len(skills)} skills")
    with tempfile.TemporaryDirectory(prefix="skills_index_") as index_dir:
        results = pt.Experiment([retriever(skills, index_dir)], queries, qrels, eval_metrics=MEASURES, names=[NAME])
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
