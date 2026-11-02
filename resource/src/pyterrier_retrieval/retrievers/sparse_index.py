"""Sparse (Terrier) index building block — whitespace tokenisation.

Token limit: none, per document -- unlike every transformer-based retriever
in this repo (BGE/contriever: 512, Qwen3-Embedding: 32,752,
ColBERT: 180, SPLADE: 256), plain whitespace tokenisation indexes every
token in a document regardless of length; there's no model context window
to truncate against. The only length-related setting here is
``properties={"max.term.length": "10000"}`` (below), which caps the
character length of a single *term* (to guard against pathological tokens),
not the number of terms per document.

Uses whitespace tokenisation with no stemming or stopword removal.
All datasets (bigcodebench, theoremqa, logicbench, champ, medcalcbench)
contain code, maths, or technical identifiers where stemming corrupts tokens
and stopword removal strips semantically meaningful terms (if, for, return…).

    from pyterrier_retrieval.retrievers.sparse_index import build_index, tokenize
    import pyterrier as pt

    index = build_index(corpus_ids, corpus_texts, "path/to/index")
    bm25  = pt.terrier.Retriever(index, wmodel="BM25", num_results=50,
                                 controls={"termpipelines": ""})

Queries must be pre-tokenized with ``tokenize`` before being passed to the
retriever so the document and query pipelines match exactly.

Each weighting model has its own runnable job in retrievers/sparse/
(bm25.py, hiemstra_lm.py).
"""

from __future__ import annotations

import re
from pathlib import Path

_PUNCT = re.compile(r"[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _PUNCT.sub(" ", text.lower()).split()


def build_index(
    corpus_ids: list[str],
    corpus_texts: list[str],
    index_path: str | Path,
):
    """Build a whitespace-tokenised Terrier index (no stemming, no stopwords).

    Always rebuilds into ``index_path`` — indexing takes ~30-50s, which is
    cheap enough that reusing a possibly-stale index isn't worth the risk.
    """
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()

    index_path = Path(index_path)
    docs = ({"docno": docno, "text": " ".join(tokenize(text))}
            for docno, text in zip(corpus_ids, corpus_texts))
    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={"docno": max(len(d) for d in corpus_ids)},
        tokeniser="WhitespaceTokeniser",
        stemmer=None,
        stopwords=None,
        overwrite=True,
        threads=1,
        properties={"max.term.length": "10000"},
    )
    indexer.index(docs)

    properties_text = (index_path / "data.properties").read_text()
    if "index.inverted.class" not in properties_text:
        raise RuntimeError(
            f"Index at {index_path} is missing its inverted index structure "
            "— Terrier's inversion step didn't complete. Retry build_index()."
        )
    return pt.IndexFactory.of(str(index_path))


def build_fielded_index(
    corpus: list[dict],
    index_path: str | Path,
    fields: tuple[str, ...] = ("name", "description", "content"),
):
    """Build a whitespace-tokenised, multi-field Terrier index (for BM25F),
    keyed by each skill's own ``name``/``description``/``content`` fields
    (see corpus.py) kept separate rather than flattened by ``skill_text()``
    -- BM25F needs per-field term statistics to weight fields differently
    at retrieval time (``controls={"w.0": ..., "c.0": ..., ...}``, in the
    same order as ``fields``). Same tokenisation as build_index() (no
    stemming, no stopwords) for the same reason (code/maths/technical
    identifiers).

    Always rebuilds into ``index_path`` (see build_index()'s docstring).
    """
    import pyterrier as pt
    if not pt.java.started():
        pt.java.init()

    index_path = Path(index_path)
    docs = [
        {
            "docno": s["skill_id"],
            **{f: " ".join(tokenize(s.get(f, "") or "")) for f in fields},
        }
        for s in corpus
    ]
    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={"docno": max(len(s["skill_id"]) for s in corpus)},
        text_attrs=list(fields),
        fields=True,
        tokeniser="WhitespaceTokeniser",
        stemmer=None,
        stopwords=None,
        overwrite=True,
        threads=1,
        properties={"max.term.length": "10000"},
    )
    indexer.index(docs)

    properties_text = (index_path / "data.properties").read_text()
    if "index.inverted.class" not in properties_text:
        raise RuntimeError(
            f"Index at {index_path} is missing its inverted index structure "
            "— Terrier's inversion step didn't complete. Retry build_fielded_index()."
        )
    return pt.IndexFactory.of(str(index_path))
