"""ColBERT v2 (late-interaction, multi-vector) index building block, via
pyterrier_colbert2.

Architecturally different from dense_index.py's build_dense_index(): ColBERT
encodes each token to its own vector and scores via MaxSim, not one vector
per document/query, so it needs its own indexer/searcher (the Stanford
colbert-ai package wrapped by pyterrier_colbert2) rather than pyterrier_dr's
FlexIndex. Runs in its own conda env (sragents-colbert), kept isolated from
sragents-pt -- pyterrier_colbert2 pins transformers<5, which conflicts with
sragents-pt's transformers 5.x (needed by sentence-transformers for the
other dense/* retrievers).

    from pyterrier_retrieval.retrievers.colbert_index import build_colbert_index

    ranker = build_colbert_index(corpus_ids, corpus_texts, "path/to/colbert_index")
    colbert = ranker.end_to_end(k=50)
"""

from __future__ import annotations

import shutil
from pathlib import Path


def build_colbert_index(
    corpus_ids: list[str],
    corpus_texts: list[str],
    index_path: str | Path,
    checkpoint: str = "colbert-ir/colbertv2.0",
    index_name: str = "colbert_index",
    nbits: int = 2,
    rebuild: bool = True,
    doc_maxlen: int | None = 512,
    batch_size: int | None = None,
):
    """Build a ColBERTv2 index over ``corpus_texts``, keyed by ``corpus_ids``.

    Rebuilds into ``index_path`` by default (matches build_dense_index() and
    build_index() -- a retrieval run should always reindex, not risk serving
    a stale index). Set ``rebuild=False`` to reuse an already-built index at
    ``index_path`` instead: unlike the other dense retrievers' FlexIndex
    (fast to rebuild), ColBERT's PLAID indexing takes tens of minutes even at
    SRA-Bench's ~26k-skill scale, so re-running against the *same* corpus
    (e.g. once per SRA-Bench dataset, which only differs in queries) should
    reuse one index rather than rebuilding it from scratch every time.

    ``doc_maxlen``, if set (default 512), overrides colbert-ir/colbertv2.0's
    own trained default of 180 tokens/doc (loaded from its artifact.metadata
    on HuggingFace) -- matching BGE/Contriever's 512-token
    budget instead, since 180 was leaving most of a typical SKILL.md unseen.
    This bypasses pyterrier_colbert2's ColbertV2Indexer (which hardcodes
    ColBERTConfig(nbits=...) with no way to pass doc_maxlen through it) and
    replicates its indexing logic directly so our value actually takes
    effect -- explicitly-set ColBERTConfig fields override a checkpoint's
    saved config, but fields left at their dataclass "soft default" (as
    ColbertV2Indexer leaves doc_maxlen) get silently overridden BY the
    checkpoint instead. Set ``doc_maxlen=None`` to go back to the
    checkpoint's own trained value via the plain ColbertV2Indexer path.

    ``batch_size``, if set, is ColBERT's indexing batch size (``index_bsize``);
    left unset, ColBERT's own default (64) applies, as in the original runs.

    Returns a ``ColBERTv2Index`` ranker -- compose retrieval yourself via
    ``ranker.end_to_end(k=...)``.
    """
    index_path = Path(index_path)
    full_index_path = index_path / index_name / "indexes" / index_name

    if not rebuild and (full_index_path / "pt_meta.json").exists():
        from pyterrier_colbert2.ranking import ColBERTv2Index
        return ColBERTv2Index(str(full_index_path), colbert=checkpoint)

    if index_path.exists() and any(index_path.iterdir()):
        shutil.rmtree(index_path)

    if doc_maxlen is None:
        from pyterrier_colbert2.indexing import ColbertV2Indexer

        indexer = ColbertV2Indexer(
            index_location=str(index_path),
            checkpoint=checkpoint,
            index_name=index_name,
            nbits=nbits,
        )
        docs = ({"docno": docno, "text": text} for docno, text in zip(corpus_ids, corpus_texts))
        return indexer.index(docs)

    # Replicates ColbertV2Indexer.index() (pyterrier_colbert2/indexing.py)
    # exactly, but with doc_maxlen explicitly set in ColBERTConfig so it
    # actually overrides the checkpoint's own saved value.
    import json
    from npids import Lookup
    from pyterrier_colbert2.indexing import ColBERTConfig, Indexer, Run, RunConfig
    from pyterrier_colbert2.ranking import ColBERTv2Index

    index_path.mkdir(parents=True, exist_ok=True)
    collection = [f"{i}\t{text}" for i, text in enumerate(corpus_texts)]

    with Run().context(RunConfig(nranks=1, avoid_fork_if_possible=True, experiment=index_name, root=str(index_path))):
        config = ColBERTConfig(nbits=nbits, doc_maxlen=doc_maxlen,
                               **({} if batch_size is None else {"index_bsize": batch_size}))
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)

    Lookup.build(corpus_ids, str(full_index_path / "docnos.npids"))
    with open(full_index_path / "pt_meta.json", "wt") as f_meta:
        json.dump({
            "type": ColBERTv2Index.ARTIFACT_TYPE,
            "format": ColBERTv2Index.ARTIFACT_FORMAT,
            "model_checkpoint": checkpoint,
            "nbits": nbits,
        }, f_meta)

    return ColBERTv2Index(str(full_index_path), colbert=checkpoint)
