"""SPLADE (learned sparse) index building block, via pyterrier_splade.

Unlike sparse_index.py's plain whitespace-tokenised weighting models, SPLADE
learns term weights via a BERT-style masked-LM (naver/splade-cocondenser-
ensembledistil by default): each document is encoded into a sparse
term:weight dict ("toks"), then indexed into a standard Terrier index in
"pretokenised" mode -- no further tokenisation/stemming/stopwords applied,
since SPLADE's own weights already are the index.

    from pyterrier_retrieval.retrievers.splade_index import build_splade_index
    import pyterrier as pt

    splade, index = build_splade_index(corpus_ids, corpus_texts, "path/to/index")
    pipeline = splade.query_encoder() >> pt.terrier.Retriever(index, wmodel="Tf")
"""

from __future__ import annotations

from pathlib import Path


def build_splade_index(
    corpus_ids: list[str],
    corpus_texts: list[str],
    index_path: str | Path,
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    batch_size: int = 32,
    max_length: int = 512,
    rebuild: bool = True,
):
    """Build a Terrier index over SPLADE-encoded ``corpus_texts``.

    Rebuilds into ``index_path`` by default (matches sparse_index.build_index()
    and dense_index.build_dense_index() -- a retrieval run should always
    reindex, not risk serving a stale index). Set ``rebuild=False`` to reuse
    an already-built index at ``index_path`` instead: e.g. re-running against
    the *same* corpus (once per SRA-Bench dataset, which only differs in
    queries) should reuse one index rather than rebuilding it from scratch
    every time -- matches build_dense_index()/build_colbert_index()'s
    identical ``rebuild=`` convention. Checks for ``index.inverted.class`` in
    ``data.properties``, the same on-disk marker sparse_index.build_index()
    uses to confirm Terrier's inversion step completed.

    Truncation: ``max_length`` defaults to 512 here, matching
    naver/splade-cocondenser-ensembledistil's actual architectural limit (a
    BERT-base checkpoint) -- overriding ``pyterrier_splade.Splade``'s own
    library default of 256, which was leaving twice as much of each
    document truncated away as necessary. Unlike ColBERT (one embedding per
    token, so raising doc_maxlen scales index size/compute with it), SPLADE
    aggregates into a single sparse vector per document (max-pooled over
    token logits), so a larger max_length mainly costs more at encoding
    time, not in the resulting index's size. Pass ``max_length=256`` to
    restore the library's own default.

    Returns ``(splade, index)`` -- compose retrieval yourself, e.g.
    ``splade.query_encoder() >> pt.terrier.Retriever(index, wmodel="Tf")``.
    """
    import pyterrier as pt
    from pyterrier_splade import Splade

    if not pt.java.started():
        pt.java.init()

    index_path = Path(index_path)
    splade = Splade(model_name, max_length=max_length)

    properties_path = index_path / "data.properties"
    if not rebuild and properties_path.exists() and "index.inverted.class" in properties_path.read_text():
        index = pt.IndexFactory.of(str(index_path))
        return splade, index

    indexer = pt.IterDictIndexer(
        str(index_path),
        meta={"docno": max(len(d) for d in corpus_ids)},
        pretokenised=True,
        overwrite=True,
        threads=1,
    )
    docs = ({"docno": docno, "text": text} for docno, text in zip(corpus_ids, corpus_texts))
    (splade.doc_encoder(batch_size=batch_size) >> indexer).index(docs)

    index = pt.IndexFactory.of(str(index_path))
    return splade, index
