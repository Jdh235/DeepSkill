"""Dense (embedding) index building block, via pyterrier_dr.

Same philosophy as sparse_index.py: build the index, then compose the
retrieval pipeline yourself with plain PyTerrier/pyterrier_dr calls::

    from pyterrier_retrieval.retrievers.dense_index import build_dense_index

    model, index = build_dense_index(corpus_ids, corpus_texts, "path/to/dense_index")
    dense = model.query_encoder() >> index.retriever()
"""

from __future__ import annotations

import shutil
from pathlib import Path


def build_dense_index(
    corpus_ids: list[str],
    corpus_texts: list[str],
    index_path: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 32,
    max_tokens: int | None = None,
    rebuild: bool = True,
):
    """Build a FlexIndex over ``corpus_texts``, keyed by ``corpus_ids``.

    Rebuilds into ``index_path`` by default (matches sparse_index.build_index()
    — a retrieval run should always reindex, not risk serving a stale index).
    Set ``rebuild=False`` to reuse an already-built index at ``index_path``
    instead: e.g. re-running against the *same* corpus (once per SRA-Bench
    dataset, which only differs in queries) should reuse one index rather
    than rebuilding it from scratch every time -- matches
    colbert_index.build_colbert_index()'s identical ``rebuild=`` convention.
    Checks for ``pt_meta.json``, the same on-disk marker FlexIndexer itself
    writes on successful completion (pyterrier_dr's Artifact convention).

    ``batch_size`` matters more for some models than others: encoder-only
    models like BGE have a hard 512-token architectural limit and truncate
    silently regardless of document length, but causal-LM-based embedding
    models (e.g. Qwen3-Embedding) support much longer context by default,
    so a large batch of long documents (some SKILL.md files here run past
    1M characters) can OOM -- the sragents repo's own qwen3_emb_06b/8b
    factories already use a smaller batch_size than BGE's for exactly this
    reason, not a cap on sequence length itself.

    ``max_tokens``, if set, truncates each document to that many tokens
    (via the model's own tokenizer) before encoding. Needed on top of
    ``batch_size`` for very long-context models: self-attention memory
    scales quadratically with sequence length, so a single extreme-length
    document can OOM even at batch_size=1, independent of batch size.
    Matches the truncate-before-encode approach the Skill-Usage benchmark's
    own search_server/index_builder.py uses for this same corpus.

    Returns ``(model, index)`` — compose retrieval yourself, e.g.
    ``model.query_encoder() >> index.retriever()``.
    """
    import pyterrier_dr

    index_path = Path(index_path)

    if not rebuild and (index_path / "pt_meta.json").exists():
        model = pyterrier_dr.SBertBiEncoder(model_name, batch_size=batch_size)
        index = pyterrier_dr.FlexIndex(str(index_path), sim_fn=pyterrier_dr.SimFn.cos)
        return model, index

    if index_path.exists() and any(index_path.iterdir()):
        shutil.rmtree(index_path)

    if max_tokens is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def _truncate(text: str) -> str:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return tokenizer.decode(ids[:max_tokens])

        corpus_texts = [_truncate(t) for t in corpus_texts]

    model = pyterrier_dr.SBertBiEncoder(model_name, batch_size=batch_size)
    # FlexIndex defaults to a raw dot product (sim_fn="dot"), not cosine --
    # score then depends on embedding magnitude, not just direction. cos
    # normalizes both vectors at retrieval time, matching the sragents
    # repo's own normalize_embeddings=True convention.
    index = pyterrier_dr.FlexIndex(str(index_path), sim_fn=pyterrier_dr.SimFn.cos)

    docs = ({"docno": docno, "text": text} for docno, text in zip(corpus_ids, corpus_texts))
    (model.doc_encoder() >> index.indexer()).index(docs)

    return model, index
