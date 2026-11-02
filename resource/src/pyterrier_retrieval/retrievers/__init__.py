"""Runnable retriever jobs, one file per retriever.

sparse/  — Terrier weighting models (bm25.py, hiemstra_lm.py) and SPLADE (splade.py)
dense/   — embedding-based retrievers (bge.py, colbertv2.py, qwen3_emb_8b.py, ...)

Each file is a standalone CLI script (load corpus -> index -> retrieve ->
save + score), independent of the others, so any one of them can diverge
(different query processing, different controls, ...) without affecting
the rest.
"""
