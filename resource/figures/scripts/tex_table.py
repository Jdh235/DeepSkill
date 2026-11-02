"""Shared table wrapper for the generated .tex files: a table float with a label and no caption (the paper adds
the captions)."""

from __future__ import annotations


def table_head(label: str) -> list[str]:
    """Opening lines: table float and label."""
    return [r"\begin{table}[t]", r"\centering", rf"\label{{{label}}}"]


TABLE_TAIL = r"\end{table}"
