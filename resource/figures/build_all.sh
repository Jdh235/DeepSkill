#!/usr/bin/env bash
# Rebuild everything the paper includes into figures/output/ (one flat folder to upload to Overleaf).
#   1. final labels from labels/human/: qrels and consensus flags (first round + re-annotation)
#   2. every table and figure in the paper, and the retriever ranking table
#
#   bash figures/build_all.sh              (uses `python` on PATH; set PYTHON=... to use another)
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" labels/human/generate_final_labels.py
"$PY" figures/scripts/generate_rq1_figures.py
"$PY" figures/scripts/generate_judge_tables.py
"$PY" figures/scripts/generate_judge_grade_bars.py
"$PY" figures/scripts/generate_label_figures.py
"$PY" figures/scripts/generate_grade_accuracy.py
"$PY" figures/scripts/generate_help_tie_hurt.py
"$PY" figures/scripts/generate_top5_tables.py
"$PY" figures/scripts/generate_retriever_ranking.py

echo
echo "figures/output/:"
ls -1R figures/output

# PNG copies of every figure, for checking locally (macOS only, needs sips; upload figures/output/)
if command -v sips >/dev/null; then
  mkdir -p figures/preview
  for f in figures/output/*.pdf; do
    sips -s format png -Z 1600 "$f" --out "figures/preview/$(basename "${f%.pdf}").png" >/dev/null
  done
  echo "PNG previews in figures/preview/"
fi
