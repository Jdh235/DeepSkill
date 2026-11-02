"""``sragents sweep`` — inject each candidate-pool skill individually."""

import sys
from pathlib import Path

from sragents.cli._common import parse_kv_list
from sragents.config import RESULTS_DIR
from sragents.experiments.sweep import resolve_annotation_files, run_sweep


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "sweep", help="Inject each candidate-pool skill individually (single-skill sweep)",
        description=(
            "For each instance, injects exactly one skill at a time from a "
            "pre-computed union candidate pool (e.g. a PyTerrier annotation "
            "file), repeating `--runs` times per instance/skill pair. Pool "
            "size varies per instance (it's a union of several retrievers' "
            "results), so the number of cells run per dataset is data-driven "
            "rather than a fixed top-K."
        ),
    )
    p.add_argument("--annotation-dir", type=Path, default=None,
                   help="Directory containing `{dataset}-annotation.json` "
                        "candidate-pool files")
    p.add_argument("--annotation", action="append", default=[],
                   metavar="DATASET=PATH",
                   help="Explicit annotation file for one dataset "
                        "(repeatable); overrides --annotation-dir for that "
                        "dataset")
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", required=True,
                   help="OpenAI-compatible endpoint URL")
    p.add_argument("--dataset", nargs="*", default=None,
                   help="Restrict to specific dataset(s); default: all "
                        "datasets with a matching annotation file")
    p.add_argument("--runs", type=int, default=3,
                   help="Independent repeats per instance/skill pair (default: 3); "
                        "ignored if --run is given")
    p.add_argument("--run", type=int, nargs="*", default=None,
                   metavar="N",
                   help="Restrict to specific run number(s) (e.g. `--run 2` "
                        "so a single process owns run 2 while sibling "
                        "processes handle the others against the same "
                        "--workspace and a shared inference server)")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Root directory for all outputs; model + run "
                        "subdirectories are created under it "
                        "(default: ./results/candidate_sweep)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Skill library path (corpus JSON; default: ../../resource/data/sra_bench/corpus/corpus.json)")
    p.add_argument("--instances-dir", type=Path, default=None,
                   help="Instances directory (default: ../../resource/data/sra_bench/instances/)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--eval-workers", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--engine", default="direct",
                   help="Engine for the non-ToolQA datasets (default: direct; `direct_full` also injects each "
                        "skill's name and description). Use a fresh --workspace when changing it: finished "
                        "cells are skipped, not redone.")
    p.add_argument("--positions", type=_position_range, default=None, metavar="LO-HI",
                   help="Only run pool positions LO..HI (inclusive), e.g. `--positions 33-60`. Lets several "
                        "processes split one (dataset, run) without working on the same position.")
    p.set_defaults(func=run)


def _position_range(text: str) -> tuple[int, int]:
    import argparse
    try:
        lo, hi = (int(x) for x in text.split("-"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LO-HI, e.g. 33-60, got {text!r}")
    if not 1 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"need 1 <= LO <= HI, got {text!r}")
    return lo, hi


def run(args) -> None:
    explicit = parse_kv_list(args.annotation)
    files = resolve_annotation_files(args.annotation_dir, explicit, args.dataset)
    if not files:
        sys.exit(
            "sragents: error: no candidate-pool annotation files found. "
            "Pass --annotation-dir DIR (containing `{dataset}-annotation.json` "
            "files) or --annotation DATASET=PATH."
        )
    print(f"Datasets: {', '.join(sorted(files))}")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")

    run_sweep(
        annotation_files=files,
        model=args.model,
        api_base=args.api_base,
        runs=args.run if args.run else args.runs,
        workspace=args.workspace or (RESULTS_DIR / "candidate_sweep"),
        corpus_path=args.corpus,
        instances_dir=args.instances_dir,
        workers=args.workers,
        eval_workers=args.eval_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        engine=args.engine,
        positions=args.positions,
    )
