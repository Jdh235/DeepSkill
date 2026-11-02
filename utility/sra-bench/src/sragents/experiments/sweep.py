"""Candidate-pool sweep: inject each pooled candidate skill individually.
"""

import contextlib
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from sragents.config import (
    CORPUS_PATH,
    INSTANCES_DIR,
    discover_datasets,
    model_short_name,
)


def _run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


@contextlib.contextmanager
def _bigcodebench_eval_lock(workspace: Path):
    lock_path = workspace / "_bigcodebench_eval.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)


def _load_pool(annotation_path: Path) -> dict[str, list[str]]:
    data = json.loads(annotation_path.read_text())
    return {
        r["instance_id"]: [c["skill_id"] for c in r.get("retrieved", [])]
        for r in data["results"]
    }


def pool_task_ids(pool_dir: Path, dataset: str) -> set[str]:
    for name in (f"{dataset}-pool.json", f"{dataset}-annotation.json"):
        path = Path(pool_dir) / name
        if path.exists():
            return set(_load_pool(path))
    raise FileNotFoundError(f"no {dataset}-pool.json or {dataset}-annotation.json in {pool_dir}")


def resolve_annotation_files(
    annotation_dir: Path | None,
    explicit: dict[str, str],
    datasets: list[str] | None = None,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if annotation_dir:
        for p in sorted(Path(annotation_dir).glob("*-annotation.json")):
            ds = p.name[: -len("-annotation.json")]
            files[ds] = p
        for p in sorted(Path(annotation_dir).glob("*-pool.json")):
            ds = p.name[: -len("-pool.json")]
            files[ds] = p
    for ds, path in explicit.items():
        files[ds] = Path(path)

    runnable = set(discover_datasets())
    files = {ds: p for ds, p in files.items() if ds in runnable}
    if datasets:
        files = {ds: files[ds] for ds in datasets if ds in files}
    return files


def _cached_position_instances(
    cache_dir: Path,
    dataset: str,
    pos: int,
    instances: list[dict],
    pool: dict[str, list[str]],
) -> Path | None:
    """Instance subset whose pool has >= pos candidates, cached to disk."""
    out = cache_dir / dataset / f"pos{pos}.json"
    if out.exists():
        return out
    subset = [i for i in instances if len(pool.get(i["instance_id"], [])) >= pos]
    if not subset:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, ensure_ascii=False))
    return out


def run_sweep(
    annotation_files: dict[str, Path],
    model: str,
    api_base: str,
    *,
    runs: int | list[int] = 3,
    workspace: Path,
    corpus_path: Path | None = None,
    instances_dir: Path | None = None,
    workers: int = 32,
    eval_workers: int = 32,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    thinking: bool = False,
    engine: str = "direct",
    positions: tuple[int, int] | None = None,
) -> None:
    workspace = Path(workspace)
    corpus_path = Path(corpus_path) if corpus_path else CORPUS_PATH
    instances_dir = Path(instances_dir) if instances_dir else INSTANCES_DIR
    run_ids = list(runs) if isinstance(runs, list) else list(range(1, runs + 1))
    model_name = model_short_name(model)
    label_suffix = "_thinking" if thinking else ""
    cache_dir = workspace / "_pool"

    for ds, annotation_path in annotation_files.items():
        instances_file = instances_dir / f"{ds}.json"
        if not instances_file.exists():
            print(f"  [skip] {ds}: instances file missing", file=sys.stderr)
            continue
        if not annotation_path.exists():
            print(f"  [skip] {ds}: annotation file missing: {annotation_path}",
                  file=sys.stderr)
            continue

        instances = json.loads(instances_file.read_text())
        pool = _load_pool(annotation_path)
        missing = sum(1 for i in instances if not pool.get(i["instance_id"]))
        if missing:
            print(f"  [warn] {ds}: {missing}/{len(instances)} instances have "
                  "no annotated candidates and will be skipped", file=sys.stderr)

        max_len = max((len(v) for v in pool.values()), default=0)
        if max_len == 0:
            print(f"  [skip] {ds}: empty candidate pool", file=sys.stderr)
            continue
        print(f"\n=== {ds}: {len(instances)} instances, "
              f"pool depth up to {max_len} ===")

        ds_engine = "react" if ds == "toolqa" else engine

        for pos in range(1, max_len + 1):
            if positions and not positions[0] <= pos <= positions[1]:
                continue
            subset_path = _cached_position_instances(
                cache_dir, ds, pos, instances, pool)
            if subset_path is None:
                continue
            label = f"pos{pos}{label_suffix}"

            for run in run_ids:
                run_ws = workspace / model_name / f"run{run}"
                infer_path = run_ws / "inference" / ds / model_name / f"{label}.jsonl"
                eval_path = run_ws / "eval" / ds / model_name / f"{label}.json"
                if eval_path.exists():
                    continue

                cmd = [
                    sys.executable, "-m", "sragents.cli.main", "infer",
                    "--instances", str(subset_path),
                    "--output", str(infer_path),
                    "--model", model,
                    "--api-base", api_base,
                    "--provider", "topk",
                    "--provider-arg", f"source={annotation_path}",
                    "--provider-arg", "k=1",
                    "--provider-arg", f"start={pos - 1}",
                    "--engine", ds_engine,
                    "--label", label,
                    "--workers", str(workers),
                    "--temperature", str(temperature),
                    "--max-tokens", str(max_tokens),
                ]
                cmd += ["--corpus", str(corpus_path)]
                if thinking:
                    cmd.append("--thinking")

                infer_rc = _run(cmd)
                if infer_rc != 0:
                    print(f"  [warn] infer failed for {ds}/{label}/run{run} "
                          f"(exit={infer_rc}); skipping evaluate", file=sys.stderr)
                    continue

                if infer_path.exists() and not eval_path.exists():
                    eval_cmd = [
                        sys.executable, "-m", "sragents.cli.main", "evaluate",
                        "--input", str(infer_path),
                        "--instances", str(instances_file),
                        "--output", str(eval_path),
                        "--workers", str(eval_workers),
                    ]
                    if ds == "bigcodebench":
                        with _bigcodebench_eval_lock(workspace):
                            _run(eval_cmd)
                    else:
                        _run(eval_cmd)
