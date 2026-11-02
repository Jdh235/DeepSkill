from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

import prompt_builder as pb

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FLAG_KEYS = [key for key, _, _ in pb.FLAG_DEFINITIONS]

SAMPLING_BY_MODEL = {
    "openai/gpt-oss": (dict(temperature=1.0, top_p=1.0),
                       dict(top_k=0, min_p=0.0, chat_template_kwargs={"reasoning_effort": "medium"})),
}


def sampling_for(model: str) -> tuple[dict, dict]:
    """The model's entry in SAMPLING_BY_MODEL, or no sampling params at all
    (so the server's own defaults apply) for any other model."""
    for prefix, sampling in SAMPLING_BY_MODEL.items():
        if model.lower().startswith(prefix):
            return sampling
    return {}, {}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Open one of the files listed under \"Task files\" or \"Skill files\" in the prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["task", "skill"], "description": "\"task\" for a Task file, \"skill\" for a Skill file."},
                "path": {"type": "string", "description": "The file's path exactly as listed in the prompt."},
            },
            "required": ["kind", "path"],
        },
    },
}


def all_datasets() -> list[str]:
    return sorted(p.name.removesuffix("-annotation.json") for p in pb.ANNOTATION_DIR.glob("*-annotation.json"))


def pairs_for(dataset: str) -> list[tuple[str, str]]:
    """(instance_id, skill_id) for every candidate in the dataset,
    including gold skills that app.py appends if they weren't retrieved."""
    out = []
    for result in pb.load_annotation_file(dataset)["results"]:
        skill_ids = [c["skill_id"] for c in result["retrieved"]]
        skill_ids += [s for s in result["gold_skill_ids"] if s not in skill_ids]
        out.extend((result["instance_id"], s) for s in skill_ids)
    return out


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if not row.get("error"):
            done.add((row["instance_id"], row["skill_id"]))
    return done


def parse_judgment(content: str) -> dict:
    """Pull the last JSON object out of the response and check its fields."""
    content = re.sub(
        r'\\(\\|"|u[0-9a-fA-F]{4})|\\',
        lambda m: m.group(0) if m.group(1) else "\\\\",
        content or "",
    )
    decoder = json.JSONDecoder()
    data = None
    for start in reversed([m.start() for m in re.finditer(r"\{", content)]):
        try:
            data, _ = decoder.raw_decode(content, start)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        raise ValueError("no JSON object in response")
    if data.get("rating") not in (0, 1, 2, 3):
        raise ValueError(f"bad rating: {data.get('rating')!r}")
    for key in FLAG_KEYS:
        if not isinstance(data.get(key), bool):
            raise ValueError(f"bad {key}: {data.get(key)!r}")
    return data


def _thinking(message) -> str | None:
    return getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)


def judge_one(client: OpenAI, model: str, dataset: str, instance_id: str, skill_id: str, max_tokens: int,
              reasoning_effort: str | None = None) -> dict:
    """One judgment. For SkillsBench pairs with bundled files, the model can
    call read_file (up to pb.MAX_FILE_READS times) before answering; once
    the limit is hit, tool_choice="none" forces a final answer.
    reasoning_effort (gpt-oss only) overrides the default "medium"."""
    sampling, sampling_extra = sampling_for(model)
    if reasoning_effort:
        sampling_extra = {**sampling_extra, "chat_template_kwargs": {
            **sampling_extra.get("chat_template_kwargs", {}), "reasoning_effort": reasoning_effort}}
    row = {
        "dataset": dataset, "instance_id": instance_id, "skill_id": skill_id, "model": model,
        "sampling": {**sampling, **sampling_extra, "max_tokens": max_tokens},
    }
    trajectory: list[dict] = []
    row["trajectory"] = trajectory
    try:
        messages = pb.build_messages(dataset, instance_id, skill_id)
        trajectory.extend(dict(m) for m in messages)
        task_files, skill_files = pb.load_bundled_files(dataset, instance_id, skill_id)
        tools = [READ_FILE_TOOL] if task_files or skill_files else None
        row["tools"] = tools
        files_opened: list[dict] = []
        thinking: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        while True:
            can_read = tools is not None and len(files_opened) < pb.MAX_FILE_READS
            tool_kwargs = {"tools": tools, "tool_choice": "auto" if can_read else "none"} if tools else {}
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body=sampling_extra,
                **sampling,
                **tool_kwargs,
            )
            choice = response.choices[0]
            message = choice.message
            if _thinking(message):
                thinking.append(_thinking(message))
            if response.usage:
                usage["prompt_tokens"] += response.usage.prompt_tokens
                usage["completion_tokens"] += response.usage.completion_tokens
            trajectory.append({
                "role": "assistant",
                "reasoning": _thinking(message),
                "content": message.content,
                "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                "tool_choice": tool_kwargs.get("tool_choice"),
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            })
            if not (can_read and message.tool_calls):
                break

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls],
            })
            for tc in message.tool_calls:
                if len(files_opened) >= pb.MAX_FILE_READS:
                    result = f"Read limit reached ({pb.MAX_FILE_READS} files). Give your final answer now."
                else:
                    try:
                        call_args = json.loads(tc.function.arguments)
                        kind, path = call_args.get("kind", ""), call_args.get("path", "")
                    except json.JSONDecodeError:
                        kind, path = "", ""
                    result = pb.read_bundled_file(dataset, instance_id, skill_id, kind, path)
                    files_opened.append({"kind": kind, "path": path, "result_chars": len(result)})
                tool_message = {"role": "tool", "tool_call_id": tc.id, "content": result}
                messages.append(tool_message)
                trajectory.append(dict(tool_message))

        row["content"] = message.content
        row["thinking"] = "\n\n---\n\n".join(thinking) or None
        row["files_opened"] = files_opened
        row["finish_reason"] = choice.finish_reason
        row["usage"] = usage
        row["answer_in_thinking"] = False
        try:
            judgment = parse_judgment(message.content)
        except ValueError:
            if not _thinking(message):
                raise
            judgment = parse_judgment(_thinking(message))
            row["answer_in_thinking"] = True
        row["reasoning"] = judgment.get("reasoning")
        row["rating"] = judgment["rating"]
        row.update({key: judgment[key] for key in FLAG_KEYS})
        row["error"] = None
    except Exception as e:  # keep going; the row is retried on the next run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API root, ending in /v1")
    parser.add_argument("--port", type=int, help="shorthand for --base-url http://localhost:PORT/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="defaults to $OPENAI_API_KEY; local servers usually accept any value")
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="model name exactly as the API serves it (see GET /v1/models)")
    parser.add_argument("--dataset", action="append", help="repeatable; defaults to every dataset")
    parser.add_argument("--limit", type=int, help="max new pairs per dataset (for smoke tests)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384, help="thinking + answer")
    parser.add_argument("--run-name", help="results subfolder; defaults to the model name (e.g. set it "
                                           "to keep a run with a changed prompt apart from earlier results)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="where run folders are written (default: llm_judge/results/)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                        help="gpt-oss only; defaults to medium (its model card's default)")
    args = parser.parse_args()
    if args.reasoning_effort and not args.model.lower().startswith("openai/gpt-oss"):
        parser.error("--reasoning-effort only applies to gpt-oss models")
    if args.port:
        args.base_url = f"http://localhost:{args.port}/v1"
    if sampling_for(args.model) == ({}, {}):
        print(f"No sampling settings for {args.model} in SAMPLING_BY_MODEL; using the server's defaults.")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800)
    out_dir = args.results_dir / (args.run_name or args.model.split("/")[-1])
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.dataset or all_datasets():
        out_path = out_dir / f"{dataset}.jsonl"
        done = load_done(out_path)
        todo = [p for p in pairs_for(dataset) if p not in done]
        if args.limit is not None:
            todo = todo[: args.limit]
        print(f"{dataset}: {len(done)} done, {len(todo)} to judge")
        if not todo:
            continue

        lock = threading.Lock()
        errors = 0
        with out_path.open("a") as f, ThreadPoolExecutor(args.concurrency) as pool:
            futures = [
                pool.submit(judge_one, client, args.model, dataset, instance_id, skill_id, args.max_tokens,
                            args.reasoning_effort)
                for instance_id, skill_id in todo
            ]
            for n, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                if row["error"]:
                    errors += 1
                    print(f"  [{n}/{len(todo)}] {row['instance_id']} / {row['skill_id']}: ERROR {row['error']}")
                else:
                    print(f"  [{n}/{len(todo)}] {row['instance_id']} / {row['skill_id']}: rating={row['rating']}")
        print(f"{dataset}: finished, {errors} errors")


if __name__ == "__main__":
    main()
