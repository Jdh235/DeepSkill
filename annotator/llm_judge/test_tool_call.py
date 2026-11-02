"""Check the served model can complete a read_file round trip: call the tool
on a real SkillsBench skill file (served as run_judge.py does) and answer from
its contents. Exits 0 on success, 1 on failure.

Run:
    python llm_judge/test_tool_call.py --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from openai import OpenAI

import prompt_builder as pb
from run_judge import READ_FILE_TOOL, sampling_for

INSTANCE_ID = "3d-scan-calc"
SKILL_ID = "benchflow-ai--mesh-analysis"
SKILL_FILE = "scripts/mesh_tool.py"
EXPECTED = "MeshAnalyzer"

PROMPT = (
    f"Use the read_file tool to open the skill file {SKILL_FILE} (kind \"skill\"). "
    "Then reply with only the name of the class defined in that file."
)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API root, ending in /v1")
    parser.add_argument("--port", type=int, help="shorthand for --base-url http://localhost:PORT/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        help="defaults to $OPENAI_API_KEY; local servers usually accept any value")
    parser.add_argument("--model", default="openai/gpt-oss-120b",
                        help="model name exactly as the API serves it (see GET /v1/models)")
    args = parser.parse_args()
    if args.port:
        args.base_url = f"http://localhost:{args.port}/v1"

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)
    sampling, sampling_extra = sampling_for(args.model)
    messages = [{"role": "user", "content": PROMPT}]

    def ask():
        return client.chat.completions.create(
            model=args.model, messages=messages, tools=[READ_FILE_TOOL], tool_choice="auto",
            max_tokens=4096, extra_body=sampling_extra, **sampling,
        ).choices[0]

    # 1. The model should call read_file.
    choice = ask()
    calls = choice.message.tool_calls or []
    print(f"1. finish_reason={choice.finish_reason}, tool calls={len(calls)}")
    if not calls:
        fail(f"no tool call; content was {choice.message.content!r}")
    call = calls[0]
    print(f"   {call.function.name}({call.function.arguments})")
    if call.function.name != "read_file":
        fail(f"called {call.function.name!r} instead of read_file")
    try:
        call_args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        fail(f"arguments aren't valid JSON: {call.function.arguments!r}")

    # 2. Return the file, as run_judge.py would.
    result = pb.read_bundled_file("skill_usage", INSTANCE_ID, SKILL_ID, call_args.get("kind", ""), call_args.get("path", ""))
    print(f"2. tool returned {len(result):,} characters: {result[:70]!r}...")
    if result.startswith("Error"):
        fail(f"tool call didn't resolve to a file: {result}")
    messages.append({"role": "assistant", "content": choice.message.content or "",
                     "tool_calls": [tc.model_dump() for tc in calls]})
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # 3. The model should answer from the file's contents.
    choice = ask()
    answer = (choice.message.content or "").strip()
    print(f"3. finish_reason={choice.finish_reason}, answer={answer!r}")
    if EXPECTED not in answer:
        fail(f"answer doesn't name {EXPECTED}, so the file contents didn't get through")

    print(f"PASS: {args.model} called read_file and answered from the file.")


if __name__ == "__main__":
    main()
