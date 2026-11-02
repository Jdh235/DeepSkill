# LLM judge

Labels every task–skill pair in `task_subset/*-annotation.json` with an LLM,
using the same rubric, flag definitions, dataset notes, task and skill content
as the human annotators. For each pair the judge returns a relevance grade
(0–3) and the three quality flags (malicious, incorrect information, not
actionable).

It works with **any OpenAI-compatible chat completions API**: a local server
such as vLLM or llama.cpp, or a hosted provider. Point it at the server and
give the model name.

| File | What it does |
|---|---|
| `prompt_builder.py` | Builds the judge prompt for one pair, and can print it |
| `run_judge.py` | Runs the judge over every pair and saves the results |
| `test_tool_call.py` | Checks that a server/model can use the judge's `read_file` tool |
| `results/` | Where runs are saved. The run reported in the paper is in `resource/labels/llm_judge/gpt-oss-120b/` |

## Setup

From the `annotator/` folder:

```bash
pip install -r requirements.txt -r llm_judge/requirements.txt
```

The judge reads the task and skill data from the same place as the annotation
apps; see [Data](../README.md#data).

## Connecting to an API

These options decide where requests go:

| Option | Meaning |
|---|---|
| `--base-url` | The API root, ending in `/v1` (default `http://localhost:8000/v1`) |
| `--port` | Shorthand for `--base-url http://localhost:PORT/v1` |
| `--api-key` | Defaults to `$OPENAI_API_KEY`; local servers usually accept any value |
| `--model` | The model name exactly as the API serves it |

To find the model name a server uses:

```bash
curl http://localhost:8000/v1/models
```

### Examples

A local server on port 8000:

```bash
python llm_judge/run_judge.py --port 8000 --model openai/gpt-oss-120b
```

A hosted provider:

```bash
export OPENAI_API_KEY=...
python llm_judge/run_judge.py --base-url https://<provider>/v1 --model <model-name>
```

### What the server needs to support

- **Chat completions** (`POST /v1/chat/completions`).
- **Tool calling**, for the Skill-Usage pairs. Their tasks and skills come
  with bundled files, which the judge opens with a `read_file` tool (see
  below). The other five datasets have no files and send no tools. With
  vLLM, tool calling has to be switched on when starting the server (e.g.
  `--enable-auto-tool-choice --tool-call-parser <parser>`); llama.cpp's
  `llama-server` needs `--jinja`. Run `test_tool_call.py` to check.
- **Reasoning output (optional).** If the server returns a model's thinking
  separately (vLLM's `--reasoning-parser`, llama.cpp's
  `--reasoning-format auto`), it's saved with each result. Otherwise only
  the final answer is.

### Settings used for the paper's run

The run in `results/` used **gpt-oss-120b** in its native MXFP4 weights
(the `ggml-org/gpt-oss-120b-GGUF` repo, file `gpt-oss-120b-MXFP4.gguf`),
served by llama.cpp's `llama-server` on one GPU, started with:

```bash
llama-server -m gpt-oss-120b-MXFP4.gguf \
    --alias openai/gpt-oss-120b \
    --host 127.0.0.1 --port 8000 \
    -ngl 999 \
    -c 262144 -np 8 --kv-unified \
    -fa on \
    --jinja --reasoning-format auto
```

The judge was then run at high reasoning effort, with gpt-oss's
recommended sampling (temperature 1.0, top-p 1.0, top-k off, min-p 0) and a
16,384-token budget per request, one judgment per pair:

```bash
python llm_judge/run_judge.py --port 8000 --model openai/gpt-oss-120b \
    --reasoning-effort high --run-name gpt-oss-120b-filenote-high
```

vLLM can serve gpt-oss too (`--reasoning-parser openai_gptoss
--enable-auto-tool-choice --tool-call-parser openai`); we used llama.cpp
only because vLLM ran our GPU out of memory loading the MXFP4 weights.

### Checking tool calling

```bash
python llm_judge/test_tool_call.py --port 8000 --model <model-name>
```

This asks the model to open a real skill script with `read_file`, returns
the file exactly as a judging run would, and checks the model answers with
a name found only in that file. It prints `PASS` (exit code 0), or `FAIL`
with the step that went wrong (exit code 1).

## Sampling

`run_judge.py` sends each model card's recommended sampling settings,
looked up by model name in `SAMPLING_BY_MODEL`:

| Model name starts with | Settings |
|---|---|
| `openai/gpt-oss` | temperature 1.0, top-p 1.0, top-k 0, min-p 0, reasoning effort medium |

Any other model gets no sampling parameters, so the server's defaults
apply, and the script says so when it starts. To use specific settings for
another model, add an entry to `SAMPLING_BY_MODEL`.

The top-k, min-p, repetition-penalty and reasoning-effort settings are sent
as extra request fields, which vLLM and llama.cpp accept. A hosted provider
that rejects unknown fields needs its own entry.

Other options:

| Option | Default | Meaning |
|---|---|---|
| `--reasoning-effort` | medium | gpt-oss only: `low`, `medium` or `high` |
| `--max-tokens` | 16384 | Output budget per request (thinking plus answer) |
| `--concurrency` | 8 | Requests sent in parallel |
| `--dataset` | all | Judge only this dataset (repeatable) |
| `--limit` | none | At most this many new pairs per dataset, for a quick test |
| `--run-name` | model name | Results subfolder |
| `--results-dir` | `llm_judge/results/` | Where run folders are written |

## The prompt

To print the full prompt for a pair:

```bash
python llm_judge/prompt_builder.py --dataset theoremqa --instance-id theoremqa_00087 --skill-id theoremqa_205
python llm_judge/prompt_builder.py --dataset skill_usage    # first pair in the dataset
```

The prompt contains, in the order of the annotators' page: the annotators'
instructions, the dataset description, the task (in `<task>` tags), the
candidate skill (in `<skill>` tags), the flag definitions with the
dataset's environment note, the rubric, and the required output format: a
JSON object with a one-sentence `reasoning`, the `rating` and the three
flags.

### Bundled files (Skill-Usage)

Skill-Usage tasks and skills come with bundled files (data, scripts,
references) that annotators could open but weren't shown by default. They
are too large to include in the prompt, so the prompt lists their names and
the judge can open any listed file with a `read_file` tool, in a loop: at
each step it either opens a file, whose contents are returned to it, or
gives its final answer. Limits:

- at most 5 files per judgment, after which tools are switched off and the
  judge has to answer;
- at most 20,000 characters per file (longer files are truncated), and
  binary files aren't shown;
- only the pair's own listed files can be opened. The task's
  `environment/skills/` folder (copies of the correct skills) and Docker
  files are never listed.

## Results

Each run writes one JSON line per pair to
`results/<run name>/<dataset>.jsonl`, with:

- the parsed judgment: `rating`, `malicious`, `incorrect_info`,
  `not_actionable` and the one-sentence `reasoning`;
- the full `trajectory`: the prompt, each model turn (thinking, tool calls,
  finish reason, token counts) and each file returned by `read_file`;
- `files_opened`, the sampling settings, the tool definition and token usage.

Re-running the same command skips pairs already judged, so an interrupted
run picks up where it left off. A pair whose answer couldn't be parsed is
saved with `error` set and judged again on the next run; when reading
results, use each pair's row without an error.

### The run in `results/`

`results/gpt-oss-120b-filenote-high/` is the judge run reported in the
paper: gpt-oss-120b (MXFP4 weights) at high reasoning effort, with the
current prompt, over all 2,275 pairs. The exact prompt of every judgment is
saved in its `trajectory`.

To reproduce it, see [Settings used for the paper's run](#settings-used-for-the-papers-run).
