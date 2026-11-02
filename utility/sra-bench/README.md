# Skill usage on SRA-Bench

This folder runs the agents on the 50 annotated SRA-Bench tasks: with no skill, with each skill in
the task's judgment pool on its own, and with the consensus top-5. The code is adapted from
[SR-Agents](https://github.com/oneal2000/SR-Agents) (Su et al.; MIT, see [LICENSE](LICENSE)), so the
package and command are still named `sragents`.

## Install

Requires Python 3.10 – 3.12.

```bash
pip install -e .       # or: uv sync
```

SRA-Bench is read from `../../resource/data/sra_bench/`, where it is
downloaded once for the whole repository (see
[resource/README.md](../../resource/README.md#data)).

Inference requires an OpenAI-compatible chat endpoint. Point
`--api-base` at any compatible server (OpenAI, vLLM, SGLang, Ollama, …).
`--model` is the served model identifier — a model ID like
`gpt-4o-mini` for hosted APIs, or whatever string the local server is
serving (often a path like `/models/Qwen3-32B`) for vLLM/SGLang. For
endpoints that require auth, set `OPENAI_API_KEY`; local unauthenticated
servers accept any value.

### Serving the models

Our results were produced with each model served by vLLM:

```bash
# Llama-3.1-8B-Instruct  (BF16, context length 131,072)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --enable-prefix-caching

# Qwen3.6-27B-FP8        (FP8, context length 262,144)
vllm serve Qwen/Qwen3.6-27B-FP8 --port 8000 --enable-prefix-caching

# GLM-4.5-Air-FP8        (FP8, context length 131,072)
vllm serve zai-org/GLM-4.5-Air-FP8 --port 8000 --enable-prefix-caching
```

Then use `--api-base http://localhost:8000/v1` and pass the same model
name to `--model`. Thinking is turned off for Qwen and GLM by the
client, which sends `chat_template_kwargs: {"enable_thinking": false}`
with every request; this is a vLLM/SGLang option, so an endpoint that
ignores it will run the model with thinking on. The client decides this
from the model name (it looks for `qwen3` or `glm-4.5` in `--model`), so
serve the models under the names above.


| Method | Provider | Engine | Description |
|---|---|---|---|
| **LLM Direct** | `none` | `direct` | No skill — parametric-only baseline |
| **Oracle** | `oracle` | `direct_full` | The task's gold skill(s) |
| **Direct injection, top-5** | `topk(k=5)` | `direct_full` | The first 5 skills of the task's top-5 pool, injected together |
| **Progressive disclosure, top-5** | `topk(k=5)` | `progressive_disclosure` | The same 5 skills as a catalog; the model loads skills on demand |
| **Candidate sweep** | `topk(k=1)` per pool position, via `sragents sweep` | `direct_full` | Every skill in the task's top-5 pool, injected one at a time |

The oracle, direct injection and the candidate sweep use `direct_full`, which injects each skill as a
full record: name, description and content (`direct` injects the content only). Progressive disclosure
shows each skill's name and description in a catalog and gives its content when the model loads it.


## Candidate-pool sweep

`sragents sweep` injects candidate skills **one at a time**: given a
per-dataset *candidate pool* — e.g. a union of several retrievers'
results, with one annotation file per dataset — it runs every
`(instance, candidate skill)` pair through `infer` + `evaluate`
individually, repeated `--runs` times. Pool depth is
**instance-specific**: a union pool has no uniform size, so the
sweep reads each instance's own candidate count from the annotation
file rather than assuming a fixed top-K.

Each file must be named `{dataset}-annotation.json` or
`{dataset}-pool.json` and have the shape `{"results": [{"instance_id",
"retrieved": [{"skill_id", ...}, ...]}, ...]}` (extra fields such as
`gold_skill_ids` are ignored):

```bash
sragents sweep \
    --annotation-dir /path/to/candidate_pool/full_annotation \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --api-base http://localhost:8000/v1 \
    --workspace results/candidate_sweep \
    --runs 3
```

Datasets default to every such file found in
`--annotation-dir` (narrow with `--dataset`); use `--annotation
dataset=path` to point at a file that doesn't follow the naming
convention. Output lands at
`{workspace}/{model}/run{N}/eval/{dataset}/{model}/pos{K}.json`
(and the matching `inference/.../pos{K}.jsonl`), where `pos{K}` is
the candidate's position in that instance's pool — the actual skill
tested is always recoverable from each record's `skill_ids_used`,
regardless of pool ordering. As with the other stages, re-invoking
the same `--workspace` resumes: cells whose eval output already
exists are skipped.

## Reproducing the top-5 subset results

Our results (written to `results/top5_subset50/`; released as `SRA-Bench/top5_subset50/` in
[jdh235/DeepSkill-agent-runs](https://huggingface.co/datasets/jdh235/DeepSkill-agent-runs)) cover the 50 SRA-Bench tasks of
the annotation subset (10 per dataset), with candidate skills from each
task's top-5 union pool. The tasks and their candidates are in the
SRA-Bench files in `resource/annotation/top5/task_subset/`. Set up as in
[Install](#install), then:

```bash
MODEL=<MODEL>
API_BASE=<API_BASE>
MODEL_NAME=$(basename $MODEL)
DATASETS="theoremqa logicbench champ medcalcbench bigcodebench"
TASKS=../../resource/annotation/top5/task_subset   # the 50 tasks and their top-5 pools
BENCH=../../resource/data/sra_bench         # SRA-Bench (see Install)
CORPUS=$BENCH/corpus/corpus.json            # the skill corpus
OUT=results/reproduce
```

All methods use temperature 0.7, at most 4096 output tokens, 3
independent runs, and 12 concurrent requests per process. Each command
below writes the same layout as the matching folder in
`results/top5_subset50/`, and resumes if re-run: finished files are
skipped.

**Candidate sweep** (`candidate_sweep/`). Every skill in each task's
top-5 pool, injected one at a time:

```bash
sragents sweep \
    --annotation-dir $TASKS \
    --engine direct_full \
    --model $MODEL --api-base $API_BASE --corpus $CORPUS \
    --workspace $OUT/candidate_sweep \
    --dataset $DATASETS --runs 3 \
    --temperature 0.7 --max-tokens 4096 --workers 12
```

`pos<K>` in the output is the K-th skill of the task's top-5 list. The
sweep also writes a `_pool/` folder (the task list for each position,
used while it runs); it holds no results and is not included in
`results/top5_subset50/`.

**LLM Direct** (`baselines/`). No skill:

```bash
for run in 1 2 3; do
    sragents experiment --exp main --methods llm_direct \
        --dataset $DATASETS --tasks-from $TASKS \
        --model $MODEL --api-base $API_BASE --corpus $CORPUS \
        --workspace $OUT/baselines/$MODEL_NAME/run$run \
        --temperature 0.7 --max-tokens 4096 --workers 12 --eval-workers 12
done
```

**Oracle** (`oracle/`). The task's gold skill(s):

```bash
for run in 1 2 3; do
    sragents experiment --exp full_skill_injection --methods oracle_skill_full \
        --dataset $DATASETS --tasks-from $TASKS \
        --model $MODEL --api-base $API_BASE --corpus $CORPUS \
        --workspace $OUT/oracle/$MODEL_NAME/run$run \
        --temperature 0.7 --max-tokens 4096 --workers 12 --eval-workers 12
done
```

**Direct injection and progressive disclosure, top-5**
(`topk5_comparison/<model>/{direct_injection_top5,progressive_disclosure_top5}/`).
The first 5 skills of each task's top-5 pool, either injected together
(`direct_full`) or offered as a catalog the model loads from
(`progressive_disclosure`):

```bash
for method in direct_injection_top5:direct_full \
              progressive_disclosure_top5:progressive_disclosure; do
    name=${method%%:*}; engine=${method##*:}
    for run in 1 2 3; do
        dir=$OUT/topk5_comparison/$MODEL_NAME/$name/run$run
        for ds in $DATASETS; do
            sragents infer \
                --instances $BENCH/instances/$ds.json --tasks-from $TASKS \
                --output $dir/inference/$ds/$MODEL_NAME/result.jsonl \
                --model $MODEL --api-base $API_BASE --corpus $CORPUS \
                --provider topk \
                  --provider-arg source=$TASKS/$ds-annotation.json \
                  --provider-arg k=5 \
                --engine $engine --label result \
                --temperature 0.7 --max-tokens 4096 --workers 12
            sragents evaluate \
                --input $dir/inference/$ds/$MODEL_NAME/result.jsonl \
                --instances $BENCH/instances/$ds.json \
                --output $dir/eval/$ds/$MODEL_NAME/result.json \
                --workers 12
        done
    done
done
```

`--tasks-from $TASKS` limits `infer` and `experiment` to the tasks listed
in `$TASKS` -- the same 50 tasks the sweep takes from `--annotation-dir`.

## License

See [LICENSE](LICENSE).
