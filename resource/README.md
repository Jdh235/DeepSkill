# DeepSkill: Evaluating Agent Skill Retrieval Beyond Reference Skills

Graded relevance labels and quality flags for agent skill retrieval on
**SRA-Bench** (50 tasks, 26,262-skill corpus) and **Skill-Usage** (25 tasks,
34,396-skill corpus). Each task's judgment pool is the union of 12 retrievers'
top 5, plus any reference skill none of them returned. Three annotators graded
every task–skill pair 0–3 and flagged skills that are malicious, contain
incorrect information or are not actionable; pairs they disagreed on were
re-annotated. Alongside the human labels are LLM-judge labels (gpt-oss-120b)
and utility labels: how agent models do on each task with each pooled skill
injected, against a no-skill baseline. The 24 reference skills no retriever
returned were not run.

This folder holds the labels, the pools, the 12 retrievers, the task
selection and the scripts behind every table and figure in the paper. The
agent runs and the annotation tool are in the other folders (see
[Related folders](#related-folders)).

| | SRA-Bench | Skill-Usage |
|---|---|---|
| Tasks | 50 (10 per sub-dataset) | 25 |
| Judged task–skill pairs | 1,525 | 750 |
| Utility models | GLM-4.5-Air-FP8, Llama-3.1-8B-Instruct, Qwen3.6-27B-FP8 | GLM-4.5-Air-FP8, Qwen3.6-27B-FP8 |

The two benchmarks are always reported separately.

## Using the labels

The final labels are standard TREC qrels, one file per dataset, so
`ir_measures`, `trec_eval` and `pt.io.read_qrels` read them directly:

```python
import ir_measures
from ir_measures import nDCG, P, R, Judged

qrels = list(ir_measures.read_trec_qrels("labels/human/final/qrels/champ-relevance.qrels"))
run = list(ir_measures.read_trec_run("my_run.trec"))
ir_measures.calc_aggregate([nDCG@5, P(rel=2)@5, R(rel=2)@5, Judged@5], qrels, run)
```

Use `rel=2` for binary measures: a skill counts as relevant when it is graded
2 or 3. Only the top 5 of the 12 pool retrievers was judged, so measure at
depth 5 and report `Judged@5`; unjudged skills count as non-relevant.

`results/subset/` has the 12 pool retrievers' ranked lists (top 50) for the
75 annotated tasks, as ready-made baselines. The metrics stored inside those
files are computed against the benchmarks' original reference skills, not
these labels.

### Files

| Path | Contents |
|---|---|
| `labels/human/final/qrels/{dataset}-relevance.qrels` | Final grade per pair: `query_id 0 skill_id grade` (0–3) |
| `labels/human/final/flags/{dataset}-flags.tsv` | Final grade and consensus quality flags per pair (tab-separated, header row; flag 1 = flagged) |
| `labels/human/first_round/task_subset_{1,2,3}/` | Each annotator's first-round grades (`{dataset}-annotation.json`) and flags (`flags.json`) |
| `labels/human/reannotation/` | Second-round grades for re-annotated pairs, and second-round flags (`flags/round2/`) |
| `labels/llm_judge/gpt-oss-120b/{dataset}.jsonl` | One line per pair: the judge's full trajectory, sampling settings, and its answer in `content` (`rating`, `malicious`, `incorrect_info`, `not_actionable`, `reasoning`) |
| `labels/utility/{sra_bench,skill_usage}/{model}/{dataset}-utility.json` | Per task: the baseline's rewards over three runs, and each pooled skill's rewards with that skill injected |
| `annotation/tasks.txt` | The 75 annotated task IDs |
| `annotation/top5/task_subset/{dataset}-annotation.json` | The judgment pools as given to annotators |
| `pool/top{5,10,20}/{dataset}-pool.json` | Union pools at depths 5, 10 and 20 for every benchmark task; top 5 is the judgment pool |
| `results/{sra_bench,skill_usage}/` | The 12 pool retrievers' runs (top 50) on every benchmark task, used by the pools and the retriever ranking table |
| `results/subset/` | The 12 pool retrievers' runs on the 75 annotated tasks |

**Final grade.** The re-annotation grade if the pair was re-annotated (its
first-round grades differed by two or more), otherwise the median of the three
first-round grades.

**Final flags.** Flags are given per skill, so a skill's flags repeat on each
of its pairs. A flag any annotator marked in the first round was re-annotated
by all three, and is final if two of the three second-round answers mark it;
any other flag is final as not flagged.

In the annotation and pool files, `is_gold` marks a benchmark's reference
skills, and `gold_not_retrieved` lists reference skills that no pool retriever
returned (added to the pool so every reference skill is judged).

## Setup

```bash
conda env create -f environment.yml
conda activate sragents-pt
pip install -e ".[analysis]"
```

`environment.yml` installs PyTerrier, Java 21, PyTorch and the retriever
libraries. It points pip at the ROCm 7.2 PyTorch wheels, which the paper's runs
used; on NVIDIA GPUs, change that index to the matching CUDA build from
https://download.pytorch.org/whl/. The labels and the analysis only need
`pip install -e ".[analysis]"` in any Python 3.10+ environment.

ColBERTv2 needs its own environment, because `pyterrier_colbert2` requires
`transformers` older than 5, and an NVIDIA GPU, because it compiles a CUDA
kernel the first time it runs:

```bash
conda env create -f environment-colbert.yml
conda activate sragents-colbert
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "colbert-ai @ git+https://github.com/stanford-futuredata/ColBERT.git@cc4f3dc91c0b45d2d08c251d9d95178285c65f1c"
pip install numpy pandas faiss-cpu \
    "git+https://github.com/cmacdonald/pyterrier_colbert2@4947df66c17e43ad1da5df916d39e464a0e0ec35"
pip install --no-deps -e .
conda install -c nvidia cuda-nvcc=12.4.131 cuda-cudart-dev=12.4 cuda-cccl=12.4 \
    cuda-cudart=12.4.127 cuda-version=12.4
```

The conda packages provide the CUDA 12.4 compiler, matching PyTorch's CUDA
version, for that kernel.

## Data

The benchmarks are not redistributed here. Download them once into `resource/data/`, running these
commands from `resource/`; `annotator/` and both `utility/` folders read them from there too:

**SRA-Bench** (corpus and task instances):

```bash
hf download WeihangSu/SRA-Bench --repo-type dataset --local-dir data/sra_bench
```

**Skill-Usage corpus** (34,396 skills: 34,198 scraped plus the 198 reference
skills):

```bash
hf download Shiyu-Lab/Skill-Usage skills-34k/skills.zip skills-34k/skills_meta.jsonl \
    --repo-type dataset --revision bc2a24d21a013cf30596dc758dbf44750e7211f6 --local-dir data/skill_usage
mkdir -p data/skill_usage/corpus/skill_usage
unzip data/skill_usage/skills-34k/skills.zip -d data/skill_usage/corpus/skill_usage/
cp data/skill_usage/skills-34k/skills_meta.jsonl data/skill_usage/corpus/skill_usage/
rm -rf data/skill_usage/skills-34k data/skill_usage/.cache
```

**Skill-Usage tasks** and their reference skills, from the
[Skill-Usage repository](https://github.com/UCSB-NLP-Chang/Skill-Usage) at the
commit we used:

```bash
git clone https://github.com/UCSB-NLP-Chang/Skill-Usage /tmp/skill-usage
git -C /tmp/skill-usage checkout 03446d16f7b659ccc93ac5bd512f62e9b7fabb45
cp -r /tmp/skill-usage/tasks data/skill_usage/tasks
cp /tmp/skill-usage/data/task_skill_mapping.json data/skill_usage/
```

**Agent runs** (no-skill baselines, the single-skill sweep and the consensus
top-5 runs), read by steps 3, 6 and 7 below:

```bash
hf download jdh235/DeepSkill-agent-runs --repo-type dataset --local-dir agent_results
```

## Reproducing the resource

Each step's outputs are already in the repository; rerun a step to check it.
Run everything from `resource/`.

### 1. Retrieval runs

Each retriever is a standalone script in `src/pyterrier_retrieval/retrievers/`:

| Type | Retrievers |
|---|---|
| Lexical | `sparse.bm25`, `sparse.hiemstra_lm` |
| Learned sparse | `sparse.splade` |
| Dense | `dense.bge`, `dense.contriever`, `dense.bge_m3` |
| Late interaction | `dense.colbertv2` (ColBERT environment) |
| LLM embedder | `dense.qwen3_emb_06b`, `dense.qwen3_emb_8b` |
| Skill fine-tuned | `dense.skillret_06b`, `dense.skillret_8b`, `dense.skillrouter_06b` |

Run each on the five SRA-Bench datasets and on Skill-Usage:

```bash
for r in sparse.bm25 sparse.hiemstra_lm sparse.splade dense.bge dense.contriever dense.bge_m3 \
         dense.qwen3_emb_06b dense.qwen3_emb_8b dense.skillret_06b dense.skillret_8b dense.skillrouter_06b; do
  for ds in bigcodebench champ logicbench medcalcbench theoremqa; do
    python -m pyterrier_retrieval.retrievers.$r --dataset sra_bench --instances data/sra_bench/instances/$ds.json
  done
  python -m pyterrier_retrieval.retrievers.$r --dataset skill_usage
done
# then dense.colbertv2 the same way, in the ColBERT environment
```

Runs are written to `results/sra_bench/{dataset}-{retriever}.json` and
`results/skill_usage/{retriever}.json` (top 50 per task), with a copy cut down
to the annotated tasks in `results/subset/`. SRA-Bench queries are the
dataset's task prompt (`prompts.py`); Skill-Usage queries are the task's
`instruction.md`.

All five SRA-Bench datasets share one corpus. Every retriever except BM25 and
Hiemstra LM can build that index once and reuse it: pass the same
`--index-path` each time, and `--no-rebuild-index` after the first dataset. 

Every retriever except BM25 and Hiemstra LM takes `--batch-size`, the number of
skills encoded at once. The defaults are the released runs' batch sizes. Those
runs used AMD MI300X GPUs (192 GB), except SkillRet-0.6B (an AMD MI210) and
ColBERTv2 (an NVIDIA A100).

### 2. Pools

```bash
python pool/generate_pool.py
```

Builds the union of the 12 retrievers' top 5, 10 and 20 for every task into
`pool/top{5,10,20}/`. Within a task, skills are ordered by how many retrievers
return them in their own top k; the paper's top-5k is the first five skills of
the depth-5 pool.

### 3. Task selection

```bash
python annotation/generate_tasks.py
```

Rewrites `annotation/tasks.txt`. SRA-Bench: 10 tasks per dataset, stratified by
evaluation type and reference-skill count, choosing the draw whose no-skill
baseline accuracy is closest to the full dataset's while ranking the models
in the same order (needs the SRA-Bench baseline runs, released with the agent
runs). Skill-Usage: 25 tasks from the 57
in `annotation/skill_usage_population.txt`, the tasks whose reference solution
passed consistently in our environment. The seed is fixed, so the output
matches `tasks.txt` exactly.

### 4. Annotation files

```bash
python annotation/generate_annotation.py
```

Filters the depth-5 pools to the annotated tasks and adds the reference skills
no retriever returned, writing `annotation/top5/task_subset/`. These files were
loaded into the annotation tool.

### 5. Final labels

```bash
python labels/human/generate_final_labels.py
```

Writes `labels/human/final/` (qrels and flags) from the first-round and
re-annotation exports.

### 6. Utility labels

```bash
python labels/utility/generate_utility.py --sra-runs DIR --skill-usage-runs DIR
```

Reads the agent runs (baselines and the single-skill sweep) and writes
`labels/utility/`, used by the single-skill tables and as the no-skill baseline
in the consensus top-5 table. The defaults point to `agent_results/`, the layout of the
released runs.

### 7. Tables and figures

```bash
bash figures/build_all.sh
```

Regenerates the final labels, every table and figure in the paper, and the
retriever ranking table into `figures/output/`. The consensus top-5 table also
reads the agent runs in `agent_results/`.

## Related folders

- **Agent runs** (baselines, single-skill sweep and consensus top-5 runs for both
  benchmarks): [jdh235/DeepSkill-agent-runs](https://huggingface.co/datasets/jdh235/DeepSkill-agent-runs);
  download them as in [Data](#data).
- [`../utility/sra-bench/`](../utility/sra-bench): runs the SRA-Bench agents.
- [`../utility/skill-usage/`](../utility/skill-usage): runs the Skill-Usage agents.
- [`../annotator/`](../annotator): the annotation tool and the LLM judge.

## Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```
