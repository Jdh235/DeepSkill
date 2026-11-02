# Skill usage on SkillsBench: 25-task experiments

This folder reruns our experiments on 25 [SkillsBench](https://github.com/benchflow-ai/skillsbench) tasks. Each task is run under four settings, which differ in which skills the agent gets and how it gets them:

| Setting | What the agent gets | Task folder | Trials per model |
|---|---|---|---|
| `no-skills` | The task only; no skills are shown | `tasks-no-skills` | 75 |
| `candidate-sweep` | One retrieved candidate skill appended to the instruction, once for every candidate in the task's pool (740 task–skill pairs) | `tasks-candidate-sweep` | 2,220 |
| `top5-progressive-disclosure` | The 5 highest-scoring candidates in the task's pool, listed by name and description | `tasks-top5k-progdisc` | 75 |
| `top5-full-injection` | The same 5 candidates' SKILL.md files appended to the instruction | `tasks-top5k-fullinject` | 75 |

Every task is attempted 3 times, so 25 tasks give 75 trials and 740 pairs give 2,220.

The candidate pool for each task is in [`resource/annotation/top5/task_subset/skill_usage-annotation.json`](../../resource/annotation/top5/task_subset/skill_usage-annotation.json) (the `retrieved` list of each entry). It contains 740 candidates, 688 distinct skills, drawn from the 34k-skill corpus of [Skill-Usage](https://github.com/UCSB-NLP-Chang/Skill-Usage). The file's metadata lists the 12 retrievers whose top-5 results were pooled. Its entries also define the 25 tasks.

In the candidate-sweep and top-5 settings, the task's own curated skill is deleted from the container, so the agent sees only the candidates being tested. In `no-skills`, the curated skills are still copied into the container by the original SkillsBench Dockerfiles, but they are never shown to the agent.

We ran two models, `Qwen/Qwen3.6-27B-FP8` and `zai-org/GLM-4.5-Air-FP8`, with the [terminus-2](https://github.com/laude-institute/harbor) agent.

## Requirements

- Linux with Docker. We ran Harbor and the task containers on aarch64 Linux.
- Internet access from inside the containers. The task images install packages when they are built, and 24 of the 25 verifiers download packages when they run.
- Enough CPU and memory for 12 trials at once, the configs' setting. Each task asks for 1–4 CPUs and up to 8 GB of memory, so 12 trials can need about 24 CPUs and 60 GB together. To run fewer at once, see [step 5](#5-run-each-setting).
- Python 3.10 or newer. The build scripts use only the standard library.
- [uv](https://docs.astral.sh/uv/), git and unzip.
- An OpenAI-compatible endpoint serving the model, for example [vLLM](https://github.com/vllm-project/vllm).
- Enough disk space for the task images. The candidate sweep alone builds 25 task images plus a thin layer for each of the 740 pairs.

Run every command below from this folder, `utility/skill-usage/`.

## 1. Install Harbor

```bash
uv tool install --python 3.12 "harbor @ git+https://github.com/yujianll/harbor@5228b49e01470901d7bdab9bbb2e01d7a1b05154"
```

This is the Harbor fork that Skill-Usage uses, `yujianll/harbor`, at the exact commit we ran (Harbor 0.1.45).

## 2. Get the tasks and the skill corpus

The scripts read the SkillsBench tasks and the 34k-skill corpus from `../../resource/data/skill_usage/`, where they are downloaded once for the whole repository (see [resource/README.md](../../resource/README.md#data)).

To check: `ls ../../resource/data/skill_usage/tasks | wc -l` should print 87, and `ls ../../resource/data/skill_usage/corpus/skill_usage | wc -l` should print 34397 (the 34,396 skill folders and `skills_meta.jsonl`).

### Using copies stored elsewhere

If you already have the corpus or the tasks, or want a different candidate pool, skip the matching download and tell the scripts where the files are:

```bash
export SKILL_USAGE_CORPUS_DIR=/path/to/corpus           # folder that directly contains the 34,396 skill folders
export SKILL_USAGE_TASKS_DIR=/path/to/tasks             # folder that directly contains the 87 task folders
export SKILL_USAGE_ANNOTATION=/path/to/annotation.json  # candidate pools, same format as the default pool file
```

Each script also accepts the matching flag (`--corpus-dir`, `--tasks-dir`, `--annotation`), which takes precedence over the variable. If neither is set, the scripts use `../../resource/data/skill_usage/corpus/skill_usage`, `../../resource/data/skill_usage/tasks` and `../../resource/annotation/top5/task_subset/skill_usage-annotation.json`. Only the scripts in steps 3 and 5 read these locations, and they never write to them. The Harbor configs don't refer to them at all; they run on the task folders built in step 3.

The corpus must be a complete unzip of `skills.zip` at the revision in resource/README.md. A copy that has lost files gives different skill folders, for example one committed to git with a `.gitignore` that skips `.env.example` or `__pycache__/`.

An annotation file needs a `results` list with one entry per task: `instance_id` (the task name) and `retrieved`, a list of candidates, each with a `skill_id` (a folder in the corpus) and a `score` (used to pick the top 5).

## 3. Build the task sets

Each setting runs on its own task folder, built from the tasks, the corpus and the annotation file:

```bash
python3 scripts/prepare_no_skills_tasks.py --output-dir tasks-no-skills
python3 scripts/prepare_candidate_sweep.py --output-dir tasks-candidate-sweep
python3 scripts/prepare_top5k_comparison_tasks.py \
    --progdisc-output-dir tasks-top5k-progdisc --fullinject-output-dir tasks-top5k-fullinject
```

To check: each script ends with a `Done.` line, which should report 25 tasks. For the candidate sweep it should read `Done. 740 trials generated in …, 765 docker builds.`

- **Which scripts read what:**
  - `prepare_no_skills_tasks.py` reads the tasks and the annotation file (for the task names).
  - `prepare_candidate_sweep.py` and `prepare_top5k_comparison_tasks.py` read the tasks, the corpus and the annotation file.
- **Task set:** each folder contains exactly the tasks it was built for, and every config runs all the tasks in its folder.
- **Docker images:** the candidate-sweep and top-5 scripts build their own images, tagged `skill-usage-sweep*` and `skill-usage-top5k*`. For the other settings, Harbor builds the images from each task's Dockerfile when the run starts.
- **Build where you run:** the candidate-sweep and top-5 folders refer to local image tags and contain absolute symlinks, so build them on the machine that will run them.

## 4. Serve the model

The configs send requests to an OpenAI-compatible endpoint at `http://localhost:8000/v1`, and read the API key from `OPENAI_API_KEY`. With vLLM:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 --port 8000 --tensor-parallel-size 2 --enable-prefix-caching
# or
vllm serve zai-org/GLM-4.5-Air-FP8 --port 8000 --tensor-parallel-size 2 --enable-prefix-caching

export OPENAI_API_KEY=EMPTY   # any value works if vLLM was started without --api-key
```

We served both models with vLLM 0.28.0 (ROCm build) on 2 × AMD MI355X GPUs. We set no reasoning or tool-call parser, because terminus-2 reads its actions from the response text, so each model's thinking stays in that text. We left `--generation-config` at its default, so vLLM takes its default sampling settings from each model's `generation_config.json` (see [Harness settings](#harness-settings)).

To use another endpoint or model, edit `api_base` and `model_name` in the config. `model_name` follows LiteLLM's naming, so for an OpenAI-compatible server it is `openai/<served model name>`.

## 5. Run each setting

Start the matching model server first. Qwen3.6-27B-FP8:

```bash
harbor run -c configs/Qwen3.6-27B-FP8/no-skills.yaml
harbor run -c configs/Qwen3.6-27B-FP8/candidate-sweep.yaml
harbor run -c configs/Qwen3.6-27B-FP8/top5-progressive-disclosure.yaml
harbor run -c configs/Qwen3.6-27B-FP8/top5-full-injection.yaml
```

GLM-4.5-Air-FP8:

```bash
harbor run -c configs/GLM-4.5-Air-FP8/no-skills.yaml
harbor run -c configs/GLM-4.5-Air-FP8/candidate-sweep.yaml
harbor run -c configs/GLM-4.5-Air-FP8/top5-progressive-disclosure.yaml
harbor run -c configs/GLM-4.5-Air-FP8/top5-full-injection.yaml
```

Each run writes to `results/<setting>/<model>/`. Running the same command again resumes the job and only runs trials that have no result yet.

The configs run 12 trials at once. To run fewer, add `-n <number>` to `harbor run`, and use the same number every time you resume that job: Harbor won't resume a job whose settings have changed.

Harbor counts agent timeouts as errors in its summary. They are a normal outcome here, because the verifier still scores the trial. In our runs, every error Harbor reported was an agent timeout, apart from one GLM candidate-sweep trial whose verifier wrote no reward file.

## Our results

The accuracy of our runs (mean reward over all trials):

| Setting | Qwen3.6-27B-FP8 | GLM-4.5-Air-FP8 |
|---|---|---|
| `no-skills` | 0.126 | 0.090 |
| `candidate-sweep` | 0.242 | 0.107 |
| `top5-progressive-disclosure` | 0.292 | 0.110 |
| `top5-full-injection` | 0.371 | 0.272 |

Some verifiers give partial credit, so accuracy is the mean reward, not the share of trials solved. The models are sampled at temperature 0.7, so a rerun gives close but not identical numbers.

## Harness settings

These are the same for every config:

| | |
|---|---|
| Agent | terminus-2 (Harbor 0.1.45, commit `5228b49`) |
| Temperature | 0.7 (terminus-2's default, sent with every request) |
| Other sampling settings | Qwen3.6-27B-FP8: top_p 0.95 and top_k 20, from its `generation_config.json`. GLM-4.5-Air-FP8: vLLM's defaults, because its `generation_config.json` sets none |
| Thinking | On, the default in both models' chat templates |
| Max output tokens per response | 8,192 |
| Context window given to the agent | 65,536 tokens |
| Attempts per task | 3 |
| Timeouts | task timeouts × 2.5; environment build timeout × 3.0 |
| Concurrent trials | 12 |
| Retries | up to 2, except for agent timeouts and verifier/reward errors |

## Folder layout

```
configs/<model>/<setting>.yaml    Harbor job configs, one per model and setting
scripts/                          task-set builders
```

The following are created by steps 3–5 and are not tracked: `tasks-*/` (built task sets), `results/` and `.local-workspace/`.

## Acknowledgements

The tasks come from [SkillsBench](https://github.com/benchflow-ai/skillsbench). We use the copy distributed with [Skill-Usage](https://github.com/UCSB-NLP-Chang/Skill-Usage), which also provides the 34k-skill corpus. Trials run in [Harbor](https://github.com/laude-institute/harbor).

```bibtex
@misc{liu2026agenticskillsworkwild,
    title={How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings},
    author={Yujian Liu and Jiabao Ji and Li An and Tommi Jaakkola and Yang Zhang and Shiyu Chang},
    year={2026},
    eprint={2604.04323},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2604.04323},
}
```
