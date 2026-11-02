# DeepSkill

Code for *DeepSkill: Evaluating Agent Skill Retrieval Beyond Reference Skills*.

The labels are on Hugging Face at
[jdh235/DeepSkill](https://huggingface.co/datasets/jdh235/DeepSkill): graded relevance
judgments (0–3) and quality flags for 2,275 task–skill pairs from SRA-Bench and Skill-Usage,
with the queries. This repository holds the code that built and analyses them, and the LLM-judge
labels (gpt-oss-120b) for the same pairs, in
[`resource/labels/llm_judge/`](resource/labels/llm_judge/gpt-oss-120b).

| Folder | What it holds |
|---|---|
| [`resource/`](resource) | The 12 skill retrievers, the judgment pools, the task sample, the labels, and the scripts behind the paper's tables and figures |
| [`annotator/`](annotator) | The annotation app (both rounds) and the LLM judge |
| [`utility/sra-bench/`](utility/sra-bench) | Runs the agents on the SRA-Bench tasks: no skill, each pooled skill on its own, and the consensus top-5, using [SR-Agents](https://github.com/oneal2000/SR-Agents) |
| [`utility/skill-usage/`](utility/skill-usage) | The same for the Skill-Usage tasks, in Harbor |

Each folder has its own README and environment.

## How the parts connect

1. `resource/` runs the 12 retrievers on both benchmarks, pools their top 5s and writes the
   annotation files to `resource/annotation/top5/task_subset/`.
2. `annotator/` shows those pools to the annotators and the LLM judge.
3. `resource/labels/` builds the final labels from the annotators' exports.
4. `utility/` runs the agents with the judged skills. The runs are on Hugging Face at
   [jdh235/DeepSkill-agent-runs](https://huggingface.co/datasets/jdh235/DeepSkill-agent-runs).
5. `resource/` turns the runs into utility labels and builds the paper's tables and figures.

Both `utility/` folders read the judgment pools from `resource/annotation/top5/task_subset/`.
`annotator/` keeps its own copy, which the app saves grades into; it also lists, in each task's
pool, the reference skills no retriever returned, so that every reference skill was judged. The
agents ran only the pooled skills.

## Data

SRA-Bench and Skill-Usage are not redistributed here. Download them once into `resource/data/`,
as in [resource/README.md](resource/README.md#data); every folder reads them from there. The one
exception is the LLM-judge files in `resource/labels/llm_judge/`: they hold each judge prompt, so
they include the task and skill text of the judged pairs, which remains under its original terms.

## Acknowledgements

DeepSkill builds on two existing skill retrieval benchmarks. We thank their authors for releasing their data and code:

- **SRA-Bench** — Su et al., *Skill Retrieval Augmentation for Agentic AI*.
  [Code](https://github.com/oneal2000/SR-Agents) ·
  [Dataset](https://huggingface.co/datasets/WeihangSu/SRA-Bench) ·
  [Paper](https://arxiv.org/abs/2604.24594)
- **Skill-Usage** — Liu et al., *How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings*.
  [Code](https://github.com/UCSB-NLP-Chang/Skill-Usage) ·
  [Paper](https://arxiv.org/abs/2604.04323)

Skill-Usage tasks and reference skills originate from
[SkillsBench](https://arxiv.org/abs/2602.12670) (Li et al.).
