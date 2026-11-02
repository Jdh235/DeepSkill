# Annotator

Local web app for rating retrieved-skill relevance (0-3 scale) against each
task in `task_subset/*-annotation.json`, one skill at a time. See `app.py`'s
module docstring for how the annotation flow works. The second round, for
re-judging disagreements, is under `reannotation/` (see
[Re-annotation](#re-annotation-round-2) below). The LLM judge, which labels
the same pairs with the same rubric, is under `llm_judge/` (see
`llm_judge/README.md`).

This folder works with [`../resource/`](../resource):

- **The annotation subset** (`task_subset/`) holds the files
  `resource/annotation/generate_annotation.py` writes to
  `resource/annotation/top5/task_subset/`, with each task's
  `gold_not_retrieved` skills appended to its `retrieved` list (the app does
  this when it loads a file): the
  75 tasks listed in `resource/annotation/tasks.txt` (10 per SRA-Bench dataset, 25
  from Skill-Usage), each with the union of 12 retrievers' top-5 skills,
  plus any gold skills none of them retrieved, 2,275 task–skill pairs in
  total.
- **The final relevance labels** are there too, in
  `resource/labels/human/final/qrels/<dataset>-relevance.qrels`, built by
  `resource/labels/human/generate_final_labels.py` from this folder's annotations: the round-2
  grade for each re-annotated pair, otherwise the median of the three
  round-1 grades.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The apps and the LLM judge read the benchmarks from `DATA_DIR`, by default
`../resource/data/`, where they are downloaded once (see
[resource/README.md](../resource/README.md#data)). To set any of the four
locations separately, use `INSTANCES_DIR`, `CORPUS_PATH`,
`SKILL_USAGE_TASKS_DIR` or `SKILL_USAGE_CORPUS_DIR`.

### Seed (one-time)

The app walks skills in a randomised order (which dataset comes first, task
order within each dataset, skill order within each task) so retriever
rank/gold status can't leak into the judgment. Order is controlled by a seed, which you must run once only:

```bash
python3 set_seed.py
```

This writes `annotation_seed.txt` (a single integer) to the project root,
which `app.py` reads on every startup. `annotation_seed.txt` is gitignored
(not committed) -- each checkout generates and keeps its own, so `app.py`
refuses to start until you've run this once. Run it exactly once per
checkout, before annotating starts: running it again later reshuffles the
walk order out from under anyone partway through, so it refuses to
overwrite an existing file unless you pass `--force`.

## Run

```bash
uvicorn app:app --reload
open http://localhost:8000
```

Set `DATA_DIR` (or the individual paths) as described under
[Data](#data). `ANNOTATION_DIR` points the app at a different annotation
subset than `task_subset/`.

Alongside the 0-3 relevance rating, each skill also has three optional
checkboxes -- "Malicious", "Incorrect information" and "Not actionable in
this environment" -- saved separately to `task_subset/flags.json`, keyed by
skill (not by task), so flagging a skill once carries over to every task it
appears in.

The app saves ratings back into `task_subset/` in place, which is why this
folder keeps its own copy of the subset rather than reading resource's. Each annotator's completed round-1 files are in
`reannotation/round1/task_subset_N/`.

## Re-annotation (round 2)

After round 1, two apps let each annotator re-judge just the items the
annotators disagreed on. They use the same rubric, flag definitions, dataset
notes and look as `app.py` (all shared from `common.py`), read the same
`data/` and `tasks/` (so there's nothing extra to download), and show each
annotator's round-1 answer for reference.

| App | Re-judges | Items | Port |
|---|---|---|---|
| `reannotation/relevance/` | relevance (0–3) of task–skill pairs rated more than one step apart (highest − lowest ≥ 2); flags shown read-only | 331 pairs | 8001 |
| `reannotation/flags/` | flags the annotators didn't all agree on: only the disputed flag is shown, with an explicit Yes/No (no default) | 239 flags (238 skills) | 8002 |

```
reannotation/
├── round1/task_subset_{1,2,3}/   # every annotator's round-1 results (annotator N = task_subset_N)
├── round2/task_subset_{1,2,3}/   # every annotator's re-judged flags.json from the flags app
├── relevance/
│   ├── app.py  build_set.py  templates/
│   └── data/task_subset/         # the pairs to re-rate + your answers
└── flags/
    ├── app.py  build_set.py  templates/
    └── data/units.json, flags.json   # the flags to re-judge + your answers
```

### Setup

With round 1's setup done (venv and data above), pull the latest and set a
seed for each app, once. Each annotator gets their own random order (dataset,
then task or flag type, then skill), fixed across restarts:

```bash
git pull
python3 set_seed.py --app relevance
python3 set_seed.py --app flags
```

These write `reannotation/relevance/annotation_seed.txt` and
`reannotation/flags/annotation_seed.txt` (gitignored, like the round-1
seed). The apps refuse to start without them.

### Run

From `annotator/`:

```bash
uvicorn reannotation.relevance.app:app --reload --port 8001   # http://localhost:8001
uvicorn reannotation.flags.app:app --reload --port 8002       # http://localhost:8002
```

Click **Start labelling**. Each **Save & next** is written to disk straight
away, and **Start labelling** always resumes at the first item you haven't
done, so you can stop and pick up later. The two apps can run at the same time.

### Where your answers are saved

- **Relevance:** `reannotation/relevance/data/task_subset/<dataset>-annotation.json`,
  each pair's new rating in `annotations.annotator_1` (`null` until re-rated).
  This app re-rates relevance only: each skill's flags are shown read-only,
  per annotator, from `reannotation/round2/task_subset_N/flags.json` where
  annotator N re-judged that flag, else from their
  `reannotation/round1/task_subset_N/flags.json`.
- **Flags:** `reannotation/flags/data/flags.json`, with an explicit true/false
  for every flag of each skill: the one you re-judged, plus the value everyone
  already agreed on for the others.

The round-1 answers shown on each page come from `reannotation/round1/` and are
never modified.

### Rebuilding the sets

The sets are already committed, so annotators never need this. To regenerate
them (e.g. after round 1 changes), run `python3 reannotation/relevance/build_set.py`
or `python3 reannotation/flags/build_set.py`. Both refuse to overwrite an
existing set without `--force`, which **wipes any saved re-annotations**.
