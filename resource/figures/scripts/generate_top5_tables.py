"""The injected top 5: accuracy with it, and which of its skills agents open.

For each annotated task, the injected top 5 is the first five skills of its judgment pool
(annotation/top5/task_subset/, verified against the agent runs). Trials solved with the top 5 injected together
come from agent_results/ (full injection or progressive disclosure), trials solved with no skill from
labels/utility/.

    table_top5_opened.tex          tab:top5-opened          mean reward over all tasks: no skill, full injection,
                                                            progressive disclosure; then progressive
                                                            disclosure: the share of the top 5 opened and of opened
                                                            skills that are relevant (annotator grade >= 2 and grade
                                                            3, reference; the LLM judge is printed only); each trial
                                                            counted on its own, opened
                                                            meaning (SRA-Bench: the run's
                                                            skill_ids_used; Skill-Usage: a SKILL.md offered in the
                                                            prompt whose folder the agent's commands touch)
    python3 figures/scripts/generate_top5_tables.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "figures" / "output"  # everything the paper includes
OUT_DIR.mkdir(exist_ok=True)
POOL_DIR = REPO_ROOT / "annotation" / "top5" / "task_subset"
UTILITY_DIR = REPO_ROOT / "labels" / "utility"
JUDGE_NAME = "gpt-oss-120b"
JUDGE_DIR = REPO_ROOT / "labels" / "llm_judge" / JUDGE_NAME
SRA_TOP5 = REPO_ROOT / "agent_results" / "SRA-Bench" / "top5_subset50" / "topk5_comparison"  # <model>/<setting>/run<R>/
SU_TOP5 = REPO_ROOT / "agent_results" / "Skill-Usage" / "25_task_set" / "top5k_comparison"     # <model>-<setting>/<trial>/

sys.path.insert(0, str(REPO_ROOT / "labels" / "human"))
from human_labels import load_labels  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tex_table import TABLE_TAIL, table_head  # noqa: E402

BENCHMARKS = {"SRA-Bench": ("sra_bench", ["bigcodebench", "champ", "logicbench", "medcalcbench", "theoremqa"]),
              "Skill-Usage": ("skill_usage", ["skill_usage"])}
MODELS = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "GLM-4.5-Air-FP8": "GLM-4.5-Air",
          "Qwen3.6-27B-FP8": "Qwen3.6-27B"}  # weakest first
# setting -> (SRA-Bench run folder, Skill-Usage job suffix). Full injection injects whole skill records (name,
# description and content; the direct_full engine), as the single-skill candidate sweep does.
SETTINGS = {"Full injection": ("direct_injection_top5", "fullinject"),
            "Progressive disclosure": ("progressive_disclosure_top5", "progdisc")}


def parse(value):
    """Pool files store some lists as their Python repr."""
    return value if isinstance(value, list) else ast.literal_eval(value)


def baseline(folder: str, model: str) -> dict[str, float]:
    """task -> trials solved with no skill (summed reward over 3 trials)."""
    return {r["instance_id"]: sum(r["baseline"]["rewards"])
            for path in (UTILITY_DIR / folder / model).glob("*-utility.json")
            for r in json.loads(path.read_text())["results"]}


def top5_reward(bench: str, model: str, setting: str, tasks: set[str]) -> dict[str, float]:
    """task -> trials solved with the top 5 injected together (summed reward over 3 trials)."""
    sra_dir, su_suffix = SETTINGS[setting]
    reward, runs = defaultdict(float), defaultdict(int)
    if bench == "SRA-Bench":
        for path in SRA_TOP5.glob(f"{model}/{sra_dir}/run*/eval/*/{model}/result.json"):
            for det in json.loads(path.read_text())["details"]:
                if det["instance_id"] in tasks:
                    reward[det["instance_id"]] += float(det["correct"])
                    runs[det["instance_id"]] += 1
    else:
        for trial in (SU_TOP5 / f"{model}-{su_suffix}").iterdir():
            cfg, res = trial / "config.json", trial / "result.json"
            if not (cfg.is_file() and res.is_file()):
                continue
            task = Path(json.loads(cfg.read_text())["task"]["path"]).name
            if task in tasks:
                rewards = (json.loads(res.read_text()).get("verifier_result") or {}).get("rewards") or {}
                reward[task] += float(rewards.get("reward") or 0.0)  # a trial with no reward counts as 0
                runs[task] += 1
    incomplete = {t: runs[t] for t in tasks if runs[t] != 3}
    assert not incomplete, f"{bench} {model} {setting}: tasks without 3 trials: {incomplete}"
    return reward


def accuracies(bench: str, folder: str, model: str, tasks: set[str]) -> list[float]:
    """Mean reward (%) over the tasks and 3 trials: no skill, then the top 5 injected in each setting."""
    base = baseline(folder, model)
    rates = [100 * sum(base[t] for t in tasks) / (3 * len(tasks))]
    for setting in SETTINGS:
        reward = top5_reward(bench, model, setting, tasks)
        rates.append(100 * sum(reward[t] for t in tasks) / (3 * len(tasks)))
    return rates


def opened_skills(bench: str, model: str, tasks: set[str], top5: dict[str, list[str]]) -> dict[tuple, list[bool]]:
    """Progressive disclosure: (task, offered skill) -> whether it was opened, one entry per trial (3 per task)."""
    opened = defaultdict(list)
    if bench == "SRA-Bench":
        # the offered list is not logged; it is the task's top 5, as verified for full injection
        for path in SRA_TOP5.glob(f"{model}/progressive_disclosure_top5/run*/inference/*/*/result.jsonl"):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                task = r["instance_id"]
                if task in tasks:
                    used = set(r.get("skill_ids_used") or [])  # no field: the model opened no skill
                    for skill in top5[task]:
                        opened[(task, skill)].append(skill in used)
    else:
        for trial in (SU_TOP5 / f"{model}-progdisc").iterdir():
            traj, cfg = trial / "agent" / "trajectory.json", trial / "config.json"
            if not (traj.is_file() and cfg.is_file()):
                continue
            task = Path(json.loads(cfg.read_text())["task"]["path"]).name
            if task not in tasks:
                continue
            steps = json.loads(traj.read_text())["steps"]
            offered = dict.fromkeys(re.findall(r"/skills/([^/<]+)/SKILL\.md", json.dumps(steps[0])))
            commands = json.dumps([st.get("tool_calls") for st in steps[1:] if st.get("source") == "agent"])
            for skill in offered:
                opened[(task, skill)].append(f"skills/{skill}" in commands)
    incomplete = {p: len(v) for p, v in opened.items() if len(v) != 3}
    assert not incomplete, f"{bench} {model}: offered skills without 3 trials: {incomplete}"
    return opened


def write_opened_table(grades: dict) -> None:
    """Accuracy with the top 5, then which top-5 skills a model opens under progressive
    disclosure: the mean number of top-5 skills opened per trial, and the share of opened skills that are relevant (annotator grade >= 2,
    annotator grade 3, or benchmark reference skill). The LLM judge's shares, and the share of relevant skills that
    were opened, are printed only. Every trial counts on its own, as in the accuracy columns: a skill offered in 3
    trials is 3 offers, each opened or not. Grade 3 shows that agents favour the most relevant skills."""
    judge = {}
    for path in JUDGE_DIR.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                if j.get("rating") is not None:
                    judge[(j["instance_id"], j["skill_id"])] = j["rating"]

    shown = ["annotator >= 2", "annotator 3", "reference"]  # sources with a column in the table
    n_cols = 5 + len(shown)  # model, accuracy (no skill, full injection, progressive disclosure), opened, shown
    lines = []
    for b_i, (bench, (folder, datasets)) in enumerate(BENCHMARKS.items()):
        top5, refs = top5_and_refs(datasets)
        top5_grades = [grades[(t, s)] for t, skills in top5.items() for s in skills]
        print(f"{bench:12} top 5 annotator grades 0/1/2/3 (% of {len(top5_grades)} skills) "
              + "/".join(f"{100 * top5_grades.count(g) / len(top5_grades):.0f}" for g in range(4)))
        relevant = {"annotator >= 2": lambda p: grades[p] >= 2, "annotator 3": lambda p: grades[p] == 3,
                    "judge >= 2": lambda p: judge.get(p, 0) >= 2, "judge 3": lambda p: judge.get(p, 0) == 3,
                    "reference": lambda p: p[1] in refs[p[0]]}
        if b_i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{bench}}}}} \\")
        for model, short in MODELS.items():
            if not (UTILITY_DIR / folder / model).is_dir():
                continue
            opened = opened_skills(bench, model, set(top5), top5)
            # every trial on its own: a skill offered in 3 trials counts 3 times, once per trial it was opened
            hits, offers = sum(sum(v) for v in opened.values()), sum(len(v) for v in opened.values())
            per_trial = hits / (3 * len({task for task, _ in opened}))  # mean skills opened in one trial
            of_opened, of_relevant = {}, {}
            for source, is_rel in relevant.items():
                rel = [v for p, v in opened.items() if is_rel(p)]
                of_opened[source] = 100 * sum(sum(v) for v in rel) / hits
                of_relevant[source] = 100 * sum(sum(v) for v in rel) / sum(len(v) for v in rel)
            print(f"{bench:12} {short:13} opened {hits:3} of {offers} ({per_trial:.2f} per trial) | relevant of opened "
                  "ann>=2/ann3/judge>=2/judge3/ref " + "/".join(f"{x:.0f}" for x in of_opened.values())
                  + " | opened of relevant " + "/".join(f"{x:.0f}" for x in of_relevant.values()))
            accuracy = " & ".join(f"{x:.0f}" for x in accuracies(bench, folder, model, set(top5)))
            shares = " & ".join(f"{of_opened[s]:.0f}" for s in shown)
            lines.append(rf"\quad {short} & {accuracy} & {per_trial:.1f} & {shares} \\")

    (OUT_DIR / "table_top5_opened.tex").write_text("\n".join([
        "% Generated by figures/scripts/generate_top5_tables.py -- needs \\usepackage{booktabs,makecell}",
        *table_head("tab:top5-opened"),
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        # a wider gap before the progressive-disclosure columns sets them apart from accuracy
        rf"\begin{{tabular}}{{@{{}}l rrr@{{\hspace{{14pt}}}}r {'r' * len(shown)}@{{}}}}", r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{Accuracy (\%)}} & \multicolumn{{{1 + len(shown)}}}{{c}}{{Progressive disclosure}} \\",
        rf"\cmidrule(r){{2-4}} \cmidrule(l){{5-{n_cols}}}",
        r"Model & Baseline & \makecell[r]{Full\\injection} & \makecell[r]{Progressive\\disclosure}"
        r" & \makecell[r]{Opened\\per trial} & \makecell[r]{\% graded\\$\geq$2} & \makecell[r]{\% graded\\3}"
        r" & \makecell[r]{\% ref.\\skill} \\",
        r"\midrule", *lines, r"\bottomrule", r"\end{tabular}", TABLE_TAIL,
    ]) + "\n")
    print(f"Saved: {OUT_DIR / 'table_top5_opened'}.tex")


def top5_and_refs(datasets: list[str]) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """task -> its top 5 (first five of its judgment pool), and task -> its reference skills."""
    top5, refs = {}, {}
    for ds in datasets:
        for r in json.loads((POOL_DIR / f"{ds}-annotation.json").read_text())["results"]:
            top5[r["instance_id"]] = [c["skill_id"] for c in parse(r["retrieved"])][:5]
            refs[r["instance_id"]] = set(parse(r["gold_skill_ids"]))
    return top5, refs


def main() -> None:
    grades = {(r["instance_id"], r["skill_id"]): r["final"] for r in load_labels()}
    write_opened_table(grades)


if __name__ == "__main__":
    main()
