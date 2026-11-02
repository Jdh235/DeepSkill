"""`direct_full` injects name + description + content; `direct` injects content only; otherwise identical."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(not (ROOT.parents[1] / "resource/data/sra_bench/corpus/corpus.json").exists(), reason="needs resource/data/sra_bench")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _corpus():
    from sragents.corpus import load_corpus_dict
    return load_corpus_dict()


def _instance(ds):
    return json.loads((ROOT.parents[1] / "resource/data/sra_bench/instances" / f"{ds}.json").read_text())[0]


class _StubClient:
    """Records every request; replies from a script (the last reply repeats)."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _first_user(client):
    return next(m["content"] for m in client.requests[0]["messages"] if m["role"] == "user")


def _run(engine_name, skill, ds, replies=("ok",)):
    from sragents.infer import get_engine
    client = _StubClient(replies)
    result = get_engine(engine_name).run(_instance(ds), [skill], client, MODEL)
    return result, client


def test_full_engine_injects_name_description_and_content_and_direct_does_not():
    skill = _corpus()["logicbench_018"]
    _, full = _run("direct_full", skill, "logicbench")
    _, plain = _run("direct", skill, "logicbench")
    template = (ROOT / "src/sragents/infer/engines/direct_full_template.txt").read_text().rstrip("\n")
    injected = (template.replace("{name}", skill["name"]).replace("{description}", skill["description"])
                .replace("{content}", skill["content"]))
    assert _first_user(full).startswith("Relevant Skill:\n" + injected + "\n\n")      # same boundary as `direct`
    assert skill["name"] in _first_user(full) and skill["description"] in _first_user(full)
    assert skill["description"] not in _first_user(plain)                      # the difference between the engines
    assert _first_user(full).split(injected)[1] == _first_user(plain).split(skill["content"])[1]   # same question after
    assert {k: v for k, v in full.requests[0].items() if k != "messages"} == \
           {k: v for k, v in plain.requests[0].items() if k != "messages"}     # same sampling settings, etc.


def test_full_engine_keeps_skill_ids_and_the_tool_loop():
    from sragents.infer.engines.direct_full import format_skill
    skill = next(s for s in _corpus().values() if s.get("tools"))
    tool = skill["tools"][0]
    call = f"TOOL_CALL: {tool['name']}({', '.join(f'{k}=1' for k in tool['parameters'])})"
    result, client = _run("direct_full", skill, "medcalcbench", replies=(f"Compute.\n{call}", "ANSWER: 5"))
    assert result.skill_ids_used == [skill["skill_id"]]
    assert len(client.requests) == 2 and client.requests[1]["messages"][-1]["content"].startswith("TOOL_RESULT:")
    assert format_skill(skill) in _first_user(client)


def test_every_injected_skill_gets_its_own_heading():
    from sragents.infer import get_engine
    from sragents.infer.engines.direct_full import format_skill
    corpus = _corpus()
    skills = [corpus["logicbench_018"], {"skill_id": "empty", "content": ""}, corpus["theoremqa_000"]]
    client = _StubClient(("ok",))
    get_engine("direct_full").run(_instance("logicbench"), skills, client, MODEL)
    first, third = format_skill(skills[0]), format_skill(skills[2])
    # the content-less skill is skipped (as in `direct`), so the second heading goes on the third skill
    assert _first_user(client).startswith(f"Relevant Skill:\n{first}\n---\nRelevant Skill:\n{third}\n\n")
    assert _first_user(client).count("Relevant Skill:") == 2

    single = _StubClient(("ok",))
    get_engine("direct_full").run(_instance("logicbench"), skills[:1], single, MODEL)
    assert _first_user(single).count("Relevant Skill:") == 1          # one skill (the sweep): unchanged


def test_template_is_filled_in_one_pass_and_left_verbatim():
    from sragents.infer.engines.direct_full import format_skill
    skill = {"name": "N {content}", "description": "D {braces} ok", "content": "C {name}"}
    assert format_skill(skill, "[{name}|{description}|{content}]") == "[N {content}|D {braces} ok|C {name}]"


def test_new_methods_are_in_the_catalog_and_use_the_new_engine():
    from sragents.experiments.definitions import EXPERIMENTS
    from sragents.infer import list_engines
    spec = EXPERIMENTS["full_skill_injection"]
    assert {m.label for m in spec.methods} == {"oracle_skill_full", "bm25_top1_full"}
    assert all(m.engine == "direct_full" and m.engine in list_engines() for m in spec.methods)
    assert "full_skill_injection" not in {m.label for m in EXPERIMENTS["main"].methods}   # `main` is unchanged
