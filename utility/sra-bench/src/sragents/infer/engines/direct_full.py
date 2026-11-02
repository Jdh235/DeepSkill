"""Direct engine variant that injects the WHOLE skill -- name, description and content.

`direct` prepends only the skill's `content` (the body); `name` and `description` are never shown to the
model. This engine is identical except that each skill is first rendered through
``direct_full_template.txt`` (placeholders ``{name}``, ``{description}``, ``{content}``), so the model reads
the skill as a complete record. Everything else -- one chat call, the tool loop for skills that expose
``tools``, ``skill_ids_used`` -- is inherited unchanged from :class:`DirectEngine`.

Edit the template file to change the injected format (trailing newlines in the file are ignored; nothing
else in it is interpreted). It is meant to match the format used in the skill-usage setting, so the two
settings inject the same thing.

When several skills are injected, each one gets its own ``Relevant Skill:`` heading. The prompt builder
writes that heading once, above the first skill, and joins the skills with ``\\n---\\n``; this engine adds
the heading to every later skill, so the block reads::

    Relevant Skill:
    <skill 1>
    ---
    Relevant Skill:
    <skill 2>

    <question>

A single skill (the candidate sweep, k=1) is unaffected. The boundary before the question is left as the
prompt builder writes it (a blank line, no separator), the same as in `direct`, so the two engines differ
only in the name/description block.
"""

import re
from pathlib import Path

from sragents.infer.base import InferenceResult, register_engine
from sragents.infer.engines.direct import DirectEngine

TEMPLATE_PATH = Path(__file__).resolve().parent / "direct_full_template.txt"
_PLACEHOLDER = re.compile(r"\{(name|description|content)\}")
# Same heading build_prompt (sragents.prompts) puts above the first skill.
SKILL_HEADING = "Relevant Skill:\n"


def load_template() -> str:
    return TEMPLATE_PATH.read_text().rstrip("\n")


def format_skill(skill: dict, template: str | None = None) -> str:
    """The skill as the model sees it. Single pass, so braces or placeholder-like text inside a skill's own
    fields are left exactly as written."""
    template = load_template() if template is None else template
    return _PLACEHOLDER.sub(lambda m: skill.get(m.group(1), ""), template)


@register_engine("direct_full")
class DirectFullEngine(DirectEngine):
    """DirectEngine that injects name + description + content instead of content alone."""

    def run(self, instance: dict, skills: list[dict], client, model: str, **kwargs) -> InferenceResult:
        template = load_template()
        full, n_injected = [], 0
        for s in skills:
            if not s.get("content"):          # DirectEngine skips these, so they don't count as injected
                full.append(s)
                continue
            text = format_skill(s, template)
            if n_injected:                    # build_prompt already heads the first one
                text = SKILL_HEADING + text
            full.append({**s, "content": text})
            n_injected += 1
        return super().run(instance, full, client, model, **kwargs)
