from typing import Callable

_BUILDERS: dict[str, Callable[[dict], tuple[str, str]]] = {}


def register_prompt_builder(dataset: str):
    """Decorator: register a prompt builder for ``dataset``."""
    def wrap(fn):
        _BUILDERS[dataset] = fn
        return fn
    return wrap


def get_builder(dataset: str) -> Callable[[dict], tuple[str, str]]:
    if dataset not in _BUILDERS:
        raise ValueError(
            f"No prompt builder registered for dataset {dataset!r}. "
            f"Registered: {list_datasets()}. "
            "Define one with @register_prompt_builder('...')."
        )
    return _BUILDERS[dataset]


def list_datasets() -> list[str]:
    return sorted(_BUILDERS)


def build_prompt(instance: dict) -> tuple[str, str]:
    """Build ``(system_prompt, user_prompt)`` for the given instance."""
    return get_builder(instance["dataset"])(instance)


# ---------------------------------------------------------------------------
# Built-in dataset builders
# ---------------------------------------------------------------------------

@register_prompt_builder("theoremqa")
def _build_theoremqa(instance: dict) -> tuple[str, str]:
    system = (
        "You are a science teacher, you are supposed to provide a solution to a "
        "given problem. You need to output the answer in your final sentence like "
        '"Therefore, the answer is ...". The answer can only be one of the following '
        "forms:\n"
        "1. a numerical value like 0.1, no symbol at all.\n"
        "2. a list of number like [2, 3, 4].\n"
        "3. True/False.\n"
        "4. an option like (a), (b), (c), (d)"
    )
    user = f"Problem:{instance['question']}\nSolution:"
    return system, user


@register_prompt_builder("logicbench")
def _build_logicbench(instance: dict) -> tuple[str, str]:
    return "", instance["question"]


@register_prompt_builder("champ")
def _build_champ(instance: dict) -> tuple[str, str]:
    system = "You are an expert on mathematics."
    user = (
        "Solve the following problem. Make sure to show your work before giving "
        "the final answer.\n\n"
        f"{instance['question']}\n\n"
        "After your solution, write your final answer on its own line in "
        "exactly this format:\n"
        "ANSWER: <your answer>\n\n"
        "The answer should be concise: a number, mathematical expression, "
        "Yes/No, or a brief phrase. Do not include explanations in the "
        "ANSWER line. Use plain text only — do not use LaTeX, dollar signs, "
        "or any other formatting (e.g., write n! not \\(n!\\) or $n!$)."
    )
    return system, user


@register_prompt_builder("medcalcbench")
def _build_medcalcbench(instance: dict) -> tuple[str, str]:
    system = (
        "You are a helpful assistant for calculating a score for a given "
        "patient note. Please think step-by-step to solve the question and "
        "then generate the required score."
    )
    user = (
        f"{instance['question']}\n\n"
        "Show your step-by-step calculation, then write your final answer on "
        "its own line in exactly this format:\n"
        "ANSWER: <your answer>\n\n"
        "For numeric answers, give the number only (e.g., 25.24). "
        "For date answers, use MM/DD/YYYY format. "
        "For scores, give the integer value. "
        "Do not include units or explanations in the ANSWER line."
    )
    return system, user


@register_prompt_builder("bigcodebench")
def _build_bigcodebench(instance: dict) -> tuple[str, str]:
    return "", instance["question"]
