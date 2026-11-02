"""Minimal OpenAI-compatible LLM wrapper.

Works with vLLM (OpenAI-compatible API), OpenAI, or any compatible endpoint.
Config via CLI args (--model, --api-base) and env vars (OPENAI_API_BASE, OPENAI_API_KEY).
"""

import os
import re
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

# Errors worth retrying: the server was unreachable or broke transiently
# (e.g. a watchdog is restarting it after a hang). Deliberately excludes
# 4xx errors like BadRequestError (context-length-exceeded, malformed
# request, ...) — those fail identically on every attempt, so retrying
# them would just waste the retry budget instead of failing fast.
_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)

# Long enough to outlast a vLLM restart cycle (detect a hang, kill -9,
# reload the model, wait for readiness) rather than the OpenAI SDK's own
# default of 2 quick retries, which is sized for a momentary blip, not a
# multi-minute service restart.
_DEFAULT_RETRY_TIMEOUT = 300.0
_DEFAULT_RETRY_INTERVAL = 10.0


def _create_chat_completion(client: OpenAI, **kwargs):
    """``client.chat.completions.create`` with a long retry budget for
    transient connection failures (server down/restarting), and no retry
    at all for request-validation errors (those never succeed)."""
    deadline = time.monotonic() + _DEFAULT_RETRY_TIMEOUT
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except _RETRYABLE_ERRORS:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_DEFAULT_RETRY_INTERVAL)


def create_llm_client(
    api_base: str | None = None, api_key: str | None = None
) -> OpenAI:
    """Create OpenAI-compatible client (works with vLLM, OpenAI, etc.)."""
    base_url = api_base or os.environ.get("OPENAI_API_BASE")
    key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=key)


_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from an LLM response.

    Handles both well-formed `<think>...</think>answer` and truncation
    cases where generation is cut mid-thinking (unclosed tag).
    """
    if "<think>" not in text:
        return text
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()


def get_extra_body(model: str, thinking: bool = False) -> dict | None:
    """Return per-model extra_body for thinking/reasoning control.

    By default (thinking=False) suppresses thinking on hybrid-thinking
    models so they're comparable to non-reasoning baselines. GPT-5 is a
    pure reasoning model; we always run it at minimal effort.

      - Qwen3 / GLM-4.5 / GLM-5 / Kimi: chat_template_kwargs.enable_thinking
      - GPT-5: reasoning_effort="minimal" (always)
      - Others (Llama, Mistral, MiniMax, ...): no flag

    GLM used to send a bare top-level {"enable_thinking": thinking} --
    vLLM's OpenAI-compatible endpoint only threads chat_template_kwargs
    through to the Jinja2 chat template render, so a top-level field like
    that is silently ignored (not an error, just a no-op). Confirmed
    directly on the GLM-4.5-Air-FP8 baseline: 176/223 (79%) of champ
    responses never closed their <think> block, burning the entire
    4096-token budget on unstoppable thinking with zero extractable
    answer -- explaining GLM's baseline scores collapsing far below every
    other model (e.g. champ: 0.16 vs Qwen3.6-27B's 0.75) despite
    `--thinking` never being passed (defaults to False already).
    """
    basename = model.lower().rsplit("/", 1)[-1]

    if "qwen3" in basename or "glm-4.5" in basename or "glm-5" in basename or "kimi" in basename:
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    if "gpt-5" in basename:
        return {"reasoning_effort": "minimal"}
    return None


def chat(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion request, return content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = _create_chat_completion(client, **kwargs)
    return response.choices[0].message.content


def chat_messages(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stop: list[str] | None = None,
    extra_body: dict | None = None,
) -> str:
    """Send chat completion with an explicit messages list (for multi-turn)."""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if stop:
        kwargs["stop"] = stop
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = _create_chat_completion(client, **kwargs)
    return response.choices[0].message.content
