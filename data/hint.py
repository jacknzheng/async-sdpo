from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from train.config import Config
from data.dataset import Task
from data.diagnostics import artifact_event
from reward.judge import chat_completion

logger = logging.getLogger(__name__)


class HintCompleter(Protocol):
    """Duck-typed local hint backend. Tests stub this; production uses HintEngine."""

    async def complete(self, system: str, user: str) -> str: ...


_HINT_ENGINE: HintCompleter | None = None


def set_hint_engine(engine: HintCompleter | None) -> None:
    """Rank-0 installs the live HintEngine here so generate_hint never imports vLLM."""
    global _HINT_ENGINE
    _HINT_ENGINE = engine


def _resolve_engine(engine: HintCompleter | None) -> HintCompleter | None:
    return engine if engine is not None else _HINT_ENGINE

@dataclass(frozen=True)
class HintFailure:
    cause: str
    detail: str


_hint_failure: contextvars.ContextVar[HintFailure | None] = contextvars.ContextVar(
    "hint_failure", default=None
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

ERROR_HINT_PROMPTS: dict[str, str] = {
    name: (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
    for name in ("answer_free", "answer_bearing", "step_hint")
}


@dataclass(frozen=True)
class Hints:
    """One rollout's teacher contexts and any generation failure cause."""

    free: str = ""
    bearing: str = ""
    cause: str = ""
    detail: str = ""

    def ok(self, mode: str) -> bool:
        if mode == "mixture":
            return bool(self.free) and bool(self.bearing)
        if mode == "answer_bearing":
            return bool(self.bearing)
        return bool(self.free)


def _format_rubric(sections: list[dict]) -> str:
    """The rubric as the hint model sees it.

    Full text in BOTH arms -- the model needs the answer key to know what the draft missed.
    Only the answer_free arm's OUTPUT is constrained, never its input.
    """
    lines: list[str] = []
    for section in sections:
        lines.append(f"[{section.get('id', 'section')}]")
        for criterion in section.get("criteria", []):
            requirement = criterion.get("requirement", "").strip()
            if requirement:
                lines.append(f"- ({criterion.get('weight', 0)}) {requirement}")
    return "\n".join(lines)


def _failure_from_exception(
    exc: BaseException, *, backend: str = "openrouter"
) -> HintFailure:
    """Classify a hint-backend failure and retain a bounded, one-line diagnostic."""
    detail = " ".join(f"{type(exc).__name__}: {exc}".split())[:800]
    lower = detail.lower()
    if (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in lower
        or "timed out" in lower
    ):
        cause = "timeout"
    elif backend == "vllm":
        cause = "vllm_error"
    elif (
        "insufficient credit" in lower
        or "insufficient credits" in lower
        or "(402)" in lower
    ):
        cause = "openrouter_credit"
    elif "auth failed (401)" in lower or "auth failed (403)" in lower:
        cause = "openrouter_auth"
    elif "(429)" in lower or "429 client error" in lower or "rate limit" in lower:
        cause = "openrouter_rate_limit"
    elif "finish_reason=length" in lower:
        cause = "openrouter_length"
    else:
        cause = "openrouter_error"
    return HintFailure(cause=cause, detail=detail)


async def build_error_hint(
    query: str = "",
    sections: list[dict] | None = None,
    response_text: str = "",
    prompt_variant: str = "answer_free",
    model: str = "deepseek/deepseek-v4-flash-latest",
    timeout: float = 90.0,
    max_retries: int = 5,
    max_tokens: int = 2048,
    reasoning_enabled: bool = False,
    user_prompt: str | None = None,
    backend: str = "openrouter",
    engine: HintCompleter | None = None,
) -> str:
    _hint_failure.set(None)
    if prompt_variant not in ERROR_HINT_PROMPTS:
        raise ValueError(
            f"unknown hint prompt {prompt_variant!r}; choose from {sorted(ERROR_HINT_PROMPTS)}"
        )

    if user_prompt is None:
        user_prompt = (
            f"<question>\n{query}\n</question>\n\n"
            f"<rubric>\n{_format_rubric(sections or [])}\n</rubric>\n\n"
            f"<draft_answer>\n{response_text}\n</draft_answer>"
        )

    if backend == "vllm":
        generated = await _complete_local_hint(
            ERROR_HINT_PROMPTS[prompt_variant],
            user_prompt,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            prompt_variant=prompt_variant,
            query=query,
            response_text=response_text,
            engine=engine,
        )
    else:
        generated = await _complete_openrouter_hint(
            ERROR_HINT_PROMPTS[prompt_variant],
            user_prompt,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            prompt_variant=prompt_variant,
            query=query,
            response_text=response_text,
        )

    if not generated:
        return ""
    if prompt_variant == "step_hint":
        return generated
    return f"Guidance for answering this question:\n{generated}\n"


async def _complete_local_hint(
    system: str,
    user_prompt: str,
    *,
    model: str,
    timeout: float,
    max_retries: int,
    prompt_variant: str,
    query: str,
    response_text: str,
    engine: HintCompleter | None,
) -> str:
    completer = _resolve_engine(engine)
    if completer is None:
        failure = HintFailure(
            cause="vllm_error",
            detail="hint engine is not started",
        )
        _hint_failure.set(failure)
        logger.warning(
            "error-hint generation failed (variant=%s, cause=%s, detail=%s)",
            prompt_variant,
            failure.cause,
            failure.detail,
        )
        artifact_event(
            "api_failures",
            "hint_generation_failed",
            provider="vllm",
            operation="hint_generation",
            model=model,
            prompt_variant=prompt_variant,
            cause=failure.cause,
            error=failure.detail,
            query=query,
        )
        return ""

    last_failure: HintFailure | None = None
    for attempt in range(max_retries):
        try:
            generated = await asyncio.wait_for(
                completer.complete(system, user_prompt),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 -- a hint failure must never break a rollout
            last_failure = _failure_from_exception(exc, backend="vllm")
            logger.warning(
                "error-hint generation failed (variant=%s, attempt=%d, cause=%s, detail=%s)",
                prompt_variant,
                attempt + 1,
                last_failure.cause,
                last_failure.detail,
            )
            continue
        generated = generated.strip()
        if generated:
            return generated
        last_failure = HintFailure(
            cause="empty",
            detail="local hint engine returned empty hint content",
        )
        break

    failure = last_failure or HintFailure(
        cause="vllm_error",
        detail="local hint engine returned no content",
    )
    _hint_failure.set(failure)
    artifact_event(
        "api_failures",
        "hint_generation_failed",
        provider="vllm",
        operation="hint_generation",
        model=model,
        prompt_variant=prompt_variant,
        cause=failure.cause,
        error=failure.detail,
        query=query,
        response_text=response_text,
    )
    return ""


async def _complete_openrouter_hint(
    system: str,
    user_prompt: str,
    *,
    model: str,
    timeout: float,
    max_retries: int,
    max_tokens: int,
    reasoning_enabled: bool,
    prompt_variant: str,
    query: str,
    response_text: str,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        _hint_failure.set(
            HintFailure(
                cause="openrouter_auth",
                detail="OPENROUTER_API_KEY is unset",
            )
        )
        logger.warning("OPENROUTER_API_KEY is unset; cannot generate %s hint", prompt_variant)
        artifact_event(
            "api_failures",
            "hint_generation_failed",
            provider="openrouter",
            operation="hint_generation",
            model=model,
            prompt_variant=prompt_variant,
            cause="openrouter_auth",
            error="OPENROUTER_API_KEY is unset",
            query=query,
        )
        return ""

    try:
        generated = await chat_completion(
            system,
            user_prompt,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            reasoning_enabled=reasoning_enabled,
        )
    except Exception as exc:  # noqa: BLE001 -- a hint failure must never break a rollout
        failure = _failure_from_exception(exc, backend="openrouter")
        _hint_failure.set(failure)
        logger.warning(
            "error-hint generation failed (variant=%s, cause=%s, detail=%s)",
            prompt_variant,
            failure.cause,
            failure.detail,
        )
        artifact_event(
            "api_failures",
            "hint_generation_failed",
            provider="openrouter",
            operation="hint_generation",
            model=model,
            prompt_variant=prompt_variant,
            cause=failure.cause,
            error=failure.detail,
            query=query,
            response_text=response_text,
        )
        return ""

    generated = generated.strip()
    if not generated:
        _hint_failure.set(
            HintFailure(
                cause="empty",
                detail="OpenRouter returned empty hint content",
            )
        )
        artifact_event(
            "api_failures",
            "hint_generation_failed",
            provider="openrouter",
            operation="hint_generation",
            model=model,
            prompt_variant=prompt_variant,
            cause="empty",
            error="OpenRouter returned empty hint content",
            query=query,
            response_text=response_text,
        )
        return ""
    return generated

# initialize hint rate limiter to None
_HINT_SEM: asyncio.Semaphore | None = None


def _hint_sem(concurrency: int) -> asyncio.Semaphore:
    global _HINT_SEM
    if _HINT_SEM is None:
        _HINT_SEM = asyncio.Semaphore(concurrency)
    return _HINT_SEM


def _failure_for(text: str) -> HintFailure | None:
    if text:
        return None
    return _hint_failure.get() or HintFailure(
        cause="empty",
        detail="hint generator returned no content",
    )


def _hint_kwargs(config: Config, engine: HintCompleter | None = None) -> dict:
    return {
        "model": config.generator.hint.model,
        "timeout": config.generator.hint.timeout,
        "max_retries": config.generator.hint.max_retries,
        "max_tokens": config.generator.hint.max_tokens,
        "reasoning_enabled": config.generator.hint.reasoning_enabled,
        "backend": config.generator.hint.backend,
        "engine": engine,
    }


async def _one_hint(
    config: Config,
    task: Task,
    response_text: str,
    variant: str,
    engine: HintCompleter | None = None,
) -> tuple[str, HintFailure | None]:
    async with _hint_sem(config.generator.hint.concurrency):
        text = await build_error_hint(
            query=task.query,
            sections=task.sections,
            response_text=response_text,
            prompt_variant=variant,
            **_hint_kwargs(config, engine),
        )
    return text, _failure_for(text)


async def generate_hint(
    config: Config,
    task: Task,
    response_text: str,
    engine: HintCompleter | None = None,
) -> Hints:
    mode = config.generator.hint.prompt
    try:
        if mode == "gold":
            from data.tau_harness import gold_suffix

            text = gold_suffix(task)
            return Hints(free=text, cause="" if text else "empty")
        if mode == "step_hint":
            from data.tau_harness import gold_material

            async with _hint_sem(config.generator.hint.concurrency):
                text = await build_error_hint(
                    prompt_variant="step_hint",
                    user_prompt=(
                        f"<gold>\n{gold_material(task)}\n</gold>\n\n"
                        f"<transcript>\n{response_text}\n</transcript>"
                    ),
                    **_hint_kwargs(config, engine),
                )
            failure = _failure_for(text)
            return Hints(
                free=text,
                cause=failure.cause if failure else "",
                detail=failure.detail if failure else "",
            )
        if mode == "mixture":
            (free, free_failure), (bearing, bearing_failure) = await asyncio.gather(
                _one_hint(config, task, response_text, "answer_free", engine),
                _one_hint(config, task, response_text, "answer_bearing", engine),
            )
            failure = free_failure or bearing_failure
            return Hints(
                free=free,
                bearing=bearing,
                cause=failure.cause if failure else "",
                detail=failure.detail if failure else "",
            )
        if mode == "answer_bearing":
            text, failure = await _one_hint(
                config, task, response_text, "answer_bearing", engine
            )
            return Hints(
                bearing=text,
                cause=failure.cause if failure else "",
                detail=failure.detail if failure else "",
            )
        text, failure = await _one_hint(config, task, response_text, "answer_free", engine)
        return Hints(
            free=text,
            cause=failure.cause if failure else "",
            detail=failure.detail if failure else "",
        )
    except Exception as exc:
        logger.exception("error-hint failed for task %s", task.task_id)
        failure = _failure_from_exception(exc, backend=config.generator.hint.backend)
        return Hints(cause="other", detail=failure.detail)
