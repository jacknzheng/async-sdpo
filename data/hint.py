from __future__ import annotations

import logging
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from train.config import Config
from data.dataset import Task
from reward.judge import chat_completion

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

ERROR_HINT_PROMPTS: dict[str, str] = {
    name: (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
    for name in ("answer_free", "answer_bearing", "step_hint")
}


@dataclass(frozen=True)
class Hints:
    """One rollout's teacher contexts. Mixture fills both; a single arm fills one."""

    free: str = ""
    bearing: str = ""

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


async def build_error_hint(
    query: str = "",
    sections: list[dict] | None = None,
    response_text: str = "",
    prompt_variant: str = "answer_free",
    model: str = "stealth/ox-alpha",
    timeout: float = 60.0,
    max_retries: int = 2,
    user_prompt: str | None = None,
) -> str:


    if prompt_variant not in ERROR_HINT_PROMPTS:
        raise ValueError(
            f"unknown hint prompt {prompt_variant!r}; choose from {sorted(ERROR_HINT_PROMPTS)}"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ""

    if user_prompt is None:
        user_prompt = (
            f"<question>\n{query}\n</question>\n\n"
            f"<rubric>\n{_format_rubric(sections or [])}\n</rubric>\n\n"
            f"<draft_answer>\n{response_text}\n</draft_answer>"
        )

    try:
        generated = await chat_completion(
            ERROR_HINT_PROMPTS[prompt_variant],
            user_prompt,
            model=model,
            api_key=api_key,
            max_tokens=1024,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception:  # noqa: BLE001 -- a hint failure must never break a rollout
        logger.debug("error-hint generation failed", exc_info=True)
        return ""

    generated = generated.strip()
    if not generated:
        return ""
    if prompt_variant == "step_hint":
        return generated
    return f"Guidance for answering this question:\n{generated}\n"

# initialize hint rate limiter to None
_HINT_SEM: asyncio.Semaphore | None = None

def _hint_sem(concurrency: int) -> asyncio.Semaphore:
    global _HINT_SEM
    if _HINT_SEM is None:
        _HINT_SEM = asyncio.Semaphore(concurrency)
    return _HINT_SEM


async def _one_hint(config: Config, task: Task, response_text: str, variant: str) -> str:
    async with _hint_sem(config.generator.hint.concurrency):
        return await build_error_hint(
            query=task.query,
            sections=task.sections,
            response_text=response_text,
            prompt_variant=variant,
            model=config.generator.hint.model,
            timeout=config.generator.hint.timeout,
        )


async def generate_hint(config: Config, task: Task, response_text: str) -> Hints:
    mode = config.generator.hint.prompt
    try:
        if mode == "gold":
            from data.tau_harness import gold_suffix

            return Hints(free=gold_suffix(task))
        if mode == "step_hint":
            from data.tau_harness import gold_material

            async with _hint_sem(config.generator.hint.concurrency):
                text = await build_error_hint(
                    prompt_variant="step_hint",
                    user_prompt=(
                        f"<gold>\n{gold_material(task)}\n</gold>\n\n"
                        f"<transcript>\n{response_text}\n</transcript>"
                    ),
                    model=config.generator.hint.model,
                    timeout=config.generator.hint.timeout,
                )
            return Hints(free=text)
        if mode == "mixture":
            free, bearing = await asyncio.gather(
                _one_hint(config, task, response_text, "answer_free"),
                _one_hint(config, task, response_text, "answer_bearing"),
            )
            return Hints(free=free, bearing=bearing)
        if mode == "answer_bearing":
            return Hints(bearing=await _one_hint(config, task, response_text, "answer_bearing"))
        return Hints(free=await _one_hint(config, task, response_text, "answer_free"))
    except Exception:
        logger.exception("error-hint failed for task %s", task.task_id)
        return Hints()
