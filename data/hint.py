from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HINT_PROMPT_ANSWER_FREE = """\
You are coaching an analyst who has just written a draft answer to a financial diligence \
question. You can see the grading rubric; they cannot, and they never will.

Write a short hint that redirects their REASONING toward what they neglected.

Absolute constraints:
- NEVER state a figure, number, date, percentage, or currency amount.
- NEVER state a conclusion or finding. Say what to examine, not what is true.
- Only raise points the draft actually missed or handled shallowly.
- If the draft already covers the rubric well, say what would deepen the weakest part.

Write 2-4 bullet points, each naming one relationship or risk worth examining. No preamble.

Good:  "- how interest-bearing deposit cost trended against balance growth"
Bad:   "- deposit cost fell to 0.39%"        (states a figure)
Bad:   "- deposit beta is clearly favorable" (states a conclusion)"""

HINT_PROMPT_ANSWER_BEARING = """\
You are coaching an analyst who has just written a draft answer to a financial diligence \
question. You can see the grading rubric; they cannot.

Write a short hint telling them exactly what their draft got wrong or left out.

Instructions:
- Cite the specific facts, figures, and conclusions from the rubric that the draft missed.
- Be concrete. Include the actual numbers and dates.
- Only raise points the draft actually missed or stated incorrectly.
- If the draft already covers the rubric well, name the most important remaining gap.

Write 2-4 bullet points. No preamble."""

ERROR_HINT_PROMPTS: dict[str, str] = {
    "answer_free": HINT_PROMPT_ANSWER_FREE,
    "answer_bearing": HINT_PROMPT_ANSWER_BEARING,
}


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
    query: str,
    sections: list[dict],
    response_text: str,
    prompt_variant: str = "answer_free",
    model: str = "deepseek/deepseek-v4-flash",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> str:
    """Generate a hint conditioned on where THIS rollout went wrong.

    Returns "" on any failure -- a missing key, an API error, or an empty generation. The
    caller drops the trajectory rather than training it with an empty hint, which would
    make the teacher identical to the student and the gradient ~zero.
    """
    # Imported here rather than at module scope: data/ is otherwise free of network code,
    # and a top-level import would drag the judge's dependencies into dataset loading.
    from reward.judge import chat_completion

    if prompt_variant not in ERROR_HINT_PROMPTS:
        raise ValueError(
            f"unknown hint prompt {prompt_variant!r}; choose from {sorted(ERROR_HINT_PROMPTS)}"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return ""

    user_prompt = (
        f"<question>\n{query}\n</question>\n\n"
        f"<rubric>\n{_format_rubric(sections)}\n</rubric>\n\n"
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
    return f"Guidance for answering this question:\n{generated}\n"
