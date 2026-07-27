"""Error-conditioned hint generation for the SDPO teacher.

The hint `c` is prepended ONLY to the teacher's context. The student never sees it -- it
answers the bare query. So the hint is not a leak risk in the usual sense; the student
cannot copy text it was never shown. What the hint controls is how much better the
teacher's next-token distribution is than the student's, and that gap IS the gradient.

Every hint is generated per-rollout by an LLM that reads the draft the student actually
produced, compares it against the rubric, and writes a hint targeting where THAT draft
fell short. Two rollouts of the same task get different hints: a draft that missed the
funding-risk angle is told about funding risk; one that missed the margin math is told
something else.

There is no static rubric-derived hint. An earlier version built hints from the rubric
alone via regex sanitization, which meant every rollout of a task received a byte-identical
hint regardless of what the model wrote -- a deliberately weak teacher, and the project's
stated main risk. That path is gone.

THE ABLATION. Two prompts receive identical inputs and differ only in whether the hint may
state the answer:

  answer_free     -- names neglected reasoning angles; no figures, no conclusions.
  answer_bearing  -- may cite the missed criteria verbatim, figures included.

`answer_free` is enforced by the PROMPT ALONE. There is no regex backstop: if the model
ignores the instruction, a figure reaches the teacher's context. That is a deliberate
trade -- the sanitizer that used to enforce it was static-hint machinery and went with it.

Note what `answer_bearing` changes conceptually: a teacher conditioned on the answer key is
no longer "the same model that knows slightly more" but "a model reading the answers,"
which is closer to supervised distillation than self-distillation. That is the point of the
arm, but a win there measures something different and should be reported as such.

RISK: a hint call that fails produces no hint, and a trajectory with no hint is DROPPED
rather than trained on (see train/rollout.py) -- an unhinted teacher is identical to the
student, so the trajectory would contribute ~zero gradient. During an API outage the
effective batch size shrinks rather than the gradient silently degrading.
"""

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
