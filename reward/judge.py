"""Rubric-based LLM judge for evaluating diligence-bench answers.

SCOPE: the judge's SCORING path is not part of the SDPO gradient. SDPO's loss is the
per-token reverse KL between the student and the hinted teacher -- it needs no reward
model, no verifier, and no human labels. `RubricJudge.score` exists only to produce eval
curves and diagnostics, and runs off the critical training path (held-out set at eval
intervals, never inside a training step).

The TRANSPORT below is a different matter. `data/hint.py` reuses `chat_completion` to
generate every teacher hint, and the hint IS the teacher's context, which is the gradient.
So this module is no longer purely off-the-gradient-path: the scoring path is, the shared
HTTP layer is not. Anything added here that changes `chat_completion`'s behavior changes
training, not just eval.

Built on the `rubric` package (PyPI `rubric`, The-LLM-Data-Company). Two notes from
reading that package's source:

* The class is `PerCriterionOneShotGrader` -- one LLM call scoring all ~36 criteria at
  once, rather than `PerCriterionGrader`'s one call per criterion. At 36 criteria x 30
  held-out tasks that is 30 calls per eval instead of ~1,080.
* `Criterion` keeps only `weight` and `requirement`; diligence-bench's criterion `id`s are
  dropped and sections are flattened on parse. Verdicts therefore come back positionally,
  which is why `Task.section_spans` (data/dataset.py) records the index ranges -- it is the
  only way to recover per-section scores.

The package ships a Gemini-backed `generate_fn`; `generate_fn` is a plain async protocol
with no model/client/key parameter, so we inject an OpenRouter-backed one below and own
auth, retries, and concurrency ourselves.

The transport is plain `requests` against OpenRouter's OpenAI-compatible
/chat/completions endpoint rather than a vendor SDK -- one POST is the entire surface we
need, and the judge is the only external-LLM egress point in the repo, so a vendor SDK
would be a dependency for a single call. `requests` is synchronous, so the call is run
via `asyncio.to_thread`: `score_all` fans out ~30 held-out tasks under a semaphore, and a
bare blocking POST inside a coroutine would serialize all of them and stall the rollout
workers sharing the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import requests
from rubric import EvaluationReport, OneShotOutput
from rubric.autograders import PerCriterionOneShotGrader

from data.diagnostics import artifact_event
from data.dataset import Task

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _strict_schema() -> dict:
    """`OneShotOutput`'s JSON schema with `additionalProperties: false` on every object.

    OpenRouter's strict mode requires that flag on all objects and pydantic does not emit
    it. Deriving the schema from the model rather than hand-writing it means a `rubric`
    upgrade that changes OneShotOutput cannot silently desync the judge's request.
    """
    schema = OneShotOutput.model_json_schema()
    for node in [schema, *schema.get("$defs", {}).values()]:  # $defs holds CriterionEvaluation
        if node.get("type") == "object":
            node["additionalProperties"] = False
    return schema


SCHEMA = _strict_schema()


@dataclass
class SectionScore:
    """Weighted score for one rubric section, recovered positionally via section spans."""

    section_id: str
    earned: float
    possible: float

    @property
    def fraction(self) -> float:
        return self.earned / self.possible if self.possible else 0.0


@dataclass
class JudgeResult:
    task_id: str
    score: float                      # 0-1, normalized by total positive weight
    raw_score: float                  # unnormalized weighted sum
    sections: dict[str, SectionScore] = field(default_factory=dict)
    error: str | None = None


class TerminalJudgeError(Exception):
    """A failure retrying cannot fix -- bad key, bad schema, unroutable model."""


async def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    api_key: str,
    max_tokens: int = 16000,
    timeout: float = 120.0,
    max_retries: int = 3,
    response_schema: dict | None = None,
    parse: Callable[[str], object] | None = None,
):
    """POST one chat completion to OpenRouter and return the content (parsed, if asked).

    Shared by the judge (which validates the result against a strict schema) and the hint
    generator in data/hint.py (which wants free text). `response_schema=None` omits both
    `response_format` and the provider filter, since requiring structured-output support
    would needlessly narrow routing for a plain-prose request.

    When a schema IS requested, models normally try strict `json_schema` +
    `provider.require_parameters`, then fall back to `json_object` if that route is
    unavailable. GLM uses `json_object` directly because OpenRouter's strict filter
    sends it to an overloaded shared pool; the parsed result is still schema-validated.
    Hints never take this path.

    `parse` runs INSIDE the retry loop. That placement matters: a schema violation is
    transient -- the model emitted prose or a truncated object, and a resample usually
    fixes it -- so it must be retried like any other transient failure rather than raised
    on the first attempt.

    Raises TerminalJudgeError for failures a retry cannot fix; retries everything else.
    """
    modes: list[str | None]
    if response_schema is not None and model == "z-ai/glm-5.3-flash":
        # OpenRouter's strict-schema filter currently routes GLM to a shared
        # DeepInfra pool that frequently returns upstream 429s. json_object
        # routes to Z.AI and is still validated below with
        # OneShotOutput.model_validate_json, so malformed output cannot score.
        modes = ["json_object"]
    elif response_schema is not None:
        modes = ["json_schema", "json_object"]
    else:
        modes = [None]

    last_error: Exception | None = None
    for i, mode in enumerate(modes):
        payload = _chat_payload(
            system_prompt,
            user_prompt,
            model=model,
            max_tokens=max_tokens,
            response_schema=response_schema,
            mode=mode,
        )
        try:
            return await _post_chat_completion(
                payload,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                parse=parse,
            )
        except TerminalJudgeError as exc:
            if i + 1 < len(modes) and _structured_output_unroutable(exc):
                logger.warning(
                    "strict json_schema unroutable on OpenRouter; falling back to json_object: %s",
                    exc,
                )
                last_error = exc
                continue
            raise
        except RuntimeError as exc:
            if i + 1 < len(modes):
                logger.warning(
                    "strict json_schema failed after retries; falling back to json_object: %s",
                    exc,
                )
                last_error = exc
                continue
            raise
    raise RuntimeError(f"openrouter call failed: {last_error}")


def _chat_payload(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    max_tokens: int,
    response_schema: dict | None,
    mode: str | None,
) -> dict:
    payload: dict = {
        "model": model,
        # Anthropic took the system prompt as a top-level kwarg; on the OpenAI wire format
        # it is just the first message.
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        # Deterministic by default: for the judge this keeps sampling noise out of the eval
        # metric, and for hints it keeps the same rollout from yielding a different teacher
        # on a rerun.
        "temperature": 0.0,
        "reasoning": {"enabled": True},
    }
    if mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "one_shot_output", "strict": True, "schema": response_schema},
        }
        # OpenRouter routes across backend providers and not all honor strict schemas.
        # Without this a provider may ignore response_format and return prose, which lands
        # as a parse failure or, worse, a wrongly-zero eval score.
        payload["provider"] = {"require_parameters": True}
    elif mode == "json_object":
        # Widely routed; the caller still validates against OneShotOutput.
        payload["response_format"] = {"type": "json_object"}
    return payload


def _structured_output_unroutable(exc: BaseException) -> bool:
    """OpenRouter 404s strict json_schema on models that otherwise chat fine."""
    text = str(exc).lower()
    return "no endpoints found" in text or "(404)" in text


async def _post_chat_completion(
    payload: dict,
    *,
    api_key: str,
    timeout: float,
    max_retries: int,
    parse: Callable[[str], object] | None,
):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def post() -> requests.Response:
        return requests.post(
            url=OPENROUTER_URL, headers=headers, data=json.dumps(payload), timeout=timeout
        )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        response: requests.Response | None = None
        try:
            response = await asyncio.to_thread(post)
            content = _extract_content(response)
            return parse(content) if parse is not None else content
        except TerminalJudgeError as exc:
            artifact_event(
                "api_failures",
                "api_call_failed",
                provider="openrouter",
                operation="chat_completion",
                model=payload.get("model"),
                response_mode=(payload.get("response_format") or {}).get("type", "text"),
                attempt=attempt + 1,
                max_retries=max_retries,
                terminal=True,
                status_code=getattr(response, "status_code", None),
                response=(getattr(response, "text", "") or "")[:4000],
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        except Exception as exc:  # noqa: BLE001 -- retried below, re-raised if terminal
            last_error = exc
            artifact_event(
                "api_failures",
                "api_call_failed",
                provider="openrouter",
                operation="chat_completion",
                model=payload.get("model"),
                response_mode=(payload.get("response_format") or {}).get("type", "text"),
                attempt=attempt + 1,
                max_retries=max_retries,
                terminal=False,
                status_code=getattr(response, "status_code", None),
                response=(getattr(response, "text", "") or "")[:4000],
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0**attempt)
    raise RuntimeError(f"openrouter call failed after {max_retries} attempts: {last_error}")


def _extract_content(response: requests.Response) -> str:
    """Pull the message content out of an OpenRouter response, classifying failures.

    Everything here is provider-level and schema-agnostic, which is what lets the judge and
    the hint generator share it.
    """
    if response.status_code in (401, 403):
        raise TerminalJudgeError(f"openrouter auth failed ({response.status_code}): {response.text}")
    # 429 and 408 are 4xx but genuinely transient; everything else in that range means the
    # request itself is malformed (bad model slug, bad schema, unroutable provider filter)
    # and will fail identically on every retry.
    if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
        raise TerminalJudgeError(
            f"openrouter rejected the request ({response.status_code}): {response.text}"
        )
    response.raise_for_status()

    body = response.json()
    # OpenRouter can report a provider failure inside a 200 response, so raise_for_status
    # above does not catch it.
    if "error" in body and not body.get("choices"):
        raise RuntimeError(f"openrouter returned an error: {body['error']}")

    choice = body["choices"][0]
    if choice.get("finish_reason") == "error":
        raise RuntimeError(f"generation errored: {choice.get('error')}")

    # `reasoning_details` may also be present; both callers are single-shot, so there is no
    # follow-up turn to feed it into and it is ignored.
    content = choice["message"].get("content")
    if not content:
        raise ValueError(f"no content returned (finish_reason={choice.get('finish_reason')})")
    return content


class OpenRouterOneShotGenerator:
    """`generate_fn` for PerCriterionOneShotGrader, backed by OpenRouter.

    The protocol is `async (system_prompt, user_prompt, **kwargs) -> OneShotOutput`, and
    the package's docstring is explicit that parsing, validation, and retries are the
    caller's job. `response_format` with a strict JSON schema handles the first, and
    `OneShotOutput.model_validate_json` the second; the retry loop is below.
    """

    def __init__(
        self,
        model: str = "z-ai/glm-5.3-flash",
        max_retries: int = 3,
        max_tokens: int = 16000,
        timeout: float = 120.0,
    ) -> None:
        # Raise at construction rather than surfacing a missing key as three retried
        # failures per task and a silently zeroed eval curve.
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def __call__(self, system_prompt: str, user_prompt: str, **kwargs) -> OneShotOutput:
        return await chat_completion(
            system_prompt,
            user_prompt,
            model=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
            response_schema=SCHEMA,
            parse=OneShotOutput.model_validate_json,
        )


class RubricJudge:
    """Scores answers against diligence-bench rubrics, with bounded concurrency."""

    def __init__(
        self,
        model: str = "z-ai/glm-5.3-flash",
        max_concurrency: int = 8,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.grader = PerCriterionOneShotGrader(
            generate_fn=OpenRouterOneShotGenerator(
                model=model, max_retries=max_retries, timeout=timeout
            ),
            # normalize=True divides by total positive weight, which is what we want:
            # diligence-bench weights are unnormalized and row totals range 115-370, so
            # raw sums are not comparable across tasks. `raw_score` keeps the sum anyway.
            normalize=True,
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def score(self, task: Task, answer: str) -> JudgeResult:
        """Score one answer. Failures are captured, not raised -- a judge outage should
        degrade the eval curve, not kill a training run."""
        async with self.semaphore:
            try:
                report: EvaluationReport = await self.grader.grade(
                    to_grade=answer,
                    rubric=task.criteria,
                    query=task.query,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "judge failed for task %s: %s: %s",
                    task.task_id,
                    type(exc).__name__,
                    exc,
                )
                artifact_event(
                    "api_failures",
                    "judge_failed",
                    provider="openrouter",
                    operation="rubric_judge",
                    task_id=task.task_id,
                    model=getattr(self, "model", "unknown"),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    query=task.query,
                    answer=answer,
                )
                return JudgeResult(task_id=task.task_id, score=0.0, raw_score=0.0, error=str(exc))

        return JudgeResult(
            task_id=task.task_id,
            score=report.score,
            raw_score=report.raw_score if report.raw_score is not None else 0.0,
            sections=_section_scores(task, report),
        )

    async def score_all(self, pairs: list[tuple[Task, str]]) -> list[JudgeResult]:
        """Score many (task, answer) pairs concurrently, bounded by the semaphore."""
        return await asyncio.gather(*(self.score(task, answer) for task, answer in pairs))


def _section_scores(task: Task, report: EvaluationReport) -> dict[str, SectionScore]:
    """Recover per-section scores by slicing the flattened per-criterion reports.

    `PerCriterionOneShotGrader.judge` builds its report list in rubric order and pads any
    criterion the judge skipped with an UNMET entry, so positional slicing is sound. Worth
    the bookkeeping because factual-accuracy carries 73% of the weight -- a run that lifts
    only the reasoning sections would otherwise be invisible in the aggregate score.
    """
    if not report.report:
        return {}

    sections: dict[str, SectionScore] = {}
    for section_id, (start, end) in task.section_spans.items():
        entries = report.report[start:end]
        earned = sum(e.weight for e in entries if e.verdict == "MET" and e.weight > 0)
        possible = sum(e.weight for e in entries if e.weight > 0)
        sections[section_id] = SectionScore(section_id, earned, possible)
    return sections


def summarize(results: list[JudgeResult]) -> dict[str, float]:
    """Aggregate judge results into the metrics the trainer logs."""
    scored = [r for r in results if r.error is None]
    if not scored:
        return {"judge_score": 0.0, "judge_errors": float(len(results))}

    metrics = {
        "judge_score": sum(r.score for r in scored) / len(scored),
        "judge_errors": float(len(results) - len(scored)),
        "judge_n": float(len(scored)),
    }
    for section_id in ("factual-accuracy", "analytical-reasoning", "risk-awareness"):
        fractions = [r.sections[section_id].fraction for r in scored if section_id in r.sections]
        if fractions:
            metrics[f"judge_{section_id}"] = sum(fractions) / len(fractions)
    return metrics
