"""Tests for the rubric judge, using a fake generate_fn so no API key or network is needed.

These verify the wiring against the real `rubric` package: that our criteria reach the
grader in rubric order, that section spans recover per-section scores correctly, and that
a judge failure degrades to an error result rather than killing the caller.

The second half covers the OpenRouter HTTP layer with `requests.post` monkeypatched. That
layer is worth testing directly because the scoring tests below bypass `RubricJudge.__init__`
entirely, so they would keep passing even if the transport were completely broken.
"""

import asyncio
import json

import pytest
import requests
from rubric import CriterionEvaluation, OneShotOutput

from data.dataset import Task
from reward.judge import (
    SCHEMA,
    OpenRouterOneShotGenerator,
    RubricJudge,
    TerminalJudgeError,
    summarize,
)

SECTIONS = [
    {"id": "factual-accuracy", "criteria": [
        {"weight": 10, "requirement": "States deposit cost was 0.39%"},
        {"weight": 10, "requirement": "States balance was $273.4B"},
    ]},
    {"id": "analytical-reasoning", "criteria": [
        {"weight": 4, "requirement": "Connects deposit growth to funding need"},
    ]},
    {"id": "risk-awareness", "criteria": [
        {"weight": 6, "requirement": "Identifies deposit beta as uncertain"},
    ]},
]


def _task() -> Task:
    from rubric import Rubric
    parsed = Rubric.from_dict({"sections": SECTIONS})
    spans, cursor = {}, 0
    for section in SECTIONS:
        n = len(section["criteria"])
        spans[section["id"]] = (cursor, cursor + n)
        cursor += n
    return Task(
        task_id="1", query="Assess the funding base.",
        sections=SECTIONS, criteria=parsed.rubric, section_spans=spans,
    )


def _fake_generator(met_indices: set[int]):
    """Return a generate_fn marking the given 1-based criterion numbers MET."""
    async def generate(system_prompt: str, user_prompt: str, **kwargs) -> OneShotOutput:
        return OneShotOutput(criteria_evaluations=[
            CriterionEvaluation(
                criterion_number=i,
                criterion_status="MET" if i in met_indices else "UNMET",
                explanation="test",
            )
            for i in range(1, 5)
        ])
    return generate


def _judge(generate) -> RubricJudge:
    judge = RubricJudge.__new__(RubricJudge)
    from rubric.autograders import PerCriterionOneShotGrader
    judge.grader = PerCriterionOneShotGrader(generate_fn=generate, normalize=True)
    judge.semaphore = asyncio.Semaphore(4)
    return judge


def test_perfect_answer_scores_one():
    result = asyncio.run(_judge(_fake_generator({1, 2, 3, 4})).score(_task(), "answer"))
    assert result.error is None
    assert result.score == pytest.approx(1.0)
    assert result.raw_score == pytest.approx(30.0)  # 10+10+4+6


def test_empty_answer_scores_zero():
    result = asyncio.run(_judge(_fake_generator(set())).score(_task(), ""))
    assert result.score == pytest.approx(0.0)


def test_good_answer_beats_bad_answer():
    """The smoke test that matters: a miswired judge often scores everything the same."""
    good = asyncio.run(_judge(_fake_generator({1, 2, 3, 4})).score(_task(), "good"))
    bad = asyncio.run(_judge(_fake_generator({4})).score(_task(), "bad"))
    assert good.score > bad.score


def test_section_scores_are_recovered_positionally():
    """Criterion ids are dropped by the rubric package, so sections are sliced by index.
    Getting this wrong silently attributes scores to the wrong section.
    """
    # Mark only the two factual criteria MET.
    result = asyncio.run(_judge(_fake_generator({1, 2})).score(_task(), "facts only"))
    assert result.sections["factual-accuracy"].fraction == pytest.approx(1.0)
    assert result.sections["analytical-reasoning"].fraction == pytest.approx(0.0)
    assert result.sections["risk-awareness"].fraction == pytest.approx(0.0)


def test_section_weights_match_the_rubric():
    result = asyncio.run(_judge(_fake_generator({1, 2, 3, 4})).score(_task(), "a"))
    assert result.sections["factual-accuracy"].possible == pytest.approx(20.0)
    assert result.sections["analytical-reasoning"].possible == pytest.approx(4.0)
    assert result.sections["risk-awareness"].possible == pytest.approx(6.0)


def test_judge_failure_is_captured_not_raised():
    """A judge outage must degrade the eval curve, not kill a training run."""
    async def broken(system_prompt: str, user_prompt: str, **kwargs) -> OneShotOutput:
        raise RuntimeError("api down")

    result = asyncio.run(_judge(broken).score(_task(), "answer"))
    assert result.error is not None and "api down" in result.error
    assert result.score == 0.0


def test_score_all_runs_concurrently():
    task = _task()
    pairs = [(task, f"answer {i}") for i in range(5)]
    results = asyncio.run(_judge(_fake_generator({1, 2, 3, 4})).score_all(pairs))
    assert len(results) == 5
    assert all(r.score == pytest.approx(1.0) for r in results)


def test_summarize_reports_mean_and_sections():
    task = _task()
    results = asyncio.run(_judge(_fake_generator({1, 2})).score_all([(task, "a"), (task, "b")]))
    metrics = summarize(results)
    assert metrics["judge_score"] == pytest.approx(20.0 / 30.0)
    assert metrics["judge_n"] == 2.0
    assert metrics["judge_errors"] == 0.0
    assert metrics["judge_factual-accuracy"] == pytest.approx(1.0)


def test_summarize_handles_all_errors():
    from reward.judge import JudgeResult
    metrics = summarize([JudgeResult("1", 0.0, 0.0, error="boom")])
    assert metrics["judge_score"] == 0.0 and metrics["judge_errors"] == 1.0


# --------------------------------------------------------------------------------------
# OpenRouter transport. `requests.post` is monkeypatched throughout -- no network, no key.
# --------------------------------------------------------------------------------------

VALID_CONTENT = json.dumps({
    "criteria_evaluations": [
        {"criterion_number": 1, "criterion_status": "MET", "explanation": "ok"},
    ]
})


class FakeResponse:
    """The slice of `requests.Response` that `_parse_response` actually touches."""

    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def _ok(content=VALID_CONTENT, finish_reason="stop"):
    return FakeResponse(body={
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    })


def _generator(monkeypatch, responses, **kwargs):
    """Build a generator whose POSTs return `responses` in order, recording each payload."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent = []
    queue = list(responses)

    def fake_post(**post_kwargs):
        sent.append(post_kwargs)
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "post", fake_post)
    return OpenRouterOneShotGenerator(**kwargs), sent


def test_missing_api_key_fails_at_construction(monkeypatch):
    """Better than surfacing as three retried failures per task and a zeroed eval curve."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(KeyError):
        OpenRouterOneShotGenerator()


def test_valid_response_parses(monkeypatch):
    gen, _ = _generator(monkeypatch, [_ok()])
    output = asyncio.run(gen("sys", "user"))
    assert isinstance(output, OneShotOutput)
    assert output.criteria_evaluations[0].criterion_status == "MET"


def test_request_payload_is_correct(monkeypatch):
    gen, sent = _generator(monkeypatch, [_ok()], model="stealth/ox-alpha")
    asyncio.run(gen("SYSTEM PROMPT", "USER PROMPT"))

    payload = json.loads(sent[0]["data"])
    assert payload["model"] == "stealth/ox-alpha"
    # A non-deterministic judge would inject sampling noise into the eval curve itself.
    assert payload["temperature"] == 0.0
    assert payload["reasoning"] == {"enabled": True}
    # Anthropic took `system` as a top-level kwarg; on the OpenAI wire it is a message.
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert payload["messages"][1] == {"role": "user", "content": "USER PROMPT"}
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    # Without this, a provider may ignore response_format and return prose.
    assert payload["provider"]["require_parameters"] is True
    assert sent[0]["headers"]["Authorization"] == "Bearer test-key"
    assert sent[0]["timeout"] == 120.0


def test_strict_schema_sets_additional_properties_false_everywhere():
    """OpenRouter rejects strict schemas missing this; pydantic does not emit it."""
    assert SCHEMA["additionalProperties"] is False
    defs = SCHEMA.get("$defs", {})
    assert defs, "expected CriterionEvaluation in $defs"
    for name, node in defs.items():
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, f"{name} missing the flag"


def test_malformed_json_is_retried_then_succeeds(monkeypatch):
    gen, sent = _generator(monkeypatch, [_ok(content="not json at all"), _ok()], max_retries=3)
    output = asyncio.run(gen("sys", "user"))
    assert len(sent) == 2
    assert output.criteria_evaluations[0].criterion_status == "MET"


def test_timeout_is_retried_then_surfaces(monkeypatch):
    gen, sent = _generator(
        monkeypatch, [requests.Timeout("slow")] * 3, max_retries=3
    )
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(gen("sys", "user"))
    assert len(sent) == 3


def test_server_error_is_retried(monkeypatch):
    gen, sent = _generator(monkeypatch, [FakeResponse(status_code=503), _ok()], max_retries=3)
    asyncio.run(gen("sys", "user"))
    assert len(sent) == 2


def test_auth_failure_is_terminal(monkeypatch):
    """A bad key must fail fast, not burn every retry plus backoff first."""
    gen, sent = _generator(
        monkeypatch, [FakeResponse(status_code=401, text="no")] * 3, max_retries=3
    )
    with pytest.raises(TerminalJudgeError, match="auth failed"):
        asyncio.run(gen("sys", "user"))
    assert len(sent) == 1


def test_bad_request_is_terminal(monkeypatch):
    """A rejected schema or unroutable provider filter fails identically on every retry."""
    gen, sent = _generator(
        monkeypatch, [FakeResponse(status_code=400, text="bad schema")] * 3, max_retries=3
    )
    with pytest.raises(TerminalJudgeError, match="rejected"):
        asyncio.run(gen("sys", "user"))
    assert len(sent) == 1


def test_rate_limit_is_retried_not_terminal(monkeypatch):
    """429 is a 4xx but genuinely transient."""
    gen, sent = _generator(monkeypatch, [FakeResponse(status_code=429), _ok()], max_retries=3)
    asyncio.run(gen("sys", "user"))
    assert len(sent) == 2


def test_error_inside_200_response_is_not_parsed_as_success(monkeypatch):
    """OpenRouter reports provider failures in-band, so raise_for_status misses them."""
    errored = FakeResponse(body={
        "choices": [{
            "message": {"content": "partial"},
            "finish_reason": "error",
            "error": {"code": 502, "message": "Provider disconnected"},
        }]
    })
    gen, _ = _generator(monkeypatch, [errored] * 3, max_retries=3)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(gen("sys", "user"))


def test_empty_content_is_not_parsed_as_success(monkeypatch):
    gen, _ = _generator(monkeypatch, [_ok(content=None, finish_reason="length")] * 3)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(gen("sys", "user"))


def test_top_level_error_body_is_handled(monkeypatch):
    gen, _ = _generator(
        monkeypatch, [FakeResponse(body={"error": {"message": "no endpoints"}})] * 3
    )
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        asyncio.run(gen("sys", "user"))
