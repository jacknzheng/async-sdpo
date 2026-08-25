"""Tests for dataset loading, splitting, and hint construction.

Split into two groups: hint tests run offline on fixtures, while tests marked `network`
hit the real dataset. The no-digits invariant is checked against all 150 real rows,
because a regex that sanitizes a fixture but leaks on row 87 is worse than no regex.
"""

import asyncio
import json

import pytest

from data.dataset import Task, build_prompt, load_tasks, split_tasks
from data.hint import build_error_hint

# Real criteria text sampled from the dataset, covering the leak patterns that matter:
# currency, percentages, ranges, quarters/halves, bare years, and prose conclusions.
FIXTURE_SECTIONS = [
    {
        "id": "factual-accuracy",
        "title": "Factual Accuracy",
        "criteria": [
            {"id": "a", "weight": 10,
             "requirement": "States interest-bearing deposit cost was 0.39% in Q1 2021"},
            {"id": "b", "weight": 10,
             "requirement": "States Electric segment revenue as $1,968M and its share as 61% of total utility revenue"},
            {"id": "c", "weight": 18,
             "requirement": "Estimates annualized interest expense savings of approximately $2.0-2.5B (82bps x ~$273B)"},
        ],
    },
    {
        "id": "analytical-reasoning",
        "title": "Analytical Reasoning",
        "criteria": [
            {"id": "d", "weight": 4,
             "requirement": "Synthesizes that Electric's 61% revenue share masks inferior earnings quality, concluding revenue scale does not equal earnings quality"},
            {"id": "e", "weight": 4,
             "requirement": "Identifies purchased power as non-controllable commodity pass-through creating volumetric and price volatility between rate cases"},
            {"id": "f", "weight": 3,
             "requirement": "Connects VELA failure to impaired ability to fund IZAR through H1 2026 via dilutive financing"},
        ],
    },
    {
        "id": "risk-awareness",
        "title": "Risk Awareness",
        "criteria": [
            {"id": "g", "weight": 6,
             "requirement": "Identifies regulatory lag between rate cases (filed every 3-4 years) as a risk compressing Electric's thin margin"},
            {"id": "h", "weight": 5,
             "requirement": "Notes uncertainty about whether the $32.3B deposit growth is core retail vs rate-sensitive or brokered"},
        ],
    },
]


def _fixture_task() -> Task:
    return Task(
        task_id="1",
        query="Assess the funding base.",
        sections=FIXTURE_SECTIONS,
    )


HINT = "Guidance for answering this question:\n- how deposit cost trended against balances\n"


# ---- prompt construction (offline) ----

def test_student_prompt_excludes_hint():
    """The student must never see the hint -- this is structural to SDPO."""
    student = build_prompt(_fixture_task(), hint=None)
    assert HINT not in student
    assert "Assess the funding base." in student


def test_teacher_prompt_includes_hint():
    task = _fixture_task()
    teacher = build_prompt(task, hint=HINT)
    assert HINT in teacher and task.query in teacher


def test_teacher_and_student_prompts_differ_only_by_hint():
    task = _fixture_task()
    student = build_prompt(task, None)
    teacher = build_prompt(task, HINT)
    assert teacher.startswith(student)
    assert teacher != student



# ---- split logic (offline) ----

def _fake_tasks(n: int = 150) -> list[Task]:
    return [Task(task_id=str(i), query=f"q{i}", sections=[]) for i in range(1, n + 1)]


def test_split_sizes_and_disjointness():
    train, heldout = split_tasks(_fake_tasks(), n_heldout=30, seed=0)
    assert len(train) == 120 and len(heldout) == 30
    train_ids = {t.task_id for t in train}
    heldout_ids = {t.task_id for t in heldout}
    assert not (train_ids & heldout_ids)
    assert len(train_ids | heldout_ids) == 150


def test_namespaced_ids_split():
    """Banking-style ids are not ints; the carve must still be 70/27 and disjoint."""
    tasks = [
        Task(task_id=f"banking_knowledge:task_{i:03d}", query=f"q{i}", sections=[])
        for i in range(1, 98)
    ]
    train, heldout = split_tasks(tasks, n_heldout=27, seed=0)
    assert len(train) == 70 and len(heldout) == 27
    assert not ({t.task_id for t in train} & {t.task_id for t in heldout})


def test_wrap_tau2_namespaces_colliding_ids():
    from types import SimpleNamespace

    from data.dataset import wrap_tau2_task

    retail = wrap_tau2_task(
        "retail",
        SimpleNamespace(id="0", user_scenario=SimpleNamespace(instructions="exchange a keyboard")),
    )
    airline = wrap_tau2_task(
        "airline",
        SimpleNamespace(id="0", user_scenario=SimpleNamespace(instructions="cancel a flight")),
    )
    assert retail.task_id == "retail:0"
    assert airline.task_id == "airline:0"
    assert retail.domain == "retail"
    assert airline.domain == "airline"


def test_split_is_deterministic_across_calls():
    a_train, a_held = split_tasks(_fake_tasks(), 30, seed=0)
    b_train, b_held = split_tasks(_fake_tasks(), 30, seed=0)
    assert [t.task_id for t in a_held] == [t.task_id for t in b_held]
    assert [t.task_id for t in a_train] == [t.task_id for t in b_train]


def test_split_is_independent_of_input_order():
    """Split must depend on the seed alone, not on dataset load order."""
    tasks = _fake_tasks()
    _, held_a = split_tasks(tasks, 30, seed=0)
    _, held_b = split_tasks(list(reversed(tasks)), 30, seed=0)
    assert {t.task_id for t in held_a} == {t.task_id for t in held_b}


def test_different_seeds_give_different_splits():
    _, held_a = split_tasks(_fake_tasks(), 30, seed=0)
    _, held_b = split_tasks(_fake_tasks(), 30, seed=1)
    assert {t.task_id for t in held_a} != {t.task_id for t in held_b}


def test_split_rejects_impossible_heldout():
    with pytest.raises(ValueError):
        split_tasks(_fake_tasks(10), n_heldout=10)


# ---- real dataset (network) ----

@pytest.mark.network
def test_real_dataset_shape():
    """Loads all 150 rows and checks the section spans tile the flattened criteria.

    Hints are no longer built here -- they are generated per rollout -- so there is no
    static hint to validate across the dataset. What still matters is the span
    bookkeeping: get it wrong and per-section judge scores silently misreport.
    """
    tasks = load_tasks()
    assert len(tasks) == 150
    assert len({t.task_id for t in tasks}) == 150

    for task in tasks:
        assert set(task.section_spans) == {
            "factual-accuracy", "analytical-reasoning", "risk-awareness"
        }
        # Spans must tile the flattened criteria exactly, or per-section scores misreport.
        covered = sorted(task.section_spans.values())
        assert covered[0][0] == 0
        assert covered[-1][1] == len(task.criteria)
        for (_, end), (start, _) in zip(covered, covered[1:]):
            assert end == start


@pytest.mark.network
def test_real_split_is_120_30():
    train, heldout = split_tasks(load_tasks(), n_heldout=30, seed=0)
    assert len(train) == 120 and len(heldout) == 30


# ---- error-conditioned hints (offline; requests.post is monkeypatched) ----

DRAFT = "The bank looks fine. Deposits grew and funding seems stable."


def _stub_openrouter(monkeypatch, content, status_code=200):
    """Make every OpenRouter POST return `content`, recording the payloads sent."""
    import requests

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent = []

    class Resp:
        def __init__(self):
            self.status_code = status_code
            self.text = ""

        def json(self):
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def fake_post(**kwargs):
        sent.append(kwargs)
        if isinstance(content, Exception):
            raise content
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    return sent


def _build(prompt_variant="answer_free"):
    return asyncio.run(
        build_error_hint(
            query="Assess the funding base.",
            sections=FIXTURE_SECTIONS,
            response_text=DRAFT,
            prompt_variant=prompt_variant,
        )
    )


def test_answer_free_hint_passes_compliant_output_through(monkeypatch):
    """A model that follows the instruction yields a digit-free hint."""
    _stub_openrouter(monkeypatch, (
        "- how interest-bearing deposit cost trended against deposit balance growth\n"
        "- whether recent deposit growth is core retail or rate-sensitive funding\n"
    ))
    hint = _build("answer_free")
    assert hint
    assert not any(c.isdigit() for c in hint)
    assert "deposit balance growth" in hint


def test_answer_free_is_prompt_enforced_only(monkeypatch):
    """DOCUMENTS A KNOWN GAP, not a desired behavior.

    The regex sanitizer that used to strip leaked figures was static-hint machinery and
    was removed with it. `answer_free` now rests entirely on the prompt instruction, so a
    model that ignores "NEVER state a figure" puts that figure into the teacher's context.
    If this test ever needs to be inverted, a sanitizer has been reintroduced.
    """
    _stub_openrouter(monkeypatch, "- the $32.3B deposit growth was 61% of the total\n")
    hint = _build("answer_free")
    assert "32.3" in hint, "unsanitized output should pass through verbatim"


def test_answer_bearing_hint_keeps_figures(monkeypatch):
    """The other arm exists precisely to let the answer through."""
    _stub_openrouter(monkeypatch, (
        "- You omitted that interest-bearing deposit cost was 0.39% in Q1 2021.\n"
        "- You did not quantify the $32.3B deposit growth.\n"
    ))
    hint = _build("answer_bearing")
    assert "0.39%" in hint and "$32.3B" in hint


def test_two_prompts_send_different_instructions(monkeypatch):
    """The ablation is the system prompt; everything else must be identical."""
    sent_free = _stub_openrouter(monkeypatch, "- how deposit cost trended against balances")
    _build("answer_free")
    free_payload = json.loads(sent_free[0]["data"])

    sent_bearing = _stub_openrouter(monkeypatch, "- deposit cost was 0.39%")
    _build("answer_bearing")
    bearing_payload = json.loads(sent_bearing[0]["data"])

    assert free_payload["messages"][0] != bearing_payload["messages"][0]
    assert "NEVER state a figure" in free_payload["messages"][0]["content"]
    assert "Cite the specific facts" in bearing_payload["messages"][0]["content"]
    # Same rollout and rubric reach both arms -- only the instruction differs.
    assert free_payload["messages"][1] == bearing_payload["messages"][1]
    assert DRAFT in free_payload["messages"][1]["content"]


def test_hint_request_sends_no_response_format(monkeypatch):
    """Hints are free text. Requiring structured-output support would narrow routing for
    no benefit."""
    sent = _stub_openrouter(monkeypatch, "- how deposit cost trended against balances")
    _build("answer_free")
    payload = json.loads(sent[0]["data"])
    assert "response_format" not in payload
    assert "provider" not in payload
    assert payload["temperature"] == 0.0


def test_api_failure_returns_empty_not_raises(monkeypatch):
    """A hint failure must never propagate -- the caller keeps the static hint."""
    import requests
    _stub_openrouter(monkeypatch, requests.Timeout("slow"))
    assert _build("answer_free") == ""


def test_missing_key_returns_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _build("answer_free") == ""


def test_blank_generation_returns_empty(monkeypatch):
    """"" is the signal to DROP the trajectory -- an empty hint would make the teacher
    identical to the student and its gradient ~zero."""
    _stub_openrouter(monkeypatch, "   \n  \n")
    assert _build("answer_free") == ""


def test_unknown_prompt_variant_raises(monkeypatch):
    _stub_openrouter(monkeypatch, "- anything")
    with pytest.raises(ValueError, match="unknown hint prompt"):
        _build("answer_leaking")


def test_rubric_reaches_the_hint_model_in_full(monkeypatch):
    """Both arms need the answer key to know what the draft missed; only the ANSWER-FREE
    arm's OUTPUT is restricted, not its input."""
    sent = _stub_openrouter(monkeypatch, "- how deposit cost trended against balances")
    _build("answer_free")
    user_prompt = json.loads(sent[0]["data"])["messages"][1]["content"]
    assert "0.39%" in user_prompt      # factual-accuracy criteria are included
    assert "risk-awareness" in user_prompt
