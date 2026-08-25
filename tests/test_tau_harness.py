"""Tau2 harness: episode loop, loss mask, gold teacher, mock domain.

The loop tests inject fake generate_turn / env / user so they run on a laptop with no
tau2, no vLLM, and no network. Real banking_knowledge tests are gated on the knowledge extra.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from data.dataset import Task, wrap_tau2_task
from data.tau_harness import (
    AgentTurn,
    SandboxNamespaceError,
    ToolCallSpec,
    assert_sandbox_ready,
    env_kwargs_for,
    format_transcript,
    gold_suffix,
    parse_tool_calls,
    run_episode,
)


def _encode(text: str) -> list[int]:
    return [ord(c) % 50 + 2 for c in text[:16]]


def _tokenize_chat(messages: list[dict], tools=None) -> list[int]:
    return [1, 2, 3]


def test_parse_qwen_tool_calls():
    text = (
        'Let me look that up.\n<tool_call>\n'
        '{"name": "KB_search_dense", "arguments": {"query": "card fee", "k": 10}}\n'
        "</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "KB_search_dense"
    assert calls[0].arguments["query"] == "card fee"


def test_parse_skips_malformed_blocks():
    assert parse_tool_calls("<tool_call>not json</tool_call>") == []
    assert parse_tool_calls("no tools here") == []


def test_env_kwargs_carry_solo_mode_so_evaluate_simulation_must_not():
    """Passing solo_mode both as a kwarg and inside env_kwargs TypeErrors and
    zeros retail/airline reward in the evaluate_episode except-handler."""
    retail = Task(task_id="r", query="x", sections=[], domain="retail", tau2_task=object())
    banking = Task(
        task_id="b", query="x", sections=[], domain="banking_knowledge", tau2_task=object()
    )
    assert env_kwargs_for(retail) == {"solo_mode": False}
    banking_kw = env_kwargs_for(banking)
    assert banking_kw["solo_mode"] is False
    assert "retrieval_variant" in banking_kw


def test_episode_masks_injected_spans_and_trains_on_sampled():
    """sum(loss_mask) == number of vLLM-sampled tokens; every injected span is 0."""

    turns = [
        AgentTurn(
            token_ids=[10, 11],
            logprobs=[-0.1, -0.2],
            content=None,
            tool_calls=[ToolCallSpec(id="c1", name="lookup", arguments={"id": "1"})],
            prompt_token_ids=[1, 2, 3],
        ),
        AgentTurn(
            token_ids=[20, 21, 22],
            logprobs=[-0.3, -0.4, -0.5],
            content="I found the order.",
            tool_calls=[],
        ),
    ]
    i = {"n": 0}

    async def generate_turn(messages, tools):
        turn = turns[i["n"]]
        i["n"] += 1
        return turn

    def execute_tool(tc: ToolCallSpec) -> str:
        return '{"status": "ok"}'

    async def user_reply(text: str):
        return "thanks, that's all ###STOP###", True

    episode = asyncio.run(
        run_episode(
            generate_turn=generate_turn,
            encode=_encode,
            tokenize_chat=_tokenize_chat,
            execute_tool=execute_tool,
            user_reply=user_reply,
            tools=[],
            system="policy",
            first_user="I need to exchange a keyboard",
            max_steps=8,
        )
    )

    sampled = len(turns[0].token_ids) + len(turns[1].token_ids)
    assert sum(episode.loss_mask) == sampled
    assert len(episode.loss_mask) == len(episode.response_token_ids) == len(episode.rollout_logprobs)
    assert episode.loss_mask[:2] == [1, 1]
    assert 0 in episode.loss_mask
    assert episode.termination == "user_stop"
    assert episode.step_spans[0] == (0, 2)
    assert len(episode.step_spans) == 2
    for lp, m in zip(episode.rollout_logprobs, episode.loss_mask):
        if m == 0:
            assert lp == 0.0


def test_gold_suffix_dumps_canonical_actions():
    action = SimpleNamespace(get_func_format=lambda: "exchange_delivered_order_items(order_id=W2378156)")
    tau = SimpleNamespace(
        id="0",
        required_documents=None,
        evaluation_criteria=SimpleNamespace(actions=[action]),
        user_scenario=SimpleNamespace(instructions="exchange a keyboard"),
    )
    task = wrap_tau2_task("retail", tau)
    hint = gold_suffix(task)
    assert "exchange_delivered_order_items" in hint
    assert "W2378156" in hint


def test_gold_suffix_empty_without_tau2_task():
    task = Task(task_id="1", query="q", sections=[])
    assert gold_suffix(task) == ""


def test_gold_hint_skips_the_llm(monkeypatch):
    from conftest import make_config
    from data.hint import generate_hint

    called = {"n": 0}

    async def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("gold path must not call the hint LLM")

    monkeypatch.setattr("data.hint.build_error_hint", boom)
    action = SimpleNamespace(get_func_format=lambda: "cancel_reservation()")
    tau = SimpleNamespace(
        id="7",
        required_documents=None,
        evaluation_criteria=SimpleNamespace(actions=[action]),
        user_scenario=SimpleNamespace(instructions="cancel"),
    )
    task = wrap_tau2_task("airline", tau)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="gold"), task, "transcript")
    )
    assert called["n"] == 0
    assert "cancel_reservation" in hints.free
    assert hints.ok("gold")


def test_step_hint_sends_gold_and_transcript(monkeypatch):
    from conftest import make_config
    from data.hint import generate_hint

    seen = {}

    async def hint(**kwargs):
        seen.update(kwargs)
        return "Call get_user_details with user_id=sophia_silva_7557."

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    action = SimpleNamespace(get_func_format=lambda: "get_user_details(user_id=sophia_silva_7557)")
    tau = SimpleNamespace(
        id="0",
        required_documents=None,
        evaluation_criteria=SimpleNamespace(actions=[action]),
        user_scenario=SimpleNamespace(instructions="lookup"),
    )
    task = wrap_tau2_task("retail", tau)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="step_hint"), task, "agent: hi")
    )
    assert seen["prompt_variant"] == "step_hint"
    assert "get_user_details" in seen["user_prompt"]
    assert "agent: hi" in seen["user_prompt"]
    assert hints.ok("step_hint")


def test_sdpo_loss_ignores_masked_interior_positions():
    """Positions with loss_mask 0 must not get a logprob gradient."""
    from train.loss import sdpo_loss

    student = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 9.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss, _ = sdpo_loss(student, teacher, None, mask, clipper=None)
    loss.backward()
    assert student.grad[0, 1].item() == 0.0
    assert student.grad[0, 0].item() != 0.0
    assert student.grad[0, 2].item() != 0.0


def test_format_transcript_skips_system():
    text = format_transcript(
        [
            {"role": "system", "content": "secret policy"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": "{\"id\": 1}"}}
                ],
            },
        ]
    )
    assert "secret policy" not in text
    assert "hello" in text
    assert "lookup" in text


def _tau2_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("tau2") is not None


def _knowledge_installed() -> bool:
    import importlib.util

    if not _tau2_installed():
        return False
    return importlib.util.find_spec("tau2.knowledge") is not None


@pytest.mark.tau2
@pytest.mark.skipif(not _tau2_installed(), reason="tau2 extra not installed")
def test_mock_domain_loads():
    from tau2.registry import registry

    tasks = registry.get_tasks_loader("mock")()
    assert tasks
    env = registry.get_env_constructor("mock")(solo_mode=False)
    assert env.get_policy()
    wrapped = wrap_tau2_task("mock", tasks[0])
    assert wrapped.task_id.startswith("mock:")
    assert wrapped.tau2_task is tasks[0]


@pytest.mark.knowledge
@pytest.mark.skipif(
    not _knowledge_installed(), reason="tau2[knowledge] extra not installed"
)
def test_banking_knowledge_loads_and_namespaces():
    from tau2.registry import registry

    tasks = registry.get_tasks_loader("banking_knowledge")()
    assert len(tasks) == 97
    wrapped = wrap_tau2_task("banking_knowledge", tasks[0])
    assert wrapped.task_id.startswith("banking_knowledge:")
    assert wrapped.domain == "banking_knowledge"
    titles = getattr(tasks[0], "required_documents", None) or []
    if titles:
        hint = gold_suffix(wrapped)
        assert "Gold knowledge" in hint


def test_sandbox_probe_skips_when_banking_not_in_domains(monkeypatch):
    monkeypatch.setattr("data.tau_harness.sys.platform", "linux")
    assert_sandbox_ready(["retail", "airline"])


def test_sandbox_probe_skips_on_macos(monkeypatch):
    monkeypatch.setattr("data.tau_harness.sys.platform", "darwin")
    monkeypatch.setattr("data.tau_harness.shutil.which", lambda _name: None)
    assert_sandbox_ready(["banking_knowledge"])


def test_sandbox_probe_missing_binaries(monkeypatch):
    monkeypatch.setattr("data.tau_harness.sys.platform", "linux")
    monkeypatch.setattr("data.tau_harness.shutil.which", lambda _name: None)
    with pytest.raises(SandboxNamespaceError, match="missing from PATH"):
        assert_sandbox_ready(["banking_knowledge"])


def test_sandbox_probe_namespace_denied(monkeypatch):
    monkeypatch.setattr("data.tau_harness.sys.platform", "linux")
    monkeypatch.setattr("data.tau_harness.shutil.which", lambda _name: f"/usr/bin/{_name}")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "bwrap: Creating new namespace failed: Operation not permitted"

    monkeypatch.setattr("data.tau_harness.subprocess.run", lambda *a, **k: _Proc())
    with pytest.raises(SandboxNamespaceError, match="container policy"):
        assert_sandbox_ready(["banking_knowledge"])


def test_sandbox_probe_ok(monkeypatch):
    monkeypatch.setattr("data.tau_harness.sys.platform", "linux")
    monkeypatch.setattr("data.tau_harness.shutil.which", lambda _name: f"/usr/bin/{_name}")

    class _Proc:
        returncode = 0
        stdout = "bwrap-ok\n"
        stderr = ""

    monkeypatch.setattr("data.tau_harness.subprocess.run", lambda *a, **k: _Proc())
    assert_sandbox_ready(["banking_knowledge"])
