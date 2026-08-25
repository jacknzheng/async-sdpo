"""Diligence TIR episode: mask search injections, stop on the memo."""

from __future__ import annotations

import asyncio

from data.dataset import Task
from data.diligence_harness import last_assistant_answer, run_diligence_episode
from data.tau_harness import AgentTurn, ToolCallSpec


def _encode(text: str) -> list[int]:
    return [ord(c) % 50 + 2 for c in text[:16]]


def _tokenize_chat(messages: list[dict], tools=None) -> list[int]:
    return [1, 2, 3]


def _task() -> Task:
    return Task(task_id="1", query="Assess the funding base.", sections=[])


def test_last_assistant_answer_skips_tool_calls():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "excerpts"},
        {"role": "assistant", "content": "The deposit beta is low."},
    ]
    assert last_assistant_answer(messages) == "The deposit beta is low."


def test_diligence_episode_masks_search_and_stops_on_memo():
    turns = [
        AgentTurn(
            token_ids=[10, 11],
            logprobs=[-0.1, -0.2],
            content=None,
            tool_calls=[ToolCallSpec(id="c1", name="web_search", arguments={"search_queries": ["x"]})],
            prompt_token_ids=[1, 2, 3],
        ),
        AgentTurn(
            token_ids=[20, 21, 22],
            logprobs=[-0.3, -0.4, -0.5],
            content="Funding is stable.",
            tool_calls=[],
        ),
    ]
    i = {"n": 0}

    async def generate_turn(messages, tools):
        turn = turns[i["n"]]
        i["n"] += 1
        return turn

    def execute_tool(tc):
        assert tc.name == "web_search"
        return "Acme 10-K excerpt"

    episode = asyncio.run(
        run_diligence_episode(
            _task(),
            generate_turn=generate_turn,
            encode=_encode,
            tokenize_chat=_tokenize_chat,
            execute_tool=execute_tool,
            max_steps=8,
        )
    )

    sampled = 5
    injected = _encode("\n[tool web_search]: Acme 10-K excerpt\n")
    assert sum(episode.loss_mask) == sampled
    assert 0 in episode.loss_mask
    assert episode.step_spans[0] == (0, 2)
    assert episode.step_spans[1] == (2 + len(injected), 2 + len(injected) + 3)
    assert len(episode.step_spans) == 2
    assert episode.termination == "agent_stop"
    assert episode.transcript == "Funding is stable."
    assert i["n"] == 2  # did not keep chatting after the memo
    for lp, m in zip(episode.rollout_logprobs, episode.loss_mask):
        if m == 0:
            assert lp == 0.0
