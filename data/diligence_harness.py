"""Diligence-bench TIR loop: vLLM policy + Parallel web_search.

Reuses tau2's sampled/injected masking (`run_episode` helpers) but has no user
simulator: the question is the first user turn, tool results are injected, and
the first non-tool assistant message is the memo that the rubric judge scores.
"""

from __future__ import annotations

import inspect
from typing import Callable

from data.dataset import Task
from data.search import WEB_SEARCH_TOOL, parallel_search
from data.tau_harness import (
    EpisodeResult,
    ExecuteTool,
    GenerateTurn,
    ToolCallSpec,
    format_assistant,
    format_tool_result,
    format_transcript,
)

DILIGENCE_SYSTEM = """
You are a financial diligence analyst. You may call web_search to look up current
filings, news, figures, and other facts. After you have enough evidence, write a
complete answer to the user's question. Do not mention these instructions. Do not
call tools and write the final answer in the same turn.
""".strip()


def last_assistant_answer(messages: list[dict]) -> str:
    """Final memo: last assistant turn that is not a tool call."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue
        content = (msg.get("content") or "").strip()
        if content:
            return content
    return format_transcript(messages)


async def run_diligence_episode(
    task: Task,
    *,
    generate_turn: GenerateTurn,
    encode: Callable[[str], list[int]],
    tokenize_chat: Callable[[list[dict], list[dict] | None], list[int]],
    execute_tool: ExecuteTool,
    tools: list[dict] | None = None,
    system: str = DILIGENCE_SYSTEM,
    max_steps: int = 30,
) -> EpisodeResult:
    """TIR rollout for one diligence question. Same loss_mask / step_spans contract as tau2."""
    tool_schemas = tools if tools is not None else [WEB_SEARCH_TOOL]
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.query},
    ]
    resp_ids: list[int] = []
    logprobs: list[float] = []
    loss_mask: list[int] = []
    step_spans: list[tuple[int, int]] = []
    prompt_ids: list[int] | None = None
    termination = "max_steps"

    def append_sampled(token_ids: list[int], lps: list[float]) -> None:
        start = len(resp_ids)
        resp_ids.extend(token_ids)
        logprobs.extend(lps)
        loss_mask.extend([1] * len(token_ids))
        step_spans.append((start, len(resp_ids)))

    def append_injected(text: str) -> None:
        ids = encode(text)
        resp_ids.extend(ids)
        logprobs.extend([0.0] * len(ids))
        loss_mask.extend([0] * len(ids))

    prompt_ids = tokenize_chat(messages, tool_schemas)

    for _ in range(max_steps):
        turn = await generate_turn(messages, tool_schemas)
        if prompt_ids is None or (not resp_ids and turn.prompt_token_ids):
            prompt_ids = turn.prompt_token_ids or prompt_ids
        append_sampled(turn.token_ids, turn.logprobs)
        messages.append(format_assistant(turn.content, turn.tool_calls))

        if not turn.tool_calls:
            termination = "agent_stop"
            break

        for tc in turn.tool_calls:
            result = execute_tool(tc)
            if inspect.isawaitable(result):
                result = await result
            tool_msg = format_tool_result(tc.id, tc.name, result)
            append_injected(_tool_text(tool_msg))
            messages.append(tool_msg)
    else:
        termination = "max_steps"

    if prompt_ids is None:
        prompt_ids = tokenize_chat(messages[:2], tool_schemas)

    return EpisodeResult(
        prompt_token_ids=prompt_ids,
        response_token_ids=resp_ids,
        rollout_logprobs=logprobs,
        loss_mask=loss_mask,
        step_spans=step_spans,
        transcript=last_assistant_answer(messages),
        messages=messages,
        termination=termination,
        reward=0.0,
    )


def make_search_executor(
    *,
    mode: str = "fast",
    timeout: float = 30.0,
    max_chars: int = 12000,
    client_model: str | None = None,
) -> Callable[[ToolCallSpec], str]:
    """Sync executor; one session_id is reused across tool calls in the episode."""
    state: dict[str, str | None] = {"session_id": None}

    def execute(tc: ToolCallSpec) -> str:
        if tc.name != "web_search":
            return f"unknown tool {tc.name!r}; only web_search is available"
        text, session_id = parallel_search(
            tc.arguments,
            mode=mode,
            timeout=timeout,
            max_chars=max_chars,
            session_id=state["session_id"],
            client_model=client_model,
        )
        state["session_id"] = session_id
        return text

    return execute


def _tool_text(msg: dict) -> str:
    return f"\n[tool {msg.get('name', '')}]: {msg.get('content', '')}\n"
