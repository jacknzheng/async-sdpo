"""Tau2 episode loop: Environment + user simulator + evaluate_simulation.

We do not use tau2's AgentGymEnv -- it `wait()`s on Python threads (fights asyncio +
vLLM) and never forwards `retrieval_variant="alltools-qwen"`. This module is the
while-loop that sits between env, user sim, and the vLLM policy.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from data.dataset import Task
from data.diagnostics import artifact_event

logger = logging.getLogger(__name__)

# tau2's banking env only checks that these names exist on PATH. That is not
# enough: GPU pods often install bwrap but seccomp-block `unshare`, so every
# `shell` call then dies with "Creating new namespace failed: Operation not permitted".
_SANDBOX_BINS = ("srt", "rg", "bwrap", "socat")
_BWRAP_PROBE = (
    "bwrap",
    "--ro-bind",
    "/",
    "/",
    "--dev",
    "/dev",
    "/bin/echo",
    "bwrap-ok",
)
_SRT_PROBE = ("srt", "-c", "echo srt-ok")
SANDBOX_NAMESPACE_HINT = """
bwrap cannot create a Linux namespace on this host
(typically: Creating new namespace failed: Operation not permitted).

This is a container policy issue, not a missing install. Recreate the pod/container
so nested namespaces are allowed:

  docker:  --privileged
       or  --security-opt seccomp=unconfined --security-opt apparmor=unconfined
  RunPod:  Edit Pod -> extra flags -> --privileged, then Start
  Baseten: request a privileged / unconfined-seccomp workstation

Then `bash scripts/setup_tau2_sandbox.sh` should print bwrap: ok and srt-ok.
BM25 / dense search still work without this; only the sandboxed shell is dead.
""".strip()


class SandboxNamespaceError(RuntimeError):
    """Banking `shell` cannot run: binaries missing, or namespaces blocked."""


def assert_sandbox_ready(domains: list[str]) -> None:
    """Fail loud before a tau2 run if banking shell cannot actually execute.

    Skip when banking is not in `domains`, and on macOS (sandbox-exec, not bwrap).
    """
    if "banking_knowledge" not in domains:
        artifact_event(
            "sandbox",
            "sandbox_preflight_skipped",
            domains=domains,
            reason="banking_knowledge_not_requested",
        )
        return
    if sys.platform == "darwin":
        logger.info("tau2 sandbox: macOS uses sandbox-exec; skipping Linux namespace probe")
        artifact_event(
            "sandbox",
            "sandbox_preflight_skipped",
            domains=domains,
            platform=sys.platform,
            reason="macos_uses_sandbox_exec",
        )
        return
    resolved = {name: shutil.which(name) for name in _SANDBOX_BINS}
    missing = [name for name, path in resolved.items() if path is None]
    artifact_event(
        "sandbox",
        "sandbox_preflight_started",
        domains=domains,
        platform=sys.platform,
        binaries=resolved,
        probes=[list(_BWRAP_PROBE), list(_SRT_PROBE)],
    )
    if missing:
        artifact_event(
            "sandbox",
            "sandbox_preflight_failed",
            cause="missing_binaries",
            missing=missing,
            binaries=resolved,
        )
        raise SandboxNamespaceError(
            "tau2 banking sandbox binaries missing from PATH: "
            f"{missing}. Run: bash scripts/setup_tau2_sandbox.sh"
        )
    try:
        proc = subprocess.run(
            _BWRAP_PROBE, capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        artifact_event(
            "sandbox",
            "sandbox_preflight_failed",
            cause="probe_exception",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise SandboxNamespaceError(
            f"bwrap namespace probe could not run: {exc}\n\n{SANDBOX_NAMESPACE_HINT}"
        ) from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        artifact_event(
            "sandbox",
            "sandbox_preflight_failed",
            cause="namespace_probe_failed",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
        raise SandboxNamespaceError(
            f"bwrap namespace probe failed (exit {proc.returncode}): {err}\n\n"
            f"{SANDBOX_NAMESPACE_HINT}"
        )
    logger.info("tau2 sandbox: bwrap namespace probe ok")

    try:
        srt_proc = subprocess.run(
            _SRT_PROBE, capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        artifact_event(
            "sandbox",
            "sandbox_preflight_failed",
            cause="srt_probe_exception",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise SandboxNamespaceError(
            f"srt end-to-end probe could not run: {exc}\n\n{SANDBOX_NAMESPACE_HINT}"
        ) from exc
    if srt_proc.returncode != 0 or "srt-ok" not in (srt_proc.stdout or ""):
        err = (srt_proc.stderr or srt_proc.stdout or "").strip()
        artifact_event(
            "sandbox",
            "sandbox_preflight_failed",
            cause="srt_probe_failed",
            returncode=srt_proc.returncode,
            stdout=srt_proc.stdout,
            stderr=srt_proc.stderr,
        )
        raise SandboxNamespaceError(
            f"srt end-to-end probe failed (exit {srt_proc.returncode}): {err}\n\n"
            f"{SANDBOX_NAMESPACE_HINT}"
        )
    logger.info("tau2 sandbox: srt end-to-end probe ok")
    artifact_event(
        "sandbox",
        "sandbox_preflight_succeeded",
        bwrap={
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
        srt={
            "returncode": srt_proc.returncode,
            "stdout": srt_proc.stdout,
            "stderr": srt_proc.stderr,
        },
    )

DEFAULT_GREETING = "Hi! How can I help you today?"
AGENT_STOP_TOKEN = "###STOP###"

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the provided policy below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ToolCallSpec:
    id: str
    name: str
    arguments: dict


@dataclass
class AgentTurn:
    token_ids: list[int]
    logprobs: list[float]
    content: str | None
    tool_calls: list[ToolCallSpec] = field(default_factory=list)
    prompt_token_ids: list[int] | None = None


@dataclass
class EpisodeResult:
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    rollout_logprobs: list[float]
    loss_mask: list[int]
    step_spans: list[tuple[int, int]]
    transcript: str
    messages: list[dict]
    termination: str
    reward: float = 0.0
    tau_messages: list[Any] = field(default_factory=list)


GenerateTurn = Callable[[list[dict], list[dict] | None], Awaitable[AgentTurn]]
ExecuteTool = Callable[[ToolCallSpec], str]
UserReply = Callable[[str], Awaitable[tuple[str, bool]]]
ScoreFn = Callable[[list[dict], str], float]


def parse_tool_calls(text: str) -> list[ToolCallSpec]:
    """Parse Qwen-style `<tool_call>{...}</tool_call>` blocks out of a completion."""
    out: list[ToolCallSpec] = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "skipping unparseable tool_call block: %s: %s",
                exc,
                raw[:500],
            )
            artifact_event(
                "rollouts",
                "tool_call_parse_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                raw=raw,
            )
            continue
        name = payload.get("name") or payload.get("function")
        if not name:
            logger.warning("skipping tool_call with no function name: %s", raw[:500])
            artifact_event(
                "rollouts",
                "tool_call_parse_failed",
                error_type="MissingFunctionName",
                error="tool call contained no name or function",
                raw=raw,
            )
            continue
        args = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        out.append(
            ToolCallSpec(id=f"call_{uuid.uuid4().hex[:8]}", name=str(name), arguments=args)
        )
    return out


def system_policy(policy: str) -> str:
    return (
        f"{AGENT_INSTRUCTION}\n\n"
        f"<policy>\n{policy}\n</policy>"
    )


def format_tool_result(tool_call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def format_user(text: str) -> dict:
    return {"role": "user", "content": text}


def format_assistant(content: str | None, tool_calls: list[ToolCallSpec]) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
        # Tau2 protocol: text XOR tool calls. Keep the sampled tokens in the
        # trajectory, but drop content from the chat history so the next turn is valid.
        msg["content"] = ""
    return msg


def format_transcript(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        if msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc)
                calls.append(f"{fn.get('name')}({fn.get('arguments')})")
            lines.append(f"{role}: " + "; ".join(calls))
            continue
        content = msg.get("content") or ""
        if role == "system":
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def env_kwargs_for(task: Task, retrieval: str = "alltools-qwen") -> dict:
    if task.domain == "banking_knowledge":
        return {
            "solo_mode": False,
            "retrieval_variant": retrieval,
            "task": task.tau2_task,
        }
    return {"solo_mode": False}


def configure_embeddings_cache(cache_dir: str | None = None) -> None:
    """Point tau2's EmbeddingsCache at `/workspace` on RunPod so dense retrieval survives restarts."""
    if cache_dir is None:
        cache_dir = (
            "/workspace/.embeddings_cache" if os.path.isdir("/workspace") else None
        )
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from tau2.knowledge import embeddings_cache as ec
    except ImportError:
        return
    ec._global_cache = ec.EmbeddingsCache(cache_dir=cache_dir)
    logger.info("tau2 embeddings cache: %s", cache_dir)


def openai_tool_schemas(env) -> list[dict]:
    try:
        return [t.openai_schema for t in env.get_tools()]
    except Exception as exc:
        logger.exception("tau2 tool schema discovery failed")
        artifact_event(
            "sandbox",
            "tau2_tool_schema_discovery_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []


def gold_material(task: Task) -> str:
    """Sierra-authored answer key: gold KB docs (banking) or canonical actions (retail/airline)."""
    tau = task.tau2_task
    if tau is None:
        return ""
    if task.domain == "banking_knowledge":
        titles = list(getattr(tau, "required_documents", None) or [])
        if not titles:
            return "(no gold documents listed)"
        bodies = _load_kb_docs(titles)
        return "\n\n".join(bodies) if bodies else "(no gold documents listed)"
    criteria = getattr(tau, "evaluation_criteria", None)
    actions = (criteria.actions if criteria is not None else None) or []
    lines = [a.get_func_format() for a in actions]
    return "\n".join(lines) if lines else "(no reference actions)"


def gold_suffix(task: Task) -> str:
    """Teacher-context dump. The trainer prepends HINT_SEPARATOR; do not include it here."""
    if task.tau2_task is None:
        return ""
    if task.domain == "banking_knowledge":
        return f"Gold knowledge for this task:\n{gold_material(task)}"
    return f"Canonical tool trajectory:\n{gold_material(task)}"


def _load_kb_docs(titles: list[str]) -> list[str]:
    try:
        from tau2.domains.banking_knowledge.environment import get_knowledge_base
    except ImportError:
        return [f"(knowledge extra not installed; cannot load {t})" for t in titles]
    kb = get_knowledge_base()
    docs = list(kb.get_all_documents())
    title_to_doc = {doc.title: doc for doc in docs}
    id_to_doc = {doc.id: doc for doc in docs}
    out: list[str] = []
    for ref in titles:
        doc = title_to_doc.get(ref) or id_to_doc.get(ref)
        if doc is None:
            logger.warning("required document not in KB: %s", ref)
            continue
        out.append(f"## {doc.title}\n\n{doc.content}")
    return out


def make_env(task: Task, retrieval: str = "alltools-qwen"):
    from tau2.registry import registry

    if task.domain is None or task.tau2_task is None:
        raise ValueError(f"{task.task_id} is not a tau2 task")
    started = time.monotonic()
    kwargs = env_kwargs_for(task, retrieval)
    artifact_event(
        "sandbox",
        "tau2_environment_build_started",
        task_id=task.task_id,
        domain=task.domain,
        retrieval=retrieval,
        env_kwargs=kwargs,
    )
    try:
        ctor = registry.get_env_constructor(task.domain)
        env = ctor(**kwargs)
        tau = task.tau2_task
        init = getattr(tau, "initial_state", None)
        env.set_state(
            initialization_data=getattr(init, "initialization_data", None)
            if init
            else None,
            initialization_actions=(
                getattr(init, "initialization_actions", None) if init else None
            ),
            message_history=(
                list(getattr(init, "message_history", None) or []) if init else []
            ),
        )
    except Exception as exc:
        logger.exception(
            "tau2 environment build failed for task %s (%s)",
            task.task_id,
            task.domain,
        )
        artifact_event(
            "sandbox",
            "tau2_environment_build_failed",
            task_id=task.task_id,
            domain=task.domain,
            retrieval=retrieval,
            elapsed_seconds=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    artifact_event(
        "sandbox",
        "tau2_environment_build_succeeded",
        task_id=task.task_id,
        domain=task.domain,
        retrieval=retrieval,
        elapsed_seconds=time.monotonic() - started,
        tool_count=len(openai_tool_schemas(env)),
    )
    return env


def make_user(task: Task, llm: str, llm_args: dict | None = None, env=None):
    from tau2.user.user_simulator import UserSimulator

    tools = None
    if env is not None and getattr(env, "user_tools", None) is not None:
        include = getattr(task.tau2_task, "user_tools", None)
        tools = env.get_user_tools(include=include)
    return UserSimulator(
        llm=llm,
        instructions=str(task.tau2_task.user_scenario),
        tools=tools,
        llm_args=llm_args if llm_args is not None else default_user_llm_args(),
    )


def default_user_llm_args() -> dict:
    """Thinking off on OpenRouter so the user-sim cannot eat the output budget."""
    return {
        "temperature": 0.0,
        "extra_body": {"reasoning": {"enabled": False, "effort": "none"}},
    }


def evaluate_episode(task: Task, tau_messages: list, termination: str, retrieval: str) -> float:
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.utils.utils import get_now

    try:
        reason = TerminationReason(termination)
    except ValueError:
        reason = TerminationReason.MAX_STEPS
    now = get_now()
    sim = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=task.tau2_task.id,
        start_time=now,
        end_time=now,
        duration=0.0,
        termination_reason=reason,
        messages=tau_messages,
    )
    info = evaluate_simulation(
        simulation=sim,
        task=task.tau2_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=task.domain,
        env_kwargs={
            k: v for k, v in env_kwargs_for(task, retrieval).items() if k != "solo_mode"
        },
    )
    return float(info.reward)


async def run_episode(
    *,
    generate_turn: GenerateTurn,
    encode: Callable[[str], list[int]],
    tokenize_chat: Callable[[list[dict], list[dict] | None], list[int]],
    execute_tool: ExecuteTool,
    user_reply: UserReply,
    tools: list[dict] | None,
    system: str,
    greeting: str = DEFAULT_GREETING,
    first_user: str | None = None,
    max_steps: int = 30,
    score: ScoreFn | None = None,
) -> EpisodeResult:
    """Flattened multi-turn rollout with a sparse loss mask.

    vLLM-sampled tokens get `loss_mask=1`. Env-injected tool results and user
    simulator tokens get `loss_mask=0` and dummy 0.0 logprobs.
    """
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "assistant", "content": greeting},
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

    if first_user is None:
        first_user, _ = await user_reply(greeting)
    messages.append(format_user(first_user))
    prompt_ids = tokenize_chat(messages, tools)

    for _ in range(max_steps):
        turn = await generate_turn(messages, tools)
        if prompt_ids is None or (not resp_ids and turn.prompt_token_ids):
            prompt_ids = turn.prompt_token_ids or prompt_ids
        append_sampled(turn.token_ids, turn.logprobs)
        assistant = format_assistant(turn.content, turn.tool_calls)
        messages.append(assistant)

        if turn.tool_calls:
            for tc in turn.tool_calls:
                result = execute_tool(tc)
                if inspect.isawaitable(result):
                    result = await result
                tool_msg = format_tool_result(tc.id, tc.name, result)
                append_injected(_tool_text(tool_msg))
                messages.append(tool_msg)
            continue

        text = turn.content or ""
        if AGENT_STOP_TOKEN in text:
            termination = "agent_stop"
            break

        user_text, done = await user_reply(text)
        append_injected(_user_text(user_text))
        messages.append(format_user(user_text))
        if done:
            termination = "user_stop"
            break
    else:
        termination = "max_steps"

    if prompt_ids is None:
        prompt_ids = tokenize_chat(messages[:3], tools)

    transcript = format_transcript(messages)
    reward = score(messages, termination) if score is not None else 0.0
    return EpisodeResult(
        prompt_token_ids=prompt_ids,
        response_token_ids=resp_ids,
        rollout_logprobs=logprobs,
        loss_mask=loss_mask,
        step_spans=step_spans,
        transcript=transcript,
        messages=messages,
        termination=termination,
        reward=reward,
    )


def _tool_text(msg: dict) -> str:
    return f"\n[tool {msg.get('name', '')}]: {msg.get('content', '')}\n"


def _user_text(text: str) -> str:
    return f"\n[user]: {text}\n"


def _to_tau_messages(messages: list[dict]):
    """Best-effort conversion of OpenAI-style dicts to tau2 Message objects."""
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    out = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "assistant":
            tcs = None
            if msg.get("tool_calls"):
                tcs = [
                    ToolCall(
                        id=tc.get("id", ""),
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"],
                    )
                    for tc in msg["tool_calls"]
                ]
            content = msg.get("content") or None
            if tcs:
                content = None
            out.append(AssistantMessage(role="assistant", content=content, tool_calls=tcs))
            continue
        if role == "user":
            out.append(UserMessage(role="user", content=msg.get("content") or ""))
            continue
        if role == "tool":
            out.append(
                ToolMessage(
                    id=msg.get("tool_call_id", ""),
                    content=msg.get("content") or "",
                    requestor="assistant",
                    role="tool",
                )
            )
    return out


async def _retry_transient(fn, *, tries: int = 6, base: float = 1.5, cap: float = 30.0):
    """Retry a thread-executed LLM op through transient provider errors (429 /
    upstream overload / timeout). Re-raises the last error once tries are spent."""
    last = None
    for i in range(tries):
        try:
            return await asyncio.to_thread(fn)
        except Exception as exc:  # noqa: BLE001
            last = exc
            s = f"{type(exc).__name__} {exc}".lower()
            transient = any(
                k in s
                for k in (
                    "ratelimit",
                    "429",
                    "rate-limited",
                    "overload",
                    "engine_overloaded",
                    "timeout",
                    "timed out",
                    "temporarily",
                    "503",
                    "502",
                    "unavailable",
                )
            )
            if i + 1 < tries and transient:
                await asyncio.sleep(min(base * (2**i), cap))
                continue
            raise
    raise last


async def run_tau2_episode(
    task: Task,
    *,
    generate_turn: GenerateTurn,
    encode: Callable[[str], list[int]],
    tokenize_chat: Callable[[list[dict], list[dict] | None], list[int]],
    user_llm: str,
    retrieval: str = "alltools-qwen",
    max_steps: int = 30,
    user_llm_args: dict | None = None,
) -> EpisodeResult:
    """Full tau2 wiring: env + user sim + evaluator around `run_episode`."""
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.user.user_simulator import UserSimulator

    env = make_env(task, retrieval=retrieval)
    user = make_user(task, llm=user_llm, llm_args=user_llm_args, env=env)
    user_state = user.get_init_state()
    tools = openai_tool_schemas(env)
    policy = system_policy(env.get_policy())

    async def _user_reply(agent_text: str) -> tuple[str, bool]:
        incoming = AssistantMessage(role="assistant", content=agent_text, cost=0.0)

        def _call():
            return user.generate_next_message(incoming, user_state)

        started = time.monotonic()
        try:
            user_msg, _ = await _retry_transient(_call)
        except Exception as exc:
            logger.exception(
                "tau2 user simulator failed for task %s (%s)",
                task.task_id,
                task.domain,
            )
            artifact_event(
                "api_failures",
                "tau2_user_simulator_failed",
                provider="openrouter",
                operation="tau2_user_simulator",
                task_id=task.task_id,
                domain=task.domain,
                model=user_llm,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
                error=str(exc),
                incoming_agent_text=agent_text,
            )
            raise
        done = UserSimulator.is_stop(user_msg)
        return user_msg.content or "", done

    first_user, _ = await _user_reply(DEFAULT_GREETING)
    # The greeting already entered user_state; subsequent replies must not
    # re-feed it. `_user_reply` appends each new assistant text.

    def _execute(tc: ToolCallSpec) -> str:
        started = time.monotonic()
        artifact_event(
            "sandbox",
            "tau2_tool_call_started",
            task_id=task.task_id,
            domain=task.domain,
            tool_call_id=tc.id,
            tool=tc.name,
            arguments=tc.arguments,
        )
        try:
            result = env.get_response(
                ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
            )
        except Exception as exc:
            logger.exception(
                "tau2 tool call failed for task %s (%s): %s",
                task.task_id,
                task.domain,
                tc.name,
            )
            artifact_event(
                "sandbox",
                "tau2_tool_call_failed",
                task_id=task.task_id,
                domain=task.domain,
                tool_call_id=tc.id,
                tool=tc.name,
                arguments=tc.arguments,
                elapsed_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        content = result.content or ""
        artifact_event(
            "sandbox",
            "tau2_tool_call_succeeded",
            task_id=task.task_id,
            domain=task.domain,
            tool_call_id=tc.id,
            tool=tc.name,
            arguments=tc.arguments,
            elapsed_seconds=time.monotonic() - started,
            result=content,
        )
        return content

    async def _generate(messages: list[dict], tool_schemas: list[dict] | None) -> AgentTurn:
        # Discoverable banking tools can appear mid-episode.
        current = openai_tool_schemas(env) or tool_schemas
        return await generate_turn(messages, current)

    try:
        episode = await run_episode(
            generate_turn=_generate,
            encode=encode,
            tokenize_chat=tokenize_chat,
            execute_tool=_execute,
            user_reply=_user_reply,
            tools=tools,
            system=policy,
            greeting=DEFAULT_GREETING,
            first_user=first_user,
            max_steps=max_steps,
        )
    except Exception as exc:
        artifact_event(
            "rollouts",
            "tau2_episode_failed",
            task_id=task.task_id,
            domain=task.domain,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    episode.tau_messages = _to_tau_messages(episode.messages)
    evaluation_started = time.monotonic()
    try:
        # evaluate_simulation is sync and can reconstruct an env (banking
        # sandbox). Running it on the event loop freezes get_batch / timeouts.
        episode.reward = await asyncio.to_thread(
            evaluate_episode,
            task,
            episode.tau_messages,
            episode.termination,
            retrieval,
        )
    except Exception as exc:
        logger.exception("tau2 evaluate_simulation failed for %s", task.task_id)
        artifact_event(
            "sandbox",
            "tau2_evaluation_failed",
            task_id=task.task_id,
            domain=task.domain,
            termination=episode.termination,
            elapsed_seconds=time.monotonic() - evaluation_started,
            error_type=type(exc).__name__,
            error=str(exc),
            messages=episode.messages,
        )
        episode.reward = 0.0
    else:
        artifact_event(
            "sandbox",
            "tau2_evaluation_succeeded",
            task_id=task.task_id,
            domain=task.domain,
            termination=episode.termination,
            elapsed_seconds=time.monotonic() - evaluation_started,
            reward=episode.reward,
        )
    return episode
