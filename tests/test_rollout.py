"""Tests for the rollout module's pure logic. No GPU, no vLLM install needed.

Only `_extract_sampled_logprobs` is testable off-GPU, but it is the piece worth testing:
these log-probs become the denominator of every IS ratio, and a misalignment here is
silent -- training runs, the loss looks fine, and the off-policy correction is wrong.
"""

import asyncio

import pytest

from data.dataset import Task
from train.backends.vllm import VLLMRolloutEngine

_extract_sampled_logprobs = VLLMRolloutEngine._extract_sampled_logprobs


class FakeLogprob:
    """Stands in for vllm.sequence.Logprob."""

    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class FakeCompletion:
    """Stands in for vLLM's CompletionOutput: token_ids + one logprob dict per position."""

    def __init__(self, token_ids, logprobs) -> None:
        self.token_ids = token_ids
        self.logprobs = logprobs


def test_extracts_sampled_token_logprobs_in_order():
    completion = FakeCompletion(
        token_ids=[10, 20, 30],
        logprobs=[
            {10: FakeLogprob(-0.1)},
            {20: FakeLogprob(-0.2)},
            {30: FakeLogprob(-0.3)},
        ],
    )
    assert _extract_sampled_logprobs(completion) == pytest.approx([-0.1, -0.2, -0.3])


def test_picks_sampled_token_not_the_top_token():
    """With logprobs>0 the dict holds alternatives too. We must select by sampled token id,
    not by rank -- taking the most likely token would bias every ratio toward 1.
    """
    completion = FakeCompletion(
        token_ids=[20],
        logprobs=[{
            10: FakeLogprob(-0.05),   # more likely, but NOT what was sampled
            20: FakeLogprob(-2.50),   # the sampled token
        }],
    )
    assert _extract_sampled_logprobs(completion) == pytest.approx([-2.50])


def test_missing_logprobs_raises():
    """Silently returning [] here would misalign the whole trajectory."""
    completion = FakeCompletion(token_ids=[1, 2], logprobs=None)
    with pytest.raises(ValueError, match="no logprobs"):
        _extract_sampled_logprobs(completion)


def test_missing_sampled_token_raises():
    completion = FakeCompletion(
        token_ids=[10, 20],
        logprobs=[{10: FakeLogprob(-0.1)}, {99: FakeLogprob(-0.2)}],  # 20 absent
    )
    with pytest.raises(ValueError, match="missing from its logprob"):
        _extract_sampled_logprobs(completion)


def test_output_length_matches_token_count():
    n = 7
    completion = FakeCompletion(
        token_ids=list(range(n)),
        logprobs=[{i: FakeLogprob(-0.1 * i)} for i in range(n)],
    )
    result = _extract_sampled_logprobs(completion)
    assert len(result) == n


def test_empty_completion_gives_empty_logprobs():
    assert _extract_sampled_logprobs(FakeCompletion([], [])) == []


def test_extracted_logprobs_feed_a_valid_trajectory():
    """The output must satisfy Trajectory's alignment invariant."""
    from train.store import Trajectory

    completion = FakeCompletion(
        token_ids=[5, 6, 7],
        logprobs=[{5: FakeLogprob(-1.0)}, {6: FakeLogprob(-2.0)}, {7: FakeLogprob(-3.0)}],
    )
    logprobs = _extract_sampled_logprobs(completion)
    trajectory = Trajectory(
        task_id="1", prompt_token_ids=[1, 2],
        response_token_ids=list(completion.token_ids),
        rollout_logprobs=logprobs, policy_version=0,
    )
    assert len(trajectory.rollout_logprobs) == len(trajectory.response_token_ids)


def test_sync_weights_posts_receive_before_sending():
    """The deadlock guard for the native weight-transfer path.

    Receivers post their broadcasts inside `update_weights`; the trainer's
    `trainer_send_weights` blocks until they do. So the receive RPC must be IN FLIGHT
    before the send starts. Awaiting the receive to completion first hangs both sides --
    and it hangs on the GPU box, never here, so the ordering has to be pinned by a test.
    """
    import asyncio

    import torch

    from run import sync_weights
    from train.backends.backend import WeightBucket

    events: list[str] = []

    class FakeEngine:
        async def pause_for_update(self):
            events.append("pause")

        async def receive_weight_bucket(self, names, dtype_names, shapes):
            events.append("receive_start")
            # Stands in for the receiver blocking until the trainer sends.
            while "send" not in events:
                await asyncio.sleep(0)
            events.append("receive_done")

        async def finish_update(self, new_version):
            events.append("finish")

    class FakeTrainer:
        # Mirrors the real nesting: run.py reads
        # trainer.config.generator.engine.weight_sync_timeout_s.
        config = type(
            "C", (), {"generator": type(
                "G", (), {"engine": type("E", (), {"weight_sync_timeout_s": 5.0})()},
            )()},
        )()
        staleness_manager = None

        def weight_buckets(self):
            yield WeightBucket(
                names=["w1"],
                dtype_names=["bfloat16"],
                shapes=[[2, 2]],
                tensors=[torch.zeros(2, 2, dtype=torch.bfloat16)],
            )

        def send_weight_bucket(self, bucket):
            events.append("send")

    class FakeStore:
        async def set_policy_version(self, version):
            events.append(f"version:{version}")

        async def prune_stale(self, staleness_manager=None):
            events.append("prune")

    async def run():
        await sync_weights(FakeTrainer(), FakeEngine(), FakeStore(), 7)

    # If sync_weights awaited the receive before sending, this never completes.
    asyncio.run(asyncio.wait_for(run(), timeout=5.0))

    assert events == [
        "pause", "receive_start", "send", "receive_done", "finish", "version:7", "prune"
    ], events


def test_weight_sync_requires_initialized_group():
    import asyncio

    from train.config import Config
    from train.backends.vllm import VLLMRolloutEngine as RolloutEngine

    engine = RolloutEngine(Config())
    with pytest.raises(RuntimeError, match="weight sync not initialized"):
        asyncio.run(engine.pause_for_update())


def test_module_imports_without_vllm_installed():
    """vLLM is imported lazily inside start(), so the package stays importable on a dev
    machine with no GPU -- which is what lets every other test run locally."""
    import train.backends.vllm as rollout
    assert hasattr(rollout, "VLLMRolloutEngine")
    # The native weight-transfer API is reached through the engine, so there is no worker
    # extension and no private load_weights call to keep importable.
    for method in ("init_weight_update_group", "pause_for_update",
                   "receive_weight_bucket", "finish_update", "generate_tir"):
        assert hasattr(rollout.VLLMRolloutEngine, method), method


def test_isolated_from_torchrun_strips_and_restores(monkeypatch):
    """vLLM TP workers inherit os.environ at spawn. torchrun's WORLD_SIZE matches TP=4
    on the 8xH100 split, so leaking it makes workers join the trainer TCPStore."""
    import os

    from train.backends.vllm import isolated_from_torchrun

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("TORCHELASTIC_USE_AGENT_STORE", "1")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc")
    monkeypatch.setenv("UNRELATED", "keep-me")

    with isolated_from_torchrun():
        assert "RANK" not in os.environ
        assert "WORLD_SIZE" not in os.environ
        assert "MASTER_PORT" not in os.environ
        assert "TORCHELASTIC_USE_AGENT_STORE" not in os.environ
        assert "TORCHELASTIC_RUN_ID" not in os.environ
        assert os.environ["UNRELATED"] == "keep-me"
        assert os.environ.get("VLLM_HOST_IP") == "127.0.0.1"

    assert os.environ["RANK"] == "0"
    assert os.environ["WORLD_SIZE"] == "4"
    assert os.environ["TORCHELASTIC_USE_AGENT_STORE"] == "1"
    assert os.environ["UNRELATED"] == "keep-me"
    assert "VLLM_HOST_IP" not in os.environ


def test_start_refuses_an_already_initialized_process_group(monkeypatch):
    """The Baseten 4+4 hang: engine.start() after dist.init_process_group."""
    import torch.distributed as dist
    from train.config import Config
    from train.backends.vllm import VLLMRolloutEngine

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    engine = VLLMRolloutEngine(Config())
    with pytest.raises(RuntimeError, match="cannot be spawned after"):
        engine.start()


def test_weight_sync_port_falls_back_when_preferred_is_taken():
    """A leftover process holding 51216 dropped the answer_bearing arm."""
    import socket

    from run import _pick_weight_sync_port

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    taken = int(holder.getsockname()[1])
    try:
        chosen = _pick_weight_sync_port("127.0.0.1", taken)
        assert chosen != taken
        assert chosen > 0
    finally:
        holder.close()


# ---- error-conditioned hint generation (no GPU, no network) ----

def _task():
    return Task(task_id="1", query="Assess the funding base.",
                sections=[{"id": "risk-awareness", "criteria": []}])


def test_hint_is_returned_for_the_trajectory(monkeypatch):
    from conftest import make_config
    from data.hint import generate_hint

    async def hint(**kwargs):
        return "LLM HINT"

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="answer_free"), _task(), "draft")
    )
    assert hints.free == "LLM HINT"
    assert hints.ok("answer_free")


def test_hint_failure_returns_empty_not_raises(monkeypatch):
    """A failed hint must surface as empty so the caller can drop the rollout, not as an
    exception that kills the worker."""
    from conftest import make_config
    from data.hint import generate_hint

    async def boom(**kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr("data.hint.build_error_hint", boom)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="answer_free"), _task(), "draft")
    )
    assert hints.free == ""
    assert not hints.ok("answer_free")
    assert hints.cause == "other"
    assert hints.detail == "RuntimeError: api down"


@pytest.mark.parametrize(
    ("message", "expected_cause"),
    [
        ("openrouter rejected the request (402): insufficient credits", "openrouter_credit"),
        ("openrouter auth failed (401): invalid key", "openrouter_auth"),
        ("429 Client Error: Too Many Requests", "openrouter_rate_limit"),
        ("provider unavailable", "openrouter_error"),
    ],
)
def test_hint_failure_logs_actionable_openrouter_cause(
    monkeypatch, message, expected_cause
):
    from conftest import make_config
    from data.hint import generate_hint

    async def fail_chat_completion(*args, **kwargs):
        raise RuntimeError(message)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("data.hint.chat_completion", fail_chat_completion)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="answer_free"), _task(), "draft")
    )

    assert hints.cause == expected_cause
    assert message in hints.detail


def test_hint_receives_the_configured_prompt_variant(monkeypatch):
    """The ablation arm must reach the hint generator."""
    from conftest import make_config
    from data.hint import generate_hint

    seen = {}

    async def hint(**kwargs):
        seen.update(kwargs)
        return "LLM HINT"

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    asyncio.run(
        generate_hint(make_config(error_hint_prompt="answer_bearing"), _task(), "the draft text")
    )
    assert seen["prompt_variant"] == "answer_bearing"
    assert seen["response_text"] == "the draft text"
    assert seen["query"] == "Assess the funding base."
    assert seen["model"] == "z-ai/glm-5.3-flash"
    assert seen["max_retries"] == 5


def test_hint_concurrency_is_bounded(monkeypatch):
    """One LLM call per trajectory, and rollout runs continuously -- without the semaphore
    this fans out to every in-flight generation at once."""
    from conftest import make_config
    from data.hint import generate_hint
    import data.hint as hint_mod

    hint_mod._HINT_SEM = None
    in_flight = 0
    peak = 0

    async def hint(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "LLM HINT"

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    config = make_config(error_hint_concurrency=2, error_hint_prompt="answer_free")

    async def main():
        await asyncio.gather(*(generate_hint(config, _task(), "draft") for _ in range(6)))

    asyncio.run(main())
    assert peak <= 2, f"expected at most 2 concurrent hint calls, saw {peak}"


def test_mixture_generates_both_hints(monkeypatch):
    from conftest import make_config
    from data.hint import generate_hint

    async def hint(**kwargs):
        return f"HINT-{kwargs['prompt_variant']}"

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="mixture"), _task(), "draft")
    )
    assert hints.free == "HINT-answer_free"
    assert hints.bearing == "HINT-answer_bearing"
    assert hints.ok("mixture")


def test_mixture_not_ok_if_either_hint_empty(monkeypatch):
    from conftest import make_config
    from data.hint import generate_hint

    async def hint(**kwargs):
        if kwargs["prompt_variant"] == "answer_bearing":
            return ""
        return "FREE"

    monkeypatch.setattr("data.hint.build_error_hint", hint)
    hints = asyncio.run(
        generate_hint(make_config(error_hint_prompt="mixture"), _task(), "draft")
    )
    assert hints.free == "FREE"
    assert hints.bearing == ""
    assert not hints.ok("mixture")


