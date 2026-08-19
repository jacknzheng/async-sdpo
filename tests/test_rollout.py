"""Tests for the rollout module's pure logic. No GPU, no vLLM install needed.

Only `_extract_sampled_logprobs` is testable off-GPU, but it is the piece worth testing:
these log-probs become the denominator of every IS ratio, and a misalignment here is
silent -- training runs, the loss looks fine, and the off-policy correction is wrong.
"""

import asyncio

import pytest

from data.dataset import Task
from train.backends.vllm import _extract_sampled_logprobs
from train.store import Trajectory


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

    async def run():
        await sync_weights(FakeTrainer(), FakeEngine(), FakeStore(), 7)

    # If sync_weights awaited the receive before sending, this never completes.
    asyncio.run(asyncio.wait_for(run(), timeout=5.0))

    assert events == [
        "pause", "receive_start", "send", "receive_done", "finish", "version:7"
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
                   "receive_weight_bucket", "finish_update"):
        assert hasattr(rollout.VLLMRolloutEngine, method), method


# ---- error-conditioned hint generation (no GPU, no network) ----

def _engine(**overrides):
    """A RolloutEngine with __init__ bypassed -- only the hint state is needed."""
    from conftest import make_config
    from train.backends.vllm import VLLMRolloutEngine as RolloutEngine

    engine = RolloutEngine.__new__(RolloutEngine)
    engine.config = make_config(**overrides)
    engine._hint_semaphore = asyncio.Semaphore(engine.config.generator.hint.concurrency)
    engine.hintless_dropped = 0
    return engine


def _task():
    return Task(task_id="1", query="Assess the funding base.",
                sections=[{"id": "risk-awareness", "criteria": []}])


def test_hint_is_returned_for_the_trajectory(monkeypatch):
    async def hint(**kwargs):
        return "LLM HINT"

    monkeypatch.setattr("train.backends.vllm.build_error_hint", hint)
    engine = _engine()
    assert asyncio.run(engine._error_hint(_task(), "draft")) == "LLM HINT"


def test_hint_failure_returns_empty_not_raises(monkeypatch):
    """A failed hint must surface as "" so the caller can drop the rollout, not as an
    exception that kills the worker."""
    async def boom(**kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr("train.backends.vllm.build_error_hint", boom)
    engine = _engine()
    assert asyncio.run(engine._error_hint(_task(), "draft")) == ""


def test_hint_receives_the_configured_prompt_variant(monkeypatch):
    """The ablation arm must reach the hint generator."""
    seen = {}

    async def hint(**kwargs):
        seen.update(kwargs)
        return "LLM HINT"

    monkeypatch.setattr("train.backends.vllm.build_error_hint", hint)
    engine = _engine(error_hint_prompt="answer_bearing")
    asyncio.run(engine._error_hint(_task(), "the draft text"))
    assert seen["prompt_variant"] == "answer_bearing"
    assert seen["response_text"] == "the draft text"
    assert seen["query"] == "Assess the funding base."


def test_hint_concurrency_is_bounded(monkeypatch):
    """One LLM call per trajectory, and rollout runs continuously -- without the semaphore
    this fans out to every in-flight generation at once."""
    in_flight = 0
    peak = 0

    async def hint(**kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "LLM HINT"

    monkeypatch.setattr("train.backends.vllm.build_error_hint", hint)
    engine = _engine(error_hint_concurrency=2)

    async def main():
        await asyncio.gather(*(engine._error_hint(_task(), "draft") for _ in range(6)))

    asyncio.run(main())
    assert peak <= 2, f"expected at most 2 concurrent hint calls, saw {peak}"


