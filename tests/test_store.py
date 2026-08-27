"""Tests for the trajectory store. CPU-only, no GPU or network.

The load-bearing behavior is staleness eviction at K=3: trajectories produced more than
three trainer steps ago must never reach the loss, because the clips bound off-policy
variance but don't make arbitrarily stale data safe.
"""

import asyncio

import pytest

from train.store import Trajectory, TrajectoryStore


def _traj(task_id: str = "1", version: int = 0, n_tokens: int = 3) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_token_ids=[1, 2],
        response_token_ids=list(range(n_tokens)),
        rollout_logprobs=[-0.5] * n_tokens,
        policy_version=version,
    )


def test_rollout_logprobs_must_align_with_tokens():
    """Misaligned logprobs would silently corrupt every IS ratio in the batch."""
    with pytest.raises(ValueError, match="must align"):
        Trajectory(
            task_id="1", prompt_token_ids=[1],
            response_token_ids=[1, 2, 3], rollout_logprobs=[-0.5],
            policy_version=0,
        )


def test_add_and_sample_roundtrip():
    async def run():
        store = TrajectoryStore(capacity=10, max_staleness=3)
        for i in range(4):
            await store.add(_traj(task_id=str(i)))
        batch = await store.get_batch(4)
        return batch

    batch = asyncio.run(run())
    assert len(batch) == 4
    assert [t.task_id for t in batch] == ["0", "1", "2", "3"]  # FIFO


def test_get_batch_waits_until_filled():
    async def run():
        store = TrajectoryStore()

        async def producer():
            for i in range(4):
                await asyncio.sleep(0.01)
                await store.add(_traj(task_id=str(i)))

        task = asyncio.create_task(producer())
        batch = await store.get_batch(4)
        await task
        return batch

    assert len(asyncio.run(run())) == 4


def test_staleness_boundary_is_inclusive_at_k3():
    """K=3 must be trained on; K=4 must be evicted. Off-by-one here silently changes the
    effective staleness bound the whole run operates at.
    """
    async def run():
        store = TrajectoryStore(max_staleness=3)
        for version in (1, 2, 3, 4, 5):   # staleness 4,3,2,1,0 at current version 5
            await store.add(_traj(task_id=str(version), version=version))
        await store.set_policy_version(5)
        return await store.get_batch(4), store

    batch, store = asyncio.run(run())
    stalenesses = sorted(t.staleness(5) for t in batch)
    assert stalenesses == [0, 1, 2, 3], "K=3 must be kept, K=4 must be dropped"
    assert store.stats.evicted_stale == 1


def test_all_stale_is_not_ready():
    async def run():
        store = TrajectoryStore(max_staleness=3)
        for _ in range(5):
            await store.add(_traj(version=0))
        await store.set_policy_version(100)
        return store.ready(1)

    assert asyncio.run(run()) is False


def test_drop_stale_false_keeps_everything():
    """Escape hatch for a K=0 synchronous baseline run."""
    async def run():
        store = TrajectoryStore(max_staleness=3)
        await store.add(_traj(version=0))
        await store.set_policy_version(50)
        return await store.get_batch(1, drop_stale=False)

    assert len(asyncio.run(run())) == 1


def test_capacity_evicts_oldest_first():
    """A stalled trainer must degrade into dropping old data, not OOM the box."""
    async def run():
        store = TrajectoryStore(capacity=3)
        for i in range(5):
            await store.add(_traj(task_id=str(i)))
        return await store.get_batch(3), store

    batch, store = asyncio.run(run())
    assert [t.task_id for t in batch] == ["2", "3", "4"]
    assert store.stats.evicted_capacity == 2
    assert len(store) == 0


def test_prune_stale_removes_and_counts():
    async def run():
        store = TrajectoryStore(max_staleness=3)
        await store.add(_traj(task_id="fresh", version=10))
        await store.add(_traj(task_id="stale", version=0))
        await store.set_policy_version(10)
        removed = await store.prune_stale()
        return removed, len(store)

    removed, size = asyncio.run(run())
    assert removed == 1 and size == 1


def test_version_bump_wakes_waiters():
    """If a version bump makes a pending batch unsatisfiable, waiters must re-evaluate
    rather than hang until timeout."""
    async def run():
        store = TrajectoryStore(max_staleness=3)
        # Only 2 fresh trajectories, so a request for 4 genuinely blocks.
        for _ in range(2):
            await store.add(_traj(version=0))

        async def refill():
            await asyncio.sleep(0.01)
            await store.set_policy_version(100)  # everything becomes stale
            for i in range(4):
                await store.add(_traj(task_id=str(i), version=100))

        task = asyncio.create_task(refill())
        batch = await store.get_batch(4)
        await task
        return batch, store

    batch, store = asyncio.run(asyncio.wait_for(run(), timeout=2.0))
    assert [t.policy_version for t in batch] == [100, 100, 100, 100]
    assert store.stats.evicted_stale == 2


def test_concurrent_producers_do_not_lose_trajectories():
    async def run():
        store = TrajectoryStore(capacity=1000)
        await asyncio.gather(*[
            store.add(_traj(task_id=f"{w}-{i}")) for w in range(5) for i in range(20)
        ])
        return store

    store = asyncio.run(run())
    assert len(store) == 100 and store.stats.added == 100


def test_metrics_expose_staleness_distribution():
    async def run():
        store = TrajectoryStore(max_staleness=3)
        for version in (5, 4, 3):  # staleness 0, 1, 2
            await store.add(_traj(version=version))
        await store.set_policy_version(5)
        await store.get_batch(3)
        return store.metrics()

    metrics = run and asyncio.run(run())
    assert metrics["store_mean_staleness"] == pytest.approx(1.0)
    assert metrics["store_max_staleness_seen"] == 2.0
    assert metrics["store_policy_version"] == 5.0


def test_hint_drop_percent_uses_all_attempts_and_tracks_cause():
    async def run():
        store = TrajectoryStore(capacity=20)
        for i in range(10):
            await store.add(_traj(task_id=str(i)))
        for _ in range(16):
            store.stats.count_hint_drop("openrouter_error")
        store.stats.count_hint_drop("openrouter_credit")
        store.stats.count_hint_drop("openrouter_auth")
        store.stats.count_hint_drop("openrouter_rate_limit")
        store.stats.count_hint_drop("openrouter_length")
        return store.metrics()

    metrics = asyncio.run(run())
    assert metrics["store_hint_attempted"] == 30.0
    assert metrics["store_hint_ok"] == 10.0
    assert metrics["store_hint_dropped"] == 20.0
    assert metrics["store_hint_dropped_percent"] == pytest.approx(2 / 3)
    assert metrics["hint_drop_openrouter_error"] == 16.0
    assert metrics["hint_drop_openrouter_credit"] == 1.0
    assert metrics["hint_drop_openrouter_auth"] == 1.0
    assert metrics["hint_drop_openrouter_rate_limit"] == 1.0
    assert metrics["hint_drop_openrouter_length"] == 1.0
