"""Trajectory store -- the buffer that decouples the rollout clock from the trainer clock.

Generating an answer is slow; a trainer step is fast. If the trainer waits for every
rollout, the training GPUs idle through the slowest tail of every batch. So rollout workers
stream finished trajectories in here, and the trainer pulls batches on its own schedule.

The cost of that decoupling is staleness. A trajectory is produced by policy version V, but
by the time the trainer samples it the policy has moved to V+K. We call K the staleness. It
is not a knob we set -- it falls out of rollout latency versus trainer throughput -- but we
do get to choose how much of it we tolerate. Per the blog (and the user's instruction),
that bound is K=3: accuracy is flat from K=0 through K=3 and degrades after, and K=3 buys a
~2x wall-clock speedup over synchronous training.

Anything staler than the bound is EVICTED rather than trained on. The two clips in loss.py
bound the variance of off-policy updates; they do not make arbitrarily stale data safe.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """One finished rollout, tagged with the policy that produced it.

    `rollout_logprobs` are the per-token log-probs recorded by the rollout engine AT
    GENERATION TIME, under policy `policy_version`. They are the denominator of the IS
    ratio in loss.py and cannot be recomputed later -- recomputing under current weights
    would make every ratio 1.0 and silently erase the off-policy correction.
    """

    task_id: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    rollout_logprobs: list[float]
    policy_version: int
    # The hint the teacher forward pass conditions on, generated from THIS rollout's draft
    # (data/hint.py) before the trajectory was stored. Never empty: a trajectory whose hint
    # could not be generated is dropped at the rollout, because an unhinted teacher is
    # identical to the student and would contribute ~zero gradient.
    #
    # Carried on the trajectory rather than recomputed at batch time so the teacher
    # conditions on the draft it was actually derived from.
    hint: str = ""
    # Judge score, attached opportunistically for diagnostics. Never used in the loss, and
    # never gates whether a trajectory is trained on: the blog's headline finding is that
    # G=1 with failures RETAINED is the most efficient configuration.
    judge_score: float | None = None

    def __post_init__(self) -> None:
        if len(self.response_token_ids) != len(self.rollout_logprobs):
            raise ValueError(
                f"task {self.task_id}: {len(self.response_token_ids)} response tokens but "
                f"{len(self.rollout_logprobs)} rollout logprobs -- these must align 1:1"
            )

    def staleness(self, current_version: int) -> int:
        return current_version - self.policy_version


@dataclass
class StoreStats:
    """Counters for diagnosing whether the rollout/trainer split is balanced."""

    added: int = 0
    sampled: int = 0
    evicted_stale: int = 0
    evicted_capacity: int = 0
    staleness_histogram: Counter[int] = field(default_factory=Counter)

    def as_metrics(self) -> dict[str, float]:
        total = sum(self.staleness_histogram.values())
        mean_staleness = (
            sum(k * v for k, v in self.staleness_histogram.items()) / total if total else 0.0
        )
        return {
            "store_added": float(self.added),
            "store_sampled": float(self.sampled),
            "store_evicted_stale": float(self.evicted_stale),
            "store_evicted_capacity": float(self.evicted_capacity),
            "store_mean_staleness": mean_staleness,
            "store_max_staleness_seen": float(max(self.staleness_histogram, default=0)),
        }


class TrajectoryStore:
    """Async-safe FIFO store with staleness-bounded sampling.

    Rollout workers call `add()`; the trainer calls `sample()`. Both take the same lock, so
    the store is safe for many concurrent producers and one consumer.
    """

    def __init__(self, capacity: int = 512, max_staleness: int = 3) -> None:
        self.capacity = capacity
        self.max_staleness = max_staleness
        self.policy_version = 0
        self._buffer: deque[Trajectory] = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self.stats = StoreStats()
        # Bumped on every weight sync. Waiters snapshot it so they can tell a version
        # change apart from a spurious wakeup -- see `wait_for_batch`.
        self._version_generation = 0

    async def add(self, trajectory: Trajectory) -> None:
        """Add a finished trajectory. Oldest-first eviction when at capacity, so a stalled
        trainer degrades into dropping old data rather than OOMing the box."""
        async with self._not_empty:
            while len(self._buffer) >= self.capacity:
                self._buffer.popleft()
                self.stats.evicted_capacity += 1
            self._buffer.append(trajectory)
            self.stats.added += 1
            self._not_empty.notify_all()

    async def sample(self, batch_size: int, drop_stale: bool = True) -> list[Trajectory]:
        """Pull up to `batch_size` trajectories, evicting anything past the staleness bound.

        Returns fewer than `batch_size` (possibly zero) if the store is short -- the caller
        decides whether to wait. Use `wait_for_batch` to block until a full batch exists.
        """
        async with self._lock:
            batch: list[Trajectory] = []
            while self._buffer and len(batch) < batch_size:
                trajectory = self._buffer.popleft()
                staleness = trajectory.staleness(self.policy_version)
                if drop_stale and staleness > self.max_staleness:
                    self.stats.evicted_stale += 1
                    continue
                self.stats.staleness_histogram[staleness] += 1
                batch.append(trajectory)
            self.stats.sampled += len(batch)
            return batch

    async def wait_for_batch(
        self, batch_size: int, timeout: float | None = None
    ) -> list[Trajectory]:
        """Block until a full batch of non-stale trajectories is available, or timeout.

        On timeout, returns whatever is available (possibly a partial or empty batch) so
        the trainer can decide between stepping on a short batch and waiting longer.

        A weight sync can make a pending request unsatisfiable by aging every buffered
        trajectory past the bound. Waking the waiter is not sufficient on its own -- the
        predicate would still be false and it would go straight back to sleep until the
        timeout. So the predicate also returns True when the store has been drained of
        fresh data since the wait began, letting the trainer re-plan immediately.
        """
        generation = self._version_generation

        def ready() -> bool:
            if self._fresh_count() >= batch_size:
                return True
            # The policy version moved while we were waiting; whatever is left is stale.
            return self._version_generation != generation

        try:
            async with self._not_empty:
                await asyncio.wait_for(
                    self._not_empty.wait_for(ready),
                    timeout=timeout,
                )
        except asyncio.TimeoutError:
            pass
        return await self.sample(batch_size)

    def _fresh_count(self) -> int:
        """Count of buffered trajectories within the staleness bound. Caller holds the lock."""
        return sum(
            1
            for t in self._buffer
            if t.staleness(self.policy_version) <= self.max_staleness
        )

    async def set_policy_version(self, version: int) -> None:
        """Bump the policy version after a weight sync. Existing trajectories get staler."""
        async with self._not_empty:
            self.policy_version = version
            self._version_generation += 1
            # Waiters may now be unsatisfiable (everything went stale) -- wake them so they
            # re-evaluate rather than hanging until timeout.
            self._not_empty.notify_all()

    async def prune_stale(self) -> int:
        """Drop everything past the staleness bound. Returns how many were removed."""
        async with self._lock:
            keep = [
                t for t in self._buffer if t.staleness(self.policy_version) <= self.max_staleness
            ]
            removed = len(self._buffer) - len(keep)
            self._buffer = deque(keep)
            self.stats.evicted_stale += removed
            return removed

    def __len__(self) -> int:
        return len(self._buffer)

    def metrics(self) -> dict[str, float]:
        """Store metrics for the trainer log.

        Watch `store_evicted_stale` and `store_mean_staleness`: a large stale-eviction
        fraction means rollout is too slow relative to the trainer, and the fix is to
        rebalance the GPU split (more rollout workers) rather than to raise the bound.
        """
        metrics = self.stats.as_metrics()
        metrics["store_size"] = float(len(self._buffer))
        metrics["store_policy_version"] = float(self.policy_version)
        return metrics
