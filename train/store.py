from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
from train.models import Trajectory

@dataclass
class StoreStats:

    added: int = 0
    sampled: int = 0
    evicted_stale: int = 0
    evicted_capacity: int = 0
    hint_dropped:int = 0
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
            "store_hint_dropped_percent": float(
                self.hint_dropped / self.added if self.added else 0.0
            ),
        }


class TrajectoryStore:
    """Async-safe FIFO store with staleness-bounded sampling.

    Rollout workers call `add()`; the trainer calls `get_batch()`. Both take the same lock, so
    the store is safe for many concurrent producers and one consumer.
    """

    def __init__(self, capacity: int = 512, max_staleness: int = 3) -> None:
        self.capacity = capacity
        self.max_staleness = max_staleness
        self.policy_version = 0
        self._buffer: deque[Trajectory] = deque()
        self._not_empty = asyncio.Condition()
        self.stats = StoreStats()

        # tracking the rollout policy
        self._version_generation = 0

    async def add(self, trajectory: Trajectory) -> None:
        # Add a finished trajectory, use a FIFO system, oldest gets evicted
        async with self._not_empty:
            while len(self._buffer) >= self.capacity:
                self._buffer.popleft() # get rid of the oldest trajectory
                self.stats.evicted_capacity += 1
            self._buffer.append(trajectory)
            self.stats.added += 1
            self._not_empty.notify_all() # anything waiting gets alerted of new trajectories! 

    def ready(self, batch_size: int, drop_stale: bool = True) -> bool:
        # do I have enough to make a batch?
        if drop_stale:
            num_ready_samples = sum(
                1 for t in self._buffer if t.staleness(self.policy_version) <= self.max_staleness
            )
        else:
            num_ready_samples = len(self._buffer)
        return num_ready_samples >= batch_size

    async def get_batch(
        self,
        batch_size: int,
        staleness_manager=None,
        drop_stale: bool = True,
    ) -> list[Trajectory]:
        """Wait until `batch_size` fresh trajectories exist, then drain them."""
        async with self._not_empty:
            batch: list[Trajectory] = []
            while len(batch) < batch_size:
                n = batch_size - len(batch)
                await self._not_empty.wait_for(
                    lambda n=n, drop_stale=drop_stale: self.ready(n, drop_stale=drop_stale)
                )
                while self._buffer and len(batch) < batch_size:
                    trajectory = self._buffer.popleft()
                    staleness = trajectory.staleness(self.policy_version)
                    if drop_stale and staleness > self.max_staleness:
                        self.stats.evicted_stale += 1
                        # Never reached training, so this is a reject, not a
                        # post-accept filter.
                        if staleness_manager is not None:
                            await staleness_manager.on_rollout_rejected()
                        continue
                    self.stats.staleness_histogram[staleness] += 1
                    batch.append(trajectory)
                    if staleness_manager is not None:
                        await staleness_manager.on_rollout_accepted()
            self.stats.sampled += len(batch)
            return batch

    async def set_policy_version(self, version: int) -> None:
        """Bump the policy version after a weight sync. Existing trajectories get staler"""
        async with self._not_empty:
            self.policy_version = version
            self._version_generation += 1
            # Waiters may now be unsatisfiable (everything went stale) -- wake them so they
            # re-evaluate rather than hanging until timeout.
            self._not_empty.notify_all()

    async def prune_stale(self, staleness_manager=None) -> int:
        # prunes stale entries - done every weight_sync
        async with self._not_empty:
            keep = [
                t for t in self._buffer if t.staleness(self.policy_version) <= self.max_staleness
            ]
            removed = len(self._buffer) - len(keep)
            self._buffer = deque(keep)
            self.stats.evicted_stale += removed
            if staleness_manager is not None:
                for _ in range(removed):
                    await staleness_manager.on_rollout_rejected()
            return removed

    def __len__(self) -> int:
        return len(self._buffer)

    def metrics(self) -> dict[str, float]:
        """Store metrics for the trainer log.

        Watch `store_evicted_stale` and `store_mean_staleness`: a large stale-eviction
        fraction means rollout is too slow relative to the treainer, and the fix is to
        rebalance the GPU split (more rollout workers) rather than to raise the bound.
        """
        metrics = self.stats.as_metrics()
        metrics["store_size"] = float(len(self._buffer))
        metrics["store_policy_version"] = float(self.policy_version)
        return metrics
