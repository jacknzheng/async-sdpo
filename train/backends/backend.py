"""
=============================================================================
INVARIANTS A TYPE SIGNATURE CANNOT ENFORCE -- read before writing a backend
=============================================================================

1. POST-PROCESSING LOGPROBS. `generate()` must return, for each sampled token, its
   log-prob under the distribution the sampler ACTUALLY drew from -- after temperature,
   top-p, and any logit processors. This is the denominator of the IS ratio in
   `train/loss.py`. vLLM needs `logprobs_mode="processed_logprobs"`; its default
   (`raw_logprobs`) is PRE-processing and would bias every ratio by exactly the sampling
   transform. That failure is silent and systematic -- no crash, a plausible loss curve,
   and a wrong gradient. SGLang's semantics differ and must be VERIFIED, not assumed.

2. PREFIX-CACHE RESET. `finish_update()` must reset any prefix/KV cache the engine holds.
   That cache was computed under the OLD weights; reusing it splices two policy versions
   into a single trajectory, which makes `Trajectory.policy_version` a lie and corrupts
   the staleness accounting the whole off-policy design rests on.

3. BUCKET ORDERING. The receiver derives its matching broadcasts from the `names` metadata
   carried in the update request, so the sender must transmit tensors in exactly the order
   `names` lists them.

4. RECEIVE-BEFORE-SEND. The receive call must be in flight before the send begins, or the
   sender blocks on a broadcast nobody is listening for. 

Deliberately ABSENT: `sleep`/`wake_up`. Those exist in frameworks that colocate the
trainer and the engine on the same GPUs and time-multiplex them. We are permanently
disaggregated -- rollout GPUs and trainer GPUs are disjoint -- so there is nothing to
swap out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class WeightBucket:
    """One group of parameters travelling together in a single weight transfer.

    Bucketing amortizes per-transfer overhead across the many small tensors in a
    transformer. `names`, `dtype_names`, `shapes`, and `tensors` are parallel lists -- all
    four are the same length and index i describes one parameter.

    The metadata (everything but `tensors`) is what the receiver needs to post its own
    matching broadcasts, so it travels to the engine while `tensors` go over the transport.
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    tensors: list[torch.Tensor]

    def __post_init__(self) -> None:
        n = len(self.names)
        if not (len(self.dtype_names) == len(self.shapes) == len(self.tensors) == n):
            raise ValueError(
                f"WeightBucket fields must be parallel lists: {n} names, "
                f"{len(self.dtype_names)} dtypes, {len(self.shapes)} shapes, "
                f"{len(self.tensors)} tensors"
            )

    def __len__(self) -> int:
        return len(self.names)

# types, bounding what our inference engine methods must include
@runtime_checkable
class InferenceEngine(Protocol):
    """Generation plus the receive side of weight sync.

    `policy_version` is the version of the weights currently loaded. The orchestrator reads
    it when tagging trajectories, so it must be updated by `finish_update`, never earlier --
    a trajectory tagged with a version the engine was not actually running is exactly the
    bookkeeping error that makes staleness meaningless.
    """

    policy_version: int

    def start(self) -> None:
        """Build the underlying engine. Separate from __init__ so construction is cheap and
        importable without GPUs (see the lazy-import note in each backend)."""
        ...

    async def generate(self, task, prompt_token_ids: list[int] | None = None):
        """Generate one completion, returning a `RolloutResult`.

        Must honour invariant 1 above: the returned per-token logprobs are the sampled
        tokens' own log-probs under the processed sampling distribution, aligned 1:1 with
        the response token ids.
        """
        ...

    async def init_weight_update_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int
    ) -> None:
        """Join this engine's GPU workers to the trainer's weight-transfer group.

        `world_size` counts GPU WORKERS plus the trainer, not engines: an engine with
        tensor parallelism contributes one rank per GPU worker. `rank_offset` is this
        engine's first rank; rank 0 is the trainer (the sender).
        """
        ...

    async def pause_for_update(self) -> None:
        """Freeze generation and open the weight-update transaction.

        In-flight generations should be suspended rather than aborted where the backend
        supports it, so a long rollout survives the swap and resumes under new weights.
        """
        ...

    async def receive_weight_bucket(
        self, names: list[str], dtype_names: list[str], shapes: list[list[int]]
    ) -> None:
        """Post the receive side for one bucket. Must be IN FLIGHT before the matching
        send begins -- see invariant 4."""
        ...

    async def finish_update(self, new_version: int) -> None:
        """Close the transaction, reset the prefix cache (invariant 2), resume generation,
        and set `policy_version = new_version`."""
        ...

    async def shutdown(self) -> None:
        """Release engine resources. Must be safe to call when never started."""
        ...


@runtime_checkable
class WeightTransport(Protocol):
    """The send side of weight sync, owned by the trainer.

    This is the seam that keeps a specific inference engine's collectives out of
    `train/trainer.py`. The trainer produces `WeightBucket`s and hands them here; how the
    bytes reach the engine is entirely the transport's business.
    """

    def setup(self, master_address: str, master_port: int, world_size: int) -> None:
        """Join the transfer group as the SENDER at rank 0.

        Blocks until every rank has joined, so the engine side must already be initializing
        concurrently -- the orchestrator issues both at once.
        """
        ...

    def send_bucket(self, bucket: WeightBucket) -> None:
        """Broadcast one bucket. Synchronous and blocking: the orchestrator runs it in a
        thread so the event loop stays free to service the receive RPC."""
        ...

    def teardown(self) -> None:
        """Release transport resources. Must be safe to call when never set up."""
        ...
