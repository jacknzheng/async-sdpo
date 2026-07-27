"""vLLM rollout engine: generation plus native NCCL weight sync.

Rollout workers run the current policy on training tasks and stream finished trajectories
into the store. The trainer pulls from the store on its own schedule, which is what makes
the run off-policy and is the whole point of the exercise.

WEIGHT SYNC uses vLLM's native `vllm.distributed.weight_transfer` API -- the same approach
as the healthbench-rl project. The engine is constructed with
`WeightTransferConfig(backend="nccl")`, and the update is a four-call transaction:

    pause_generation(mode="keep") -> start_weight_update()
      -> update_weights(...)  [once per bucket]
      -> finish_weight_update() -> resume_generation()

This is much better than hand-rolling `collective_rpc` against a StatelessProcessGroup:
vLLM owns the group lifecycle, the receive side posts its own matching broadcasts from the
names/dtypes/shapes metadata carried in the request, and bucketing is supported natively
via `packed=True`. There is no worker extension and no private `load_weights` call, so
none of this rides on unversioned internals.

Ordering still matters: the receivers must have their `update_weights` RPC in flight before
the trainer enters `trainer_send_weights`, or the sender blocks on a broadcast nobody is
listening for. The orchestrator owns that interleaving -- see `run.py`.

Two generation details are correctness-critical:

1. `logprobs_mode="processed_logprobs"`. The IS ratio's denominator must be the log-prob
   under the distribution the sampler ACTUALLY drew from -- after temperature and any
   logit processors. vLLM's default (`raw_logprobs`) is pre-processing, which would bias
   every ratio by exactly the sampling transform: silent and systematic, not a crash.

2. `logprobs=0` returns exactly the sampled token's own log-prob, which is all the
   sampled-token SDPO estimator needs. vLLM's sampler concatenates the sampled token's
   logprob first, so it is always present.

VERSION PIN: written against vLLM 0.26.x, which is where the native weight-transfer API
(`init_weight_transfer_engine` / `start_weight_update` / `update_weights` /
`finish_weight_update`) lives. It does NOT exist in 0.11/0.12 -- those releases need the
older `collective_rpc` + `worker_extension_cls` path. Pin exactly.

This module is the only one that needs real GPUs, so it is deliberately thin -- the loss,
the store, the batching, and the trainer are all testable without it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import asdict, dataclass

from config import Config
from data.dataset import Task, build_prompt
from data.hint import build_error_hint
from train.store import Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)

# vLLM uses CUDA in subprocesses, which requires spawn rather than fork.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class RolloutResult:
    """One generation, with the per-token log-probs recorded at sampling time."""

    task_id: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprobs: list[float]
    text: str


class RolloutEngine:
    """Async wrapper over vLLM's AsyncLLM for off-policy trajectory generation."""

    def __init__(self, config: Config, model: str | None = None, seed: int = 0) -> None:
        self.config = config
        self.model = model or config.model
        self.seed = seed
        self.engine = None
        self.policy_version = 0
        self._weight_group_ready = False
        self._request_counter = 0
        self._hint_semaphore = asyncio.Semaphore(config.error_hint_concurrency)
        # Rollouts discarded because no hint could be generated. Watch this: it is wasted
        # generation compute, and a sustained nonzero value means the hint model is the
        # bottleneck on data production, not the rollout engine.
        self.hintless_dropped = 0

    def start(self) -> None:
        """Build the AsyncLLM engine. vLLM is imported lazily so the rest of the package
        stays importable on machines without it (e.g. the dev laptop running the tests)."""
        from vllm import AsyncEngineArgs
        from vllm.config import WeightTransferConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(
            model=self.model,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.n_rollout_gpus,
            distributed_executor_backend=(
                "mp" if self.config.n_rollout_gpus > 1 else "uni"
            ),
            # Opts the engine into the native NCCL weight-transfer path.
            weight_transfer_config=WeightTransferConfig(backend="nccl"),
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enforce_eager=False,
            enable_prefix_caching=True,
            max_model_len=self.config.max_prompt_tokens + self.config.max_response_tokens,
            seed=self.seed,
            # See module docstring -- the IS ratio denominator must come from the same
            # processed distribution the sampler drew from.
            logprobs_mode="processed_logprobs",
        )
        # Synchronous: from_engine_args is not a coroutine, and the engine core process
        # starts in __init__. There is no start()/await to call.
        self.engine = AsyncLLM.from_engine_args(engine_args)
        logger.info(
            "rollout engine ready: %s on %d GPU(s)", self.model, self.config.n_rollout_gpus
        )

    def _sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_response_tokens,
            logprobs=0,  # sampled token's own logprob only -- see module docstring
        )

    # ---- generation ----

    async def generate(self, task: Task, prompt_token_ids: list[int] | None = None) -> RolloutResult:
        """Generate one completion. The prompt carries NO hint -- the student answers the
        bare query; only the teacher's context gets the hint."""
        if self.engine is None:
            raise RuntimeError("engine not started; call start() first")

        from vllm import TokensPrompt

        if prompt_token_ids is None:
            prompt = build_prompt(task, hint=None)
            request_input = prompt
        else:
            request_input = TokensPrompt(prompt_token_ids=prompt_token_ids)

        self._request_counter += 1
        request_id = f"{task.task_id}-{self._request_counter}-{uuid.uuid4().hex[:6]}"

        # vLLM streams cumulative snapshots; the last one holds the full completion.
        last_output = None
        async for output in self.engine.generate(
            prompt=request_input, sampling_params=self._sampling_params(), request_id=request_id
        ):
            last_output = output

        if last_output is None:
            raise RuntimeError(f"no output for request {request_id}")
        # An aborted request can come back finished but with no completions.
        if not last_output.outputs:
            raise RuntimeError(f"request {request_id} finished with no output (aborted?)")

        completion = last_output.outputs[0]
        return RolloutResult(
            task_id=task.task_id,
            prompt_token_ids=list(last_output.prompt_token_ids),
            response_token_ids=list(completion.token_ids),
            logprobs=_extract_sampled_logprobs(completion),
            text=completion.text,
        )

    async def _error_hint(self, task: Task, response_text: str) -> str:
        """Generate the teacher's hint for one rollout. Returns "" if it could not be made.

        Bounded by a semaphore: one LLM call per trajectory, and rollout runs continuously,
        so an unbounded fan-out would open a call for every in-flight generation at once.
        """
        try:
            async with self._hint_semaphore:
                return await build_error_hint(
                    query=task.query,
                    sections=task.sections,
                    response_text=response_text,
                    prompt_variant=self.config.error_hint_prompt,
                    model=self.config.error_hint_model,
                    timeout=self.config.error_hint_timeout,
                )
        except Exception:  # noqa: BLE001 -- one failed hint must not kill the worker
            logger.exception("error-hint failed for task %s", task.task_id)
            return ""

    async def run_forever(
        self,
        store: TrajectoryStore,
        tasks: list[Task],
        stop_event: asyncio.Event,
        concurrency: int = 8,
    ) -> None:
        """Continuously sample tasks, generate, and push trajectories into the store.

        Tasks are cycled rather than exhausted: with 120 training tasks and a long run the
        same task is revisited many times. That is fine, and is not the same as reusing a
        trajectory -- each visit draws a fresh completion from the current policy, which is
        new gradient signal.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def one(task: Task, version: int) -> None:
            async with semaphore:
                try:
                    result = await self.generate(task)
                except Exception:  # noqa: BLE001 -- one bad rollout must not kill the worker
                    logger.exception("rollout failed for task %s", task.task_id)
                    return
                # The hint must exist before the trajectory is stored: a trajectory with no
                # hint gives the teacher the bare query, making it identical to the student
                # and its gradient ~zero. Dropping it keeps the batch meaningful at the
                # cost of a wasted generation.
                hint = await self._error_hint(task, result.text)
                if not hint:
                    self.hintless_dropped += 1
                    logger.warning(
                        "dropping rollout for task %s: no hint could be generated",
                        task.task_id,
                    )
                    return
                await store.add(
                    Trajectory(
                        task_id=result.task_id,
                        prompt_token_ids=result.prompt_token_ids,
                        response_token_ids=result.response_token_ids,
                        rollout_logprobs=result.logprobs,
                        # Tagged with the version that GENERATED it, not the version
                        # current when it lands. This is what makes staleness measurable.
                        policy_version=version,
                        hint=hint,
                    )
                )

        index = 0
        pending: set[asyncio.Task] = set()
        while not stop_event.is_set():
            task = tasks[index % len(tasks)]
            index += 1
            pending.add(asyncio.create_task(one(task, self.policy_version)))
            pending = {t for t in pending if not t.done()}
            while len(pending) >= concurrency and not stop_event.is_set():
                await asyncio.sleep(0.05)
                pending = {t for t in pending if not t.done()}

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # ---- native weight sync ----

    async def init_weight_update_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int
    ) -> None:
        """Join every vLLM GPU worker in this engine to the trainer's NCCL group.

        `rank_offset` is this engine's first rank in the group. Each engine contributes
        `tensor_parallel_size` ranks -- one per GPU worker, NOT one per engine -- so the
        offsets must be spaced by TP size and `world_size` must count GPU workers, not
        engines. Rank 0 is the trainer (the sender).
        """
        from vllm.distributed.weight_transfer.base import WeightTransferInitRequest
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferInitInfo

        await asyncio.wait_for(
            self.engine.init_weight_transfer_engine(
                WeightTransferInitRequest(
                    init_info=asdict(
                        NCCLWeightTransferInitInfo(
                            master_address=master_address,
                            master_port=master_port,
                            rank_offset=rank_offset,
                            world_size=world_size,
                        )
                    )
                )
            ),
            timeout=self.config.weight_sync_timeout_s,
        )
        self._weight_group_ready = True
        logger.info("joined weight-sync group at rank offset %d", rank_offset)

    async def pause_for_update(self) -> None:
        """Freeze generation and open the weight-update transaction.

        `mode="keep"` suspends in-flight rollouts rather than aborting them, so a long
        generation survives the swap and resumes under the NEW weights.
        """
        if not self._weight_group_ready:
            raise RuntimeError("weight sync not initialized; call init_weight_update_group")
        await self.engine.pause_generation(mode="keep")
        await self.engine.start_weight_update()

    async def receive_weight_bucket(
        self, names: list[str], dtype_names: list[str], shapes: list[list[int]]
    ) -> None:
        """Post the receive side for one bucket of weights.

        The receivers derive their matching broadcasts from this metadata, so the trainer's
        send order must match `names` exactly. Must be IN FLIGHT before the trainer enters
        `trainer_send_weights` -- the orchestrator handles that interleaving.
        """
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferUpdateInfo

        await self.engine.update_weights(
            WeightTransferUpdateRequest(
                update_info=asdict(
                    NCCLWeightTransferUpdateInfo(
                        names=names, dtype_names=dtype_names, shapes=shapes, packed=True,
                    )
                )
            )
        )

    async def finish_update(self, new_version: int) -> None:
        """Close the transaction and resume generation under the new weights."""
        await self.engine.finish_weight_update()
        # The prefix cache holds KV computed under the OLD weights. Reusing it would splice
        # two policy versions into a single trajectory, making `policy_version` a lie and
        # corrupting the staleness accounting the whole design rests on.
        await self.engine.reset_prefix_cache()
        await self.engine.resume_generation()
        self.policy_version = new_version

    async def shutdown(self) -> None:
        if self.engine is None:
            return
        self.engine.shutdown()  # synchronous, not a coroutine
        self.engine = None


def _extract_sampled_logprobs(completion) -> list[float]:
    """Pull the log-prob of each SAMPLED token, aligned 1:1 with `token_ids`.

    vLLM returns `logprobs` as one dict per generated position, mapping token_id ->
    Logprob. With `logprobs=0` the sampled token is the only entry, but we look it up by id
    rather than taking the sole value, so this stays correct if someone raises `logprobs`
    to inspect alternatives -- with logprobs>0 the dict holds alternatives too, and taking
    the most likely one would bias every ratio toward 1. A missing entry would silently
    misalign the IS ratio, so it raises instead.
    """
    if completion.logprobs is None:
        raise ValueError(
            "rollout returned no logprobs; SamplingParams(logprobs=...) must be set -- "
            "they cannot be recomputed later without destroying the off-policy correction"
        )

    out: list[float] = []
    for position, token_id in enumerate(completion.token_ids):
        entry = completion.logprobs[position].get(token_id)
        if entry is None:
            raise ValueError(f"sampled token {token_id} missing from its logprob dict")
        out.append(entry.logprob)

    if len(out) != len(completion.token_ids):
        raise ValueError(
            f"extracted {len(out)} logprobs for {len(completion.token_ids)} tokens"
        )
    return out
