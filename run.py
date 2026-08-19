from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import timedelta
from pathlib import Path

from train.store import Trajectory
from data.dataset import Task

import torch
import torch.distributed as dist
from dotenv import load_dotenv

# Must run before reward.judge is touched: the judge reads OPENROUTER_API_KEY from the
# environment when it constructs its generator.
load_dotenv()

# Persistent caches, BEFORE the project imports below: huggingface_hub reads HF_HOME at
# import time, and the vLLM engine processes rank 0 spawns inherit this environment.
# /workspace is RunPod's volume disk -- it survives pod stop/start, so everything cached
# here (model weights, vLLM's torch.compile artifacts, the trainer's inductor/Triton
# kernels) is paid for once per pod, not once per boot. All setdefault: whatever the user
# exports wins. Single engine -> one shared VLLM_CACHE_ROOT (healthbench-rl's per-engine
# dirs guard against multi-engine races we don't have).
if os.path.isdir("/workspace"):
    os.environ.setdefault("VLLM_CACHE_ROOT", "/workspace/.cache/vllm")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/workspace/.cache/torchinductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/.cache/triton")
    os.environ.setdefault("HF_HOME", "/workspace/hf")
# Qwen3-8B pure-bf16 AdamW sits at ~65 of 80 GB per trainer rank (16 weights + 16 grads
# + 33 optimizer state); expandable segments lets the allocator grow in place instead of
# fragmenting. Drop this line if vLLM ever objects -- it is an aid, not a requirement.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train.config import Config  # noqa: E402
from data.dataset import load_tasks, split_tasks  # noqa: E402
from reward.judge import RubricJudge, summarize  # noqa: E402
from train.backends import get_backend  # noqa: E402
from train.backends.backend import InferenceEngine  # noqa: E402
from train.store import TrajectoryStore  # noqa: E402
from train.trainer import SDPOTrainer, log_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sdpo")


def _dist_info() -> tuple[int, int]:
    """(rank, world_size) under torchrun; (0, 1) when launched as a plain process."""
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def build_trainer(
    config: Config,
    smoke: bool,
    store: TrajectoryStore,
    tasks,
    transport=None,
) -> SDPOTrainer:
    """Model + tokenizer + (maybe FSDP2) + trainer, pinned to this rank's own GPU.

    The pin must happen before any CUDA work in this process: the trainer must never touch
    the rollout GPUs, where the engine preallocates gpu_memory_utilization of the whole card.

    `transport` is the send side of weight sync and is passed only on rank 0 -- ranks
    1..N-1 never push weights, so they get None and would raise if they tried.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, world = _dist_info()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{config.generator.engine.n_rollout_gpus + local_rank}")
    torch.cuda.set_device(device)

    model_name = config.model.smoke_model if smoke else config.model.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, config.model.dtype)
    ).to(device)
    if world > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
        )

        # FSDP2 shards weights, gradients, and optimizer state across the trainer ranks,
        # so per-rank cost is ~total/N instead of DDP's full copy -- that is what lets a
        # 27B model fit at all. The cost is communication: weights are all-gathered on
        # every forward and again on backward.
        #
        # One mesh dimension over the DEFAULT process group, which torchrun sized to the
        # trainer ranks only -- the rollout GPUs are separate processes and never join it.
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))

        # param_dtype: what all-gathered weights are cast to for compute.
        # reduce_dtype fp32: the gradient reduce-scatter accumulates in fp32 so bf16
        # rounding error does not compound across ranks. Cheap, and the usual default.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=getattr(torch, config.model.dtype),
            reduce_dtype=torch.float32,
        )
        fsdp_kwargs = {"mesh": mesh, "mp_policy": mp_policy}
        if config.trainer.fsdp.cpu_offload:
            fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=True)

        # Shard each transformer block as its own FSDP unit, THEN the root. Without the
        # per-layer pass the whole model becomes one unit and the all-gather would
        # materialize every parameter at once, defeating the point of sharding. The root
        # must come last: fully_shard sees already-sharded children and treats them as
        # separate groups.
        for layer in model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
    if config.trainer.compile_trainer:
        # Compile AFTER fully_shard, the reverse of the old DDP order. fully_shard mutates
        # the module in place and returns the same object, so unlike DDP it introduces no
        # `module.` prefix -- parameter names keep their real HF spelling for vLLM's weight
        # receiver and the checkpoints.
        # dynamic=True because padded batch shapes differ every step -- a static compile
        # would recompile per shape. Kernels cache to TORCHINDUCTOR_CACHE_DIR (above), so
        # only the first-ever step pays the compile.
        model = torch.compile(model, dynamic=True)

    return SDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        store=store,
        config=config,
        tasks_by_id={t.task_id: t for t in tasks},
        device=device,
        transport=transport,
    )


def save_checkpoint(trainer: SDPOTrainer, output_dir: str, tag, world: int = 1) -> None:
    """Write model + optimizer + trainer state (incl. the EMA clipper) to output_dir.

    Under FSDP2 `trainer.state_dict()` gathers sharded DTensors, which is a COLLECTIVE:
    every rank must call it even though only rank 0 holds the assembled result and writes
    the file. Same hazard as sync_weights -- skip it on the workers and rank 0 hangs.
    """
    if world > 1:
        dist.broadcast_object_list([{"cmd": "checkpoint"}], src=0)

    state = trainer.state_dict()
    path = Path(output_dir) / f"step_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "state.pt")
    logger.info("checkpoint saved: %s", path / "state.pt")


async def sync_weights(
    trainer: SDPOTrainer, engine: InferenceEngine, store, new_version: int, world: int = 1
) -> None:
    """Push the trainer's weights into the rollout engine, then advance the version.

    The four-call transaction, per bucket:

        pause_generation(mode="keep") + start_weight_update()
          -> for each bucket: post receive RPC, THEN trainer_send_weights, then join
        finish_weight_update() + reset_prefix_cache() + resume_generation()

    ORDERING: the receive-side `update_weights` RPC must be in flight before the trainer
    enters `trainer_send_weights`, or the sender blocks on a broadcast nobody is listening
    for. We create the receive task, yield once so it dispatches, run the (blocking,
    synchronous) send in a thread so the event loop stays free, then join.
    """
    await engine.pause_for_update()

    # weight_buckets() calls DTensor.full_tensor(), which is a COLLECTIVE over the FSDP
    # mesh: every trainer rank must walk the identical generator or rank 0 blocks forever
    # in the first gather. Ranks 1..N-1 are parked in worker_loop's broadcast; this
    # releases them to participate. They gather and discard -- only rank 0 has a transport.
    if world > 1:
        dist.broadcast_object_list([{"cmd": "sync_weights"}], src=0)

    for bucket in trainer.weight_buckets():
        receive = asyncio.create_task(
            engine.receive_weight_bucket(bucket.names, bucket.dtype_names, bucket.shapes)
        )
        await asyncio.sleep(0)  # let the receive RPC dispatch before we send
        await asyncio.to_thread(trainer.send_weight_bucket, bucket)
        await asyncio.wait_for(
            receive, timeout=trainer.config.generator.engine.weight_sync_timeout_s
        )

    await engine.finish_update(new_version)
    await store.set_policy_version(new_version)
    await store.prune_stale()

async def evaluate(engine: InferenceEngine, judge: RubricJudge, tasks) -> dict[str, float]:
    """Generate on held-out tasks and score with the rubric judge.

    Held-out only: these tasks are never trained on, so the gap between this score and
    train-set performance is the overfitting signal
    """
    results = await asyncio.gather(*(engine.generate(task) for task in tasks))
    scored = await judge.score_all(
        [(task, result.text) for task, result in zip(tasks, results)]
    )
    return summarize(scored)

async def run_loop(
    engine,
    store: TrajectoryStore,
    tasks: list[Task],
    stop_event: asyncio.Event,
    concurrency: int = 8,
) -> None:
    """
    Continuously sample tasks, generate, and push trajectories into the store.
    This continues until stop_event (asyncio.Event) 
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(task: Task, version: int) -> None:
        async with semaphore:
            try:
                result = await engine.generate(task)
            except Exception: 
                logger.exception("rollout failed for task %s", task.task_id)
                return
            # generate the hint before adding to the store!
            hint = await engine._error_hint(task, result.text)
            if not hint:
                engine.hintless_dropped += 1
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
        pending.add(asyncio.create_task(one(task, engine.policy_version)))
        pending = {t for t in pending if not t.done()}
        while len(pending) >= concurrency and not stop_event.is_set():
            await asyncio.sleep(0.05)
            pending = {t for t in pending if not t.done()}

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def train(config: Config, smoke: bool = False) -> None:
    """Rank 0: the async orchestrator plus trainer rank 0."""

    ########################
    ## SETUP THE TRAINERS ##
    ######################## 

    _, world = _dist_info()

    tasks = load_tasks(config.data.dataset_name, config.data.dataset_split)
    train_tasks, heldout_tasks = split_tasks(
        tasks, config.data.n_heldout, config.data.split_seed
    )
    if smoke:
        train_tasks, heldout_tasks = train_tasks[:4], heldout_tasks[:2]
    logger.info("loaded %d train / %d held-out tasks", len(train_tasks), len(heldout_tasks))

    # Engine and transport are always chosen together and from the same backend: the
    # receiver derives its broadcasts from metadata the sender chose, so a mismatched pair
    # would hang the rendezvous rather than fail loudly.
    engine_cls, transport_cls = get_backend(config.generator.engine.backend)

    store = TrajectoryStore(
        config.generator.engine.store_capacity, config.trainer.algorithm.max_staleness
    )
    trainer = build_trainer(config, smoke, store, tasks, transport=transport_cls())

    engine = engine_cls(
        config, model=config.model.smoke_model if smoke else config.model.model
    )
    engine.start()

    judge = RubricJudge(
        config.judge.model,
        config.judge.max_concurrency,
        config.judge.max_retries,
        config.judge.timeout,
    )

    # Weight-sync group: vLLM rollout workers + one trainer GPU
    weight_world_size = config.generator.engine.n_rollout_gpus + 1
    # Both sides must join concurrently -- each blocks until the rendezvous completes.
    await asyncio.gather(
        engine.init_weight_update_group(
            master_address=config.generator.engine.weight_sync_host,
            master_port=config.generator.engine.weight_sync_port,
            rank_offset=1,           # rank 0 is the trainer
            world_size=weight_world_size,
        ),
        asyncio.to_thread(
            trainer.setup_weight_sync,
            config.generator.engine.weight_sync_host,
            config.generator.engine.weight_sync_port,
            weight_world_size,
        ),
    )

    stop = asyncio.Event()

    ##################
    ## ROLLOUT LOOP ##
    ##################

    rollout_task = asyncio.create_task(run_loop(engine, store, train_tasks, stop))

    ###################
    ## TRAINING LOOP ##
    ###################

    try:
        for step in range(config.trainer.total_steps):
            batch = await store.wait_for_batch(config.trainer.batch_size, timeout=300.0)
            # Every DDP rank must get a non-empty shard: a rank with nothing to train on
            # would skip its backward and the others would hang in the gradient all-reduce.
            if len(batch) < world:
                logger.warning("no non-stale trajectories available; rollout may be too slow")
                continue

            if world > 1:
                dist.broadcast_object_list([{"cmd": "train", "trajectories": batch}], src=0)
            metrics = trainer.train_step(batch[0::world])
            if step % config.logging.log_interval == 0:
                log_metrics(metrics, store)

            # Weight sync: push the new weights, then bump the version so everything
            # already in the store ages by one step and anything past K=3 is evicted on
            # the next sample.
            if (step + 1) % config.generator.engine.weight_sync_interval == 0:
                version = trainer.bump_policy_version()
                await sync_weights(trainer, engine, store, version, world)

            if (step + 1) % config.judge.eval_interval == 0:
                eval_metrics = await evaluate(engine, judge, heldout_tasks)
                logger.info("EVAL step %d: %s", step + 1, eval_metrics)

            if (step + 1) % config.logging.checkpoint_interval == 0:
                save_checkpoint(trainer, config.logging.output_dir, step + 1, world)

        save_checkpoint(trainer, config.logging.output_dir, "final", world)
    finally:
        if world > 1:
            # Release the workers from their blocking broadcast, even on a crash --
            # otherwise they hang until the NCCL timeout.
            dist.broadcast_object_list([{"cmd": "stop"}], src=0)
        stop.set()
        rollout_task.cancel()
        await engine.shutdown()

def worker_loop(config: Config, smoke: bool = False) -> None:
    """DDP ranks 1..N-1: receive a batch, train on this rank's shard, repeat.

    Weights stay identical to rank 0's because DDP averages gradients on every step's
    final backward and the optimizer is deterministic given those gradients.
    """
    rank, world = _dist_info()
    tasks = load_tasks(config.data.dataset_name, config.data.dataset_split)
    # The store is a constructor requirement; workers never touch it -- rank 0 owns the
    # only real store.
    trainer = build_trainer(
        config,
        smoke,
        TrajectoryStore(
            config.generator.engine.store_capacity, config.trainer.algorithm.max_staleness
        ),
        tasks,
    )
    logger.info("worker rank %d ready on %s", rank, trainer.device)

    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        command = message[0]["cmd"]

        if command == "stop":
            logger.info("worker rank %d stopping", rank)
            return

        # The two commands below exist purely so this rank joins collectives that rank 0
        # is about to enter. Under FSDP2 the model state is sharded across every rank, so
        # reassembling it -- for a weight push or a checkpoint -- requires all of them.
        # Skipping either leaves rank 0 blocked until the NCCL timeout kills the run.
        if command == "sync_weights":
            # Participate in every .full_tensor() all-gather and throw the result away:
            # only rank 0 holds a transport and actually ships bytes to the engine.
            for _ in trainer.weight_buckets():
                pass
            continue

        if command == "checkpoint":
            # Same deal for get_state_dict: collective on all ranks, result only on 0.
            trainer.state_dict()
            continue

        trainer.train_step(message[0]["trajectories"][rank::world])


async def baseline(config: Config) -> None:
    """Zero-shot held-out judge score, before any training. The number every later eval
    must beat for the run to have meant anything."""
    tasks = load_tasks(config.data.dataset_name, config.data.dataset_split)
    _, heldout = split_tasks(tasks, config.data.n_heldout, config.data.split_seed)

    engine_cls, _ = get_backend(config.generator.engine.backend)
    engine = engine_cls(config)
    engine.start()
    judge = RubricJudge(
        config.judge.model,
        config.judge.max_concurrency,
        config.judge.max_retries,
        config.judge.timeout,
    )
    try:
        metrics = await evaluate(engine, judge, heldout)
        logger.info("ZERO-SHOT BASELINE: %s", metrics)
    finally:
        await engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Off-policy SDPO on diligence-bench",
        epilog="Any trailing dotted args override config, e.g. trainer.optimizer.learning_rate=1e-5",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny model, 4 tasks, 10 steps")
    parser.add_argument("--baseline", action="store_true", help="zero-shot eval, no training")
    parser.add_argument(
        "--hint-prompt",
        choices=("answer_free", "answer_bearing"),
        default=None,
        help="ablation arm: may the generated hint state the answer?",
    )
    # parse_known_args so `run.py --smoke trainer.optimizer.learning_rate=1e-5` works:
    # the flags below are ergonomic presets, the leftovers are arbitrary config overrides.
    args, overrides = parser.parse_known_args()

    # Presets first, user overrides last -- so an explicit dotted arg always wins over a
    # preset. Every rank re-executes this file with the same argv, so all ranks build an
    # identical config; there is no config broadcast.
    dotlist = []
    if args.smoke:
        # compile_trainer off: smoke exists to validate the plumbing in minutes, and
        # compiling the throwaway 0.6B model fights exactly that.
        dotlist += [
            "trainer.total_steps=10",
            "trainer.batch_size=4",
            "trainer.mini_batch_size=2",
            "trainer.n_trainer_gpus=1",
            "trainer.compile_trainer=false",
            "generator.engine.n_rollout_gpus=1",
            "judge.eval_interval=5",
        ]
    if args.hint_prompt:
        dotlist.append(f"generator.hint.prompt={args.hint_prompt}")
    config = Config.from_cli_overrides(dotlist + overrides)

    rank, world = _dist_info()
    if world > 1:
        if world != config.trainer.n_trainer_gpus:
            raise SystemExit(
                f"torchrun world size {world} != n_trainer_gpus {config.trainer.n_trainer_gpus}; "
                f"launch with torchrun --nproc-per-node={config.trainer.n_trainer_gpus} run.py"
            )
        if config.trainer.batch_size % world != 0:
            raise SystemExit(
                f"batch_size {config.trainer.batch_size} not divisible by world size {world}"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(config.generator.engine.n_rollout_gpus + local_rank)
        # Generous timeout: ranks 1..N-1 sit blocked in a broadcast while rank 0 waits up
        # to 300s for rollouts or runs a multi-minute eval; the default NCCL watchdog
        # would kill the run for that.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    elif config.trainer.n_trainer_gpus > 1 and not args.baseline:
        logger.warning(
            "launched as a single process with n_trainer_gpus=%d -- only one trainer GPU "
            "will be used; launch with torchrun --nproc-per-node=%d run.py to use them all",
            config.trainer.n_trainer_gpus, config.trainer.n_trainer_gpus,
        )

    try:
        if args.baseline:
            if rank == 0:
                asyncio.run(baseline(config))
        elif rank == 0:
            asyncio.run(train(config, smoke=args.smoke))
        else:
            worker_loop(config, smoke=args.smoke)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
