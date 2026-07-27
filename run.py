"""Entrypoint: wires rollout workers, the trajectory store, and the trainer together.

The two clocks run independently. Rollout workers generate continuously and push into the
store; the trainer pulls batches whenever enough non-stale data exists. Neither waits for
the other, which is exactly what produces off-policy data with staleness K > 0 -- and K=3
is the operating point this whole implementation is built around.

GPU LAYOUT (8x H100). Rollout GPUs come first because vLLM's multiprocess workers pick
cuda:0..TP-1 by worker index; each trainer rank then pins cuda:{n_rollout_gpus + rank}.
The trainer is DDP across n_trainer_gpus ranks -- rank 0 owns everything async (rollout
loop, store, hints, judge, vLLM weight sync) and broadcasts each batch; ranks 1..N-1 run
a bare receive -> train_step loop. The hinted teacher needs no GPU of its own: it is the
same weights as the student, and every DDP rank holds a full copy, so each rank runs its
own teacher forward locally under no_grad.

Run:
    torchrun --nproc-per-node=4 run.py     # full run on Qwen3-4B, all 8 GPUs
    python run.py --smoke                  # 10 steps on Qwen3-0.6B, 2 GPUs, single process
    python run.py --baseline               # zero-shot judge score, no training
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from dotenv import load_dotenv

# Must run before reward.judge is touched: the judge reads OPENROUTER_API_KEY from the
# environment when it constructs its generator.
load_dotenv()

from config import CONFIG, Config  # noqa: E402
from data.dataset import load_tasks, split_tasks  # noqa: E402
from reward.judge import RubricJudge, summarize  # noqa: E402
from train.rollout import RolloutEngine  # noqa: E402
from train.store import TrajectoryStore  # noqa: E402
from train.trainer import SDPOTrainer, log_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sdpo")


def _dist_info() -> tuple[int, int]:
    """(rank, world_size) under torchrun; (0, 1) when launched as a plain process."""
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def build_trainer(config: Config, smoke: bool, store: TrajectoryStore, tasks) -> SDPOTrainer:
    """Model + tokenizer + (maybe DDP) + trainer, pinned to this rank's own GPU.

    The pin must happen before any CUDA work in this process: the trainer must never touch
    the rollout GPUs, where vLLM preallocates gpu_memory_utilization of the whole card.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, world = _dist_info()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{config.n_rollout_gpus + local_rank}")
    torch.cuda.set_device(device)

    model_name = config.smoke_model if smoke else config.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, config.dtype)
    ).to(device)
    if world > 1:
        # DDP: every rank holds full weights (Qwen3-4B fits on one H100), only gradients
        # are all-reduced. The constructor broadcasts rank 0's weights, so all ranks start
        # identical and stay identical.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    return SDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        store=store,
        config=config,
        tasks_by_id={t.task_id: t for t in tasks},
        device=device,
    )


def save_checkpoint(trainer: SDPOTrainer, output_dir: str, tag) -> None:
    """Write model + optimizer + trainer state (incl. the EMA clipper) to output_dir."""
    path = Path(output_dir) / f"step_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), path / "state.pt")
    logger.info("checkpoint saved: %s", path / "state.pt")


async def sync_weights(trainer: SDPOTrainer, engine: RolloutEngine, store, new_version: int) -> None:
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

    for names, dtype_names, shapes, tensors in trainer.weight_buckets():
        receive = asyncio.create_task(
            engine.receive_weight_bucket(names, dtype_names, shapes)
        )
        await asyncio.sleep(0)  # let the receive RPC dispatch before we send
        await asyncio.to_thread(trainer.send_weight_bucket, names, tensors)
        await asyncio.wait_for(receive, timeout=trainer.config.weight_sync_timeout_s)

    await engine.finish_update(new_version)
    await store.set_policy_version(new_version)


async def evaluate(engine: RolloutEngine, judge: RubricJudge, tasks) -> dict[str, float]:
    """Generate on held-out tasks and score with the rubric judge.

    Held-out only: these tasks are never trained on, so the gap between this score and
    train-set performance is the overfitting signal -- which matters here because the
    rubric is effectively an answer key.
    """
    results = await asyncio.gather(*(engine.generate(task) for task in tasks))
    scored = await judge.score_all(
        [(task, result.text) for task, result in zip(tasks, results)]
    )
    return summarize(scored)


async def train(config: Config, smoke: bool = False) -> None:
    """Rank 0: the async orchestrator plus trainer rank 0."""
    _, world = _dist_info()

    tasks = load_tasks(config.dataset_name, config.dataset_split)
    train_tasks, heldout_tasks = split_tasks(tasks, config.n_heldout, config.split_seed)
    if smoke:
        train_tasks, heldout_tasks = train_tasks[:4], heldout_tasks[:2]
    logger.info("loaded %d train / %d held-out tasks", len(train_tasks), len(heldout_tasks))

    store = TrajectoryStore(config.store_capacity, config.max_staleness)
    trainer = build_trainer(config, smoke, store, tasks)

    engine = RolloutEngine(config, model=config.smoke_model if smoke else config.model)
    engine.start()

    judge = RubricJudge(
        config.judge_model,
        config.judge_max_concurrency,
        config.judge_max_retries,
        config.judge_timeout,
    )

    # Weight-sync group: one rank per vLLM GPU WORKER (i.e. tensor_parallel_size ranks per
    # engine, not one per engine) plus rank 0 for the trainer as sender. Only trainer
    # rank 0 joins -- DDP ranks 1..N-1 are not part of this group.
    weight_world_size = config.n_rollout_gpus + 1
    # Both sides must join concurrently -- each blocks until the rendezvous completes.
    await asyncio.gather(
        engine.init_weight_update_group(
            master_address=config.weight_sync_host,
            master_port=config.weight_sync_port,
            rank_offset=1,           # rank 0 is the trainer
            world_size=weight_world_size,
        ),
        asyncio.to_thread(
            trainer.setup_weight_sync,
            config.weight_sync_host, config.weight_sync_port, weight_world_size,
        ),
    )

    stop = asyncio.Event()
    rollout_task = asyncio.create_task(engine.run_forever(store, train_tasks, stop))

    try:
        for step in range(config.total_steps):
            batch = await store.wait_for_batch(config.batch_size, timeout=300.0)
            # Every DDP rank must get a non-empty shard: a rank with nothing to train on
            # would skip its backward and the others would hang in the gradient all-reduce.
            if len(batch) < world:
                logger.warning("no non-stale trajectories available; rollout may be too slow")
                continue

            if world > 1:
                dist.broadcast_object_list([{"cmd": "train", "trajectories": batch}], src=0)
            metrics = trainer.train_step(batch[0::world])
            if step % config.log_interval == 0:
                log_metrics(metrics, store)

            # Weight sync: push the new weights, then bump the version so everything
            # already in the store ages by one step and anything past K=3 is evicted on
            # the next sample.
            if (step + 1) % config.weight_sync_interval == 0:
                version = trainer.bump_policy_version()
                await sync_weights(trainer, engine, store, version)

            if (step + 1) % config.eval_interval == 0:
                eval_metrics = await evaluate(engine, judge, heldout_tasks)
                logger.info("EVAL step %d: %s", step + 1, eval_metrics)

            if (step + 1) % config.checkpoint_interval == 0:
                save_checkpoint(trainer, config.output_dir, step + 1)

        save_checkpoint(trainer, config.output_dir, "final")
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
    tasks = load_tasks(config.dataset_name, config.dataset_split)
    # The store is a constructor requirement; workers never touch it -- rank 0 owns the
    # only real store.
    trainer = build_trainer(
        config, smoke, TrajectoryStore(config.store_capacity, config.max_staleness), tasks
    )
    logger.info("worker rank %d ready on %s", rank, trainer.device)

    while True:
        message = [None]
        dist.broadcast_object_list(message, src=0)
        if message[0]["cmd"] == "stop":
            logger.info("worker rank %d stopping", rank)
            return
        trainer.train_step(message[0]["trajectories"][rank::world])


async def baseline(config: Config) -> None:
    """Zero-shot held-out judge score, before any training. The number every later eval
    must beat for the run to have meant anything."""
    tasks = load_tasks(config.dataset_name, config.dataset_split)
    _, heldout = split_tasks(tasks, config.n_heldout, config.split_seed)

    engine = RolloutEngine(config)
    engine.start()
    judge = RubricJudge(
        config.judge_model,
        config.judge_max_concurrency,
        config.judge_max_retries,
        config.judge_timeout,
    )
    try:
        metrics = await evaluate(engine, judge, heldout)
        logger.info("ZERO-SHOT BASELINE: %s", metrics)
    finally:
        await engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Off-policy SDPO on diligence-bench")
    parser.add_argument("--smoke", action="store_true", help="tiny model, 4 tasks, 10 steps")
    parser.add_argument("--baseline", action="store_true", help="zero-shot eval, no training")
    parser.add_argument(
        "--hint-prompt",
        choices=("answer_free", "answer_bearing"),
        default=None,
        help="ablation arm: may the generated hint state the answer",
    )
    args = parser.parse_args()

    config = CONFIG
    if args.hint_prompt:
        config = replace(config, error_hint_prompt=args.hint_prompt)
    if args.smoke:
        config = replace(config, total_steps=10, batch_size=4, mini_batch_size=2,
                         eval_interval=5, n_rollout_gpus=1, n_trainer_gpus=1,
                         n_teacher_gpus=0)

    rank, world = _dist_info()
    if world > 1:
        if world != config.n_trainer_gpus:
            raise SystemExit(
                f"torchrun world size {world} != n_trainer_gpus {config.n_trainer_gpus}; "
                f"launch with torchrun --nproc-per-node={config.n_trainer_gpus} run.py"
            )
        if config.batch_size % world != 0:
            raise SystemExit(f"batch_size {config.batch_size} not divisible by world size {world}")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(config.n_rollout_gpus + local_rank)
        # Generous timeout: ranks 1..N-1 sit blocked in a broadcast while rank 0 waits up
        # to 300s for rollouts or runs a multi-minute eval; the default NCCL watchdog
        # would kill the run for that.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    elif config.n_trainer_gpus > 1 and not args.baseline:
        logger.warning(
            "launched as a single process with n_trainer_gpus=%d -- only one trainer GPU "
            "will be used; launch with torchrun --nproc-per-node=%d run.py to use them all",
            config.n_trainer_gpus, config.n_trainer_gpus,
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
