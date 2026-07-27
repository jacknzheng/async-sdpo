"""Entrypoint: wires rollout workers, the trajectory store, and the trainer together.

The two clocks run independently. Rollout workers generate continuously and push into the
store; the trainer pulls batches whenever enough non-stale data exists. Neither waits for
the other, which is exactly what produces off-policy data with staleness K > 0 -- and K=3
is the operating point this whole implementation is built around.

Run:
    python run.py                    # full run on Qwen3-4B
    python run.py --smoke            # 10 steps on Qwen3-0.6B, 4 tasks (GPU smoke test)
    python run.py --baseline         # zero-shot judge score, no training
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace

import torch
from dotenv import load_dotenv

# Must run before reward.judge is touched: the judge reads OPENROUTER_API_KEY from the
# environment when it constructs its generator.
load_dotenv()

from config import CONFIG, Config  # noqa: E402
from data.dataset import build_prompt, load_tasks, split_tasks  # noqa: E402
from reward.judge import RubricJudge, summarize  # noqa: E402
from train.rollout import RolloutEngine  # noqa: E402
from train.store import TrajectoryStore  # noqa: E402
from train.trainer import SDPOTrainer, log_metrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sdpo")


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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = load_tasks(config.dataset_name, config.dataset_split)
    train_tasks, heldout_tasks = split_tasks(tasks, config.n_heldout, config.split_seed)
    if smoke:
        train_tasks, heldout_tasks = train_tasks[:4], heldout_tasks[:2]
    logger.info("loaded %d train / %d held-out tasks", len(train_tasks), len(heldout_tasks))

    model_name = config.smoke_model if smoke else config.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, config.dtype)
    ).cuda()

    store = TrajectoryStore(config.store_capacity, config.max_staleness)
    engine = RolloutEngine(config, model=model_name)
    engine.start()

    trainer = SDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        store=store,
        config=config,
        tasks_by_id={t.task_id: t for t in tasks},
    )
    judge = RubricJudge(
        config.judge_model,
        config.judge_max_concurrency,
        config.judge_max_retries,
        config.judge_timeout,
    )

    # Weight-sync group: one rank per vLLM GPU WORKER (i.e. tensor_parallel_size ranks per
    # engine, not one per engine) plus rank 0 for the trainer as sender.
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
            if not batch:
                logger.warning("no non-stale trajectories available; rollout may be too slow")
                continue

            metrics = trainer.train_step(batch)
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
    finally:
        stop.set()
        rollout_task.cancel()
        await engine.shutdown()


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
                         eval_interval=5, n_rollout_gpus=1)

    if args.baseline:
        asyncio.run(baseline(config))
    else:
        asyncio.run(train(config, smoke=args.smoke))


if __name__ == "__main__":
    main()
