from __future__ import annotations

from datetime import timedelta
import asyncio
import logging
import os
import sys
from pathlib import Path

# huggingface_hub snapshots this at import time. Anonymous or cold model pulls on
# workstation disks regularly exceed its short default read timeout.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

def _prepend_vllm_cudart() -> None:
    """vLLM 0.26's PyPI wheel links libcudart.so.13; this box's torch is cu128.

    Without this, `import vllm` fails with `libcudart.so.13: cannot open shared
    object file`. Changing LD_LIBRARY_PATH after the process has started does
    not always affect dlopen, so we also preload the .so by absolute path.
    """
    import ctypes

    for lib in Path(sys.prefix).glob("lib/python*/site-packages/nvidia/cu13/lib"):
        if not lib.is_dir():
            continue
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        prefix = str(lib)
        if prefix not in cur.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{prefix}{os.pathsep}{cur}" if cur else prefix
            )
        so = lib / "libcudart.so.13"
        if so.exists():
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        break


_prepend_vllm_cudart()

from train.store import Trajectory
import argparse

from data.dataset import Task, TaskDataset
from data.hint import generate_hint
from data.tau_harness import configure_embeddings_cache

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

# Must run before reward.judge is touched: the judge reads OPENROUTER_API_KEY from the
# environment when it constructs its generator.
load_dotenv()

# tau2's UserSimulator talks litellm. If OPENAI_API_KEY is unset it collapses the
# openrouter/ model slug onto the openai provider and tool-calling breaks. Point
# both at OpenRouter when only the OR key is present.
if os.environ.get("OPENROUTER_API_KEY"):
    os.environ.setdefault("OPENAI_API_KEY", os.environ["OPENROUTER_API_KEY"])
    os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# NCCL on some workstation images (Baseten) livelocks P2P/CUMEM during weight sync.
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# tau2 reads TAU2_DATA_DIR at import. The pip package does not ship `data/`, so we
# keep a sparse clone of the pinned tau2-bench commit next to the repo.
_tau2_data = Path(__file__).resolve().parent / ".deps" / "tau2-bench" / "data"
if _tau2_data.is_dir():
    os.environ.setdefault("TAU2_DATA_DIR", str(_tau2_data))

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
configure_embeddings_cache()
# Qwen3-8B pure-bf16 AdamW sits at ~65 of 80 GB per trainer rank (16 weights + 16 grads
# + 33 optimizer state); expandable segments lets the allocator grow in place instead of
# fragmenting. Drop this line if vLLM ever objects -- it is an aid, not a requirement.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from train.config import Config  # noqa: E402
from train.logger import (  # noqa: E402
    RunContext,
    evaluate,
    evaluate_and_log,
    evaluate_pass1,
    init_wandb,
    setup_run_logging,
)
from data.dataset import load_split  # noqa: E402
from reward.judge import RubricJudge  # noqa: E402
from train.backends import get_backend  # noqa: E402
from train.backends.backend import InferenceEngine  # noqa: E402
from train.store import TrajectoryStore  # noqa: E402
from train.trainer import (  # noqa: E402
    AsyncDataLoader,
    AsyncStalenessManager,
    SDPOTrainer,
    build_dataloader,
    log_metrics,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("sdpo")

def _dist_info():
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

def build_trainer(
    config: Config,
    smoke: bool,
    tasks,
    transport=None,
) -> SDPOTrainer:
    """
    Start trainer process group - device mesh, labels each GPU with cuda:2..5 for the trainer GPUs
    - implictly sets up weight sync abilites between the GPUs and cross-trainer GPU comms 
    - each is a SDPOTrainer worker
    """
    

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

        # Shard each transformer block as its own FSDP unit. Do NOT fully_shard the
        # CausalLM root: `_response_logprobs` calls `unwrapped.model` (the inner
        # backbone) and `lm_head` as submodules so the packed LM-head path never
        # materializes [B, T, V] logits. If the root owns a sharded embed_tokens,
        # that lookup is `F.embedding(plain input_ids, DTensor weight)` and dies
        # with "aten.embedding.default got mixed torch.Tensor and DTensor" -- the
        # 4+4 first-step crash on Baseten. Layers are the 27B; embed + lm_head
        # (~1.5 GB bf16, often tied) stay replicated, which is cheap.
        for layer in model.model.layers:
            fully_shard(layer, **fsdp_kwargs)

    trainer = SDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        tasks_by_id={t.task_id: t for t in tasks},
        device=device,
        transport=transport,
    )
    if config.trainer.compile_trainer:
        trainer._response_logprobs = torch.compile(trainer._response_logprobs, dynamic=True)

    return trainer


def save_checkpoint(trainer: SDPOTrainer, output_dir: str, tag) -> None:
    """Write model + optimizer + trainer state (incl. the EMA clipper) to output_dir.

    Called by EVERY rank. Under FSDP2 `trainer.state_dict()` gathers sharded DTensors,
    which is a collective -- every rank must enter it, but only rank 0 comes out holding
    populated tensors, so only rank 0 writes the file.
    """
    state = trainer.state_dict()
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    path = Path(output_dir) / f"step_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "state.pt")
    logger.info("checkpoint saved: %s", path / "state.pt")


async def sync_weights(
    trainer: SDPOTrainer,
    engine: InferenceEngine | None,
    store,
    new_version: int,
    rank: int = 0,
    world: int = 1,
) -> None:
    """Push the trainer's weights into the rollout engine, then advance the version.

    Every trainer rank walks `weight_buckets()` because `.full_tensor()` is a collective
    over the FSDP mesh. Only rank 0 has an engine/store and actually sends; other ranks
    gather and discard.

    The four-call transaction, per bucket (rank 0):

        pause_generation(mode="keep") + start_weight_update()
          -> for each bucket: post receive RPC, THEN trainer_send_weights, then join
        finish_weight_update() + reset_prefix_cache() + resume_generation()

    ORDERING: the receive-side `update_weights` RPC must be in flight before the trainer
    enters `trainer_send_weights`, or the sender blocks on a broadcast nobody is listening
    for. We create the receive task, yield once so it dispatches, run the (blocking,
    synchronous) send in a thread so the event loop stays free, then join.
    """
    if rank == 0:
        await engine.pause_for_update()

    for bucket in trainer.weight_buckets():
        if rank == 0:
            receive = asyncio.create_task(
                engine.receive_weight_bucket(bucket.names, bucket.dtype_names, bucket.shapes)
            )
            await asyncio.sleep(0)  # let the receive RPC dispatch before we send
            await asyncio.to_thread(trainer.send_weight_bucket, bucket)
            await asyncio.wait_for(
                receive, timeout=trainer.config.generator.engine.weight_sync_timeout_s
            )

    if rank == 0:
        await engine.finish_update(new_version)
        await store.set_policy_version(new_version)
        await store.prune_stale(staleness_manager=trainer.staleness_manager)


async def generate_trajectory(
    config,
    engine: InferenceEngine,
    store: TrajectoryStore,
    task: Task,
    version: int,
    staleness_manager: AsyncStalenessManager,
) -> None:
    """
    Produce one trajectory. Success is silent - get_batch() marks it accepted.
    Failures never reach the store, so they reject here to free the submission slot.
    """
    try:
        result = await engine.generate_tir(task)
    except Exception:
        logger.exception("rollout failed for task %s", task.task_id)
        await staleness_manager.on_rollout_rejected()
        return
    # generate the hint before adding to the store!
    hints = await generate_hint(config, task, result.text)
    if not hints.ok(config.generator.hint.prompt):
        cause = hints.cause or "other"
        store.stats.count_hint_drop(cause)
        logger.warning(
            "dropping rollout for task %s: no hint could be generated "
            "(cause=%s, detail=%s)",
            task.task_id,
            cause,
            hints.detail or "unavailable",
        )
        await staleness_manager.on_rollout_rejected()
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
            hint_free=hints.free,
            hint_bearing=hints.bearing,
            judge_score=result.judge_score,
            step_spans=list(result.step_spans),
            loss_mask=list(result.loss_mask),
        )
    )

async def run_loop(
    config: Config,
    engine,
    store: TrajectoryStore,
    dataloader,
    stop_event: asyncio.Event,
    staleness_manager: AsyncStalenessManager,
) -> None:
    """
    PRODUCER LOOP

    Continuously sample tasks, generate, and push trajectories into the store.
    This continues until stop_event (asyncio.Event)
    """
    dataloader = AsyncDataLoader(dataloader, mini_batch_size=config.trainer.mini_batch_size)

    pending_trajectories: set[asyncio.Task] = set()

    while not stop_event.is_set():
        await staleness_manager.acquire_submission_slot()

        row = await dataloader.get_next_non_consumed_data()
        if row is None:
            await dataloader.reset_at_epoch_end()
            row = await dataloader.get_next_non_consumed_data()
            if row is None:
                logger.warning("train dataloader is empty; cannot produce more rollouts")
                await staleness_manager.on_rollout_rejected()
                break

        task: Task = row[0]["task"]
        t = asyncio.create_task(
            generate_trajectory(
                config, engine, store, task, engine.policy_version, staleness_manager
            )
        )
        pending_trajectories.add(t)
        t.add_done_callback(pending_trajectories.discard)

    if pending_trajectories:
        await asyncio.gather(*pending_trajectories, return_exceptions=True)

def _pick_weight_sync_port(host: str, preferred: int) -> int:
    """Bind `preferred`, or an ephemeral port if a previous run still holds it.

    A stuck EngineCore / weight-sync process on 51216 is what killed the answer_bearing
    arm (`Address already in use`). Both sides of the rendezvous run in this process, so
    they share whatever we return. The probe socket is closed before NCCL binds.
    """
    import socket

    for port in (preferred, 0):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # No SO_REUSEADDR: a leftover listener on 51216 must fail this probe,
            # not look free and then explode inside NCCL.
            sock.bind((host, port))
            chosen = int(sock.getsockname()[1])
        except OSError:
            continue
        finally:
            sock.close()
        if port != preferred:
            logger.warning(
                "weight-sync port %d in use; using %d instead", preferred, chosen
            )
        return chosen
    raise RuntimeError(f"could not bind a weight-sync port on {host}")


def _init_trainer_process_group(config: Config) -> None:
    """Join the FSDP group. Rank 0 must call this AFTER the rollout engine has spawned.

    torchrun already sized WORLD_SIZE to the trainer ranks. The rollout GPUs never join.
    """
    rank, world = _dist_info()
    if world <= 1 or (dist.is_available() and dist.is_initialized()):
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(config.generator.engine.n_rollout_gpus + local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    logger.info("trainer process group ready: rank %d / %d", rank, world)


async def train(config: Config, smoke: bool = False, ctx: RunContext | None = None) -> None:
    """Every trainer rank runs this. Rank 0 also owns rollout + eval."""
    rank, world = _dist_info()
    wandb = None
    if rank == 0 and ctx is not None:
        wandb = init_wandb(config, ctx)

    train_tasks, heldout_tasks = load_split(config.data)
    if smoke:
        train_tasks, heldout_tasks = train_tasks[:4], heldout_tasks[:2]
    if rank == 0:
        logger.info("loaded %d train / %d held-out tasks", len(train_tasks), len(heldout_tasks))

    engine_cls, transport_cls = get_backend(config.generator.engine.backend)

    # Rank 0 starts vLLM BEFORE the trainer process group exists. The engine's TP
    # workers are spawned from this process; if they inherit torchrun's rendezvous
    # (or a live default process group) they hang on TCPStore instead of forming
    # their own. Other ranks skip this and wait at init_process_group.
    engine = None
    store = None
    judge = None
    stop = None
    rollout_task = None
    eval_task = None

    if rank == 0:
        store = TrajectoryStore(
            config.generator.engine.store_capacity,
            config.trainer.algorithm.max_staleness,
        )
        engine = engine_cls(
            config, model=config.model.smoke_model if smoke else config.model.model
        )
        engine.start()
        if config.data.dataset == "diligence":
            judge = RubricJudge(
                config.judge.model,
                config.judge.max_concurrency,
                config.judge.max_retries,
                config.judge.timeout,
            )

    _init_trainer_process_group(config)

    # Only rank 0 pushes weights into vLLM.
    trainer = build_trainer(
        config, smoke, train_tasks + heldout_tasks, transport=transport_cls() if rank == 0 else None
    )

    if rank == 0:
        weight_world_size = config.generator.engine.n_rollout_gpus + 1
        sync_host = config.generator.engine.weight_sync_host
        sync_port = _pick_weight_sync_port(sync_host, config.generator.engine.weight_sync_port)
        await asyncio.gather(
            engine.init_weight_update_group(
                master_address=sync_host,
                master_port=sync_port,
                rank_offset=1,
                world_size=weight_world_size,
            ),
            asyncio.to_thread(
                trainer.setup_weight_sync,
                sync_host,
                sync_port,
                weight_world_size,
            ),
        )
        stop = asyncio.Event()
        train_loader = build_dataloader(
            config, TaskDataset(train_tasks), is_train=True, is_fully_async=True
        )
        rollout_task = asyncio.create_task(
            run_loop(
                config, engine, store, train_loader, stop, trainer.staleness_manager
            )
        )

    try:
        while trainer.state.step < config.trainer.total_steps:
            payload = [None]
            if rank == 0:
                payload[0] = await store.get_batch(
                    config.trainer.batch_size,
                    staleness_manager=trainer.staleness_manager,
                )
            if world > 1:
                dist.broadcast_object_list(payload, src=0)
            batch = payload[0]

            if len(batch) < world:
                if rank == 0:
                    logger.warning(
                        "no non-stale trajectories available; rollout may be too slow"
                    )
                continue

            metrics = trainer.train_step(batch[rank::world])

            version = trainer.bump_policy_version()
            await sync_weights(trainer, engine, store, version, rank=rank, world=world)

            if rank == 0:
                await trainer.staleness_manager.notify_capacity_change(
                    trainer.state.step + 1
                )
                if trainer.state.step % config.logging.log_interval == 0:
                    log_metrics(metrics, store)
                    if wandb is not None:
                        wandb.log(
                            {**metrics, **store.metrics()}, step=trainer.state.step
                        )
                if trainer.state.step % config.judge.eval_interval == 0:
                    if eval_task is not None and not eval_task.done():
                        logger.warning(
                            "skipping eval at step %d: previous still running",
                            trainer.state.step,
                        )
                    else:
                        launched_step = trainer.state.step
                        launched_version = trainer.state.policy_version
                        eval_task = asyncio.create_task(
                            evaluate_and_log(
                                engine,
                                judge,
                                heldout_tasks,
                                launched_step,
                                launched_version,
                                dataset=config.data.dataset,
                                max_concurrency=config.judge.max_concurrency,
                            )
                        )
                        eval_task.add_done_callback(
                            lambda t: (
                                t.exception()
                                if not t.cancelled() and t.exception()
                                else None
                            )
                        )

            # All ranks enter: FSDP gather inside state_dict(); rank 0 writes.
            if trainer.state.step % config.logging.checkpoint_interval == 0:
                save_checkpoint(
                    trainer, config.logging.output_dir, trainer.state.step
                )

        save_checkpoint(trainer, config.logging.output_dir, "final")
        if rank == 0 and eval_task is not None and not eval_task.done():
            await eval_task
    finally:
        if rank == 0:
            if stop is not None:
                stop.set()
            if rollout_task is not None:
                rollout_task.cancel()
            if eval_task is not None and not eval_task.done():
                eval_task.cancel()
            if engine is not None:
                await engine.shutdown()
            if wandb is not None:
                wandb.finish()

async def baseline(config: Config, ctx: RunContext | None = None) -> None:
    """Zero-shot held-out score, before any training."""
    wandb = None
    if ctx is not None:
        wandb = init_wandb(config, ctx)
    _, heldout = load_split(config.data)
    engine_cls, _ = get_backend(config.generator.engine.backend)
    engine = engine_cls(config)
    engine.start()
    try:
        if config.data.dataset == "tau2":
            metrics = await evaluate_pass1(
                engine, heldout, max_concurrency=config.judge.max_concurrency
            )
        else:
            judge = RubricJudge(
                config.judge.model,
                config.judge.max_concurrency,
                config.judge.max_retries,
                config.judge.timeout,
            )
            metrics = await evaluate(
                engine, judge, heldout, max_concurrency=config.judge.max_concurrency
            )
        logger.info("ZERO-SHOT BASELINE: %s", metrics)
        if wandb is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=0)
            wandb.summary.update(metrics)
    finally:
        await engine.shutdown()
        if wandb is not None:
            wandb.finish()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Off-policy SDPO on tau2 (banking_knowledge / retail / airline)",
        epilog="Any trailing dotted args override config, e.g. trainer.optimizer.learning_rate=1e-5",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny model, 4 tasks, 10 steps")
    parser.add_argument("--baseline", action="store_true", help="zero-shot eval, no training")
    parser.add_argument(
        "--hint-prompt",
        choices=("answer_free", "answer_bearing", "mixture", "gold", "step_hint"),
        default=None,
        help="teacher hint: gold dump, step_hint LLM, or diligence answer_free/bearing/mixture",
    )
    args, overrides = parser.parse_known_args()
    dotlist = []
    if args.smoke:
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
    ctx = setup_run_logging(
        config, sys.argv, rank=rank, smoke=args.smoke, baseline=args.baseline
    )
    # Rank 0, before NCCL: `which bwrap` is a false green on GPU pods that
    # seccomp-block unshare. Fail here so we do not spend an hour generating
    # zero-reward banking episodes.
    if rank == 0 and config.data.dataset == "tau2":
        from data.tau_harness import SandboxNamespaceError, assert_sandbox_ready

        try:
            assert_sandbox_ready(config.data.domains)
        except SandboxNamespaceError as exc:
            raise SystemExit(f"tau2 sandbox not usable:\n{exc}") from exc
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
        # Do NOT init_process_group here. Rank 0 has to spawn the vLLM engine first
        # (see train()), or TP workers inherit this rendezvous and hang on TCPStore.
        # Ranks 1..N wait inside train() at _init_trainer_process_group.
    elif config.trainer.n_trainer_gpus > 1 and not args.baseline:
        logger.warning(
            "launched as a single process with n_trainer_gpus=%d -- only one trainer GPU "
            "will be used; launch with torchrun --nproc-per-node=%d run.py to use them all",
            config.trainer.n_trainer_gpus, config.trainer.n_trainer_gpus,
        )
    try:
        if args.baseline:
            if rank == 0:
                asyncio.run(baseline(config, ctx=ctx))
        else: 
            asyncio.run(train(config, smoke=args.smoke, ctx=ctx))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()