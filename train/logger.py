from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from reward.judge import RubricJudge, summarize
from train.backends.backend import InferenceEngine
from train.config import Config, to_yaml

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Where this process is writing logs, and the argv that launched it."""

    run_name: str
    log_dir: Path  # /log/<run_name>
    argv: list[str]
    cli: str


def make_run_name(
    config: Config,
    *,
    smoke: bool = False,
    baseline: bool = False,
    now: datetime | None = None,
) -> str:
    if config.logging.run_name:
        return config.logging.run_name
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    parts = [config.data.dataset, config.generator.hint.prompt]
    if smoke:
        parts.append("smoke")
    if baseline:
        parts.append("baseline")
    parts.append(ts)
    return "-".join(parts)


def resolve_log_root(preferred: str = "/log") -> Path:
    """Prefer `/log` (the 8xH100 box). Fall back if that path is not writable."""
    candidates = [preferred, "/workspace/log", "log"]
    # Don't retry the same path twice.
    seen: set[str] = set()
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return path
        except OSError:
            continue
    raise RuntimeError(
        f"could not create a log directory; tried {candidates}. "
        "mkdir /log or set logging.log_dir=..."
    )


def setup_run_logging(
    config: Config,
    argv: list[str] | None = None,
    *,
    rank: int = 0,
    smoke: bool = False,
    baseline: bool = False,
) -> RunContext:
    """Create `/log/<run_name>/`, dump argv + resolved config, attach file handlers.

    Rank 0 writes `train.log`; other FSDP ranks write `rank{N}.log` so a hang on
    rank 2 is still on disk. Stderr/stdout keep going to the console (and to
    whatever `tee` the launch script wraps around torchrun).
    """
    argv = list(argv if argv is not None else sys.argv)
    cli = " ".join(argv)
    run_name = make_run_name(config, smoke=smoke, baseline=baseline)
    root = resolve_log_root(config.logging.log_dir)
    log_dir = root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / "args.txt").write_text(
        "\n".join(
            [
                f"run_name: {run_name}",
                f"cli: {cli}",
                f"rank: {rank}",
                f"cwd: {os.getcwd()}",
                f"dataset: {config.data.dataset}",
                f"hint: {config.generator.hint.prompt}",
                f"model: {config.model.model}",
                f"gpus: {config.generator.engine.n_rollout_gpus} rollout + "
                f"{config.trainer.n_trainer_gpus} trainer",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (log_dir / "config.yaml").write_text(to_yaml(config), encoding="utf-8")

    if rank == 0:
        latest = root / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(log_dir.name)
        except OSError:
            pass

    log_file = log_dir / ("train.log" if rank == 0 else f"rank{rank}.log")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    try:
        from loguru import logger as loguru_logger

        loguru_logger.add(log_file, enqueue=True, backtrace=False, diagnose=False)
    except Exception:
        pass

    logging.getLogger("sdpo").info("run %s | %s", run_name, cli)
    logging.getLogger("sdpo").info("file logs: %s", log_dir)
    return RunContext(run_name=run_name, log_dir=log_dir, argv=argv, cli=cli)


def wandb_run_config(config: Config, ctx: RunContext) -> dict:
    """What wandb stores so two runs with different CLI args are distinguishable."""
    payload = asdict(config)
    payload["cli"] = ctx.cli
    payload["cli_args"] = ctx.argv
    payload["run_name"] = ctx.run_name
    payload["log_dir"] = str(ctx.log_dir)
    return payload


def init_wandb(config: Config, ctx: RunContext):
    """Start a wandb run tagged with the full CLI + resolved config. None if disabled."""
    if not config.logging.wandb_enabled:
        return None
    import wandb

    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        logging.getLogger("sdpo").warning(
            "WANDB_API_KEY is unset; wandb will run offline. "
            "export WANDB_API_KEY=... to log to the cloud."
        )
        os.environ["WANDB_MODE"] = "offline"

    settings = {"project": config.logging.wandb_project, "name": ctx.run_name}
    if config.logging.wandb_entity:
        settings["entity"] = config.logging.wandb_entity
    try:
        run = wandb.init(
            **settings,
            config=wandb_run_config(config, ctx),
            dir=str(ctx.log_dir),
            notes=ctx.cli,
        )
        # Eval runs asynchronously and may finish after training has advanced.
        # Give eval its own x-axis instead of attempting an out-of-order global step.
        run.define_metric("eval/launched_at_step")
        run.define_metric("eval/*", step_metric="eval/launched_at_step")
    except Exception:
        logging.getLogger("sdpo").exception(
            "wandb.init failed; continuing with file logs only"
        )
        return None
    logging.getLogger("sdpo").info("wandb run: %s (%s)", ctx.run_name, wandb.run.url if wandb.run else "offline")
    return run


async def evaluate(
    engine: InferenceEngine,
    judge: RubricJudge,
    tasks,
    max_concurrency: int = 8,
) -> dict[str, float]:
    """Generate on held-out tasks (with web search) and score with the rubric judge.

    Held-out only: these tasks are never trained on, so the gap between this score and
    train-set performance is the overfitting signal.
    """
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def one(task):
        async with sem:
            return await engine.generate_tir(task)

    results = await asyncio.gather(*(one(task) for task in tasks))
    scored = await judge.score_all(
        [(task, result.text) for task, result in zip(tasks, results)]
    )
    return summarize(scored)


async def evaluate_pass1(
    engine: InferenceEngine, tasks, max_concurrency: int = 8
) -> dict[str, float]:
    """Held-out tau2 pass^1 overall and per domain."""
    sem = asyncio.Semaphore(max_concurrency)

    async def one(task):
        async with sem:
            return task, await engine.generate_tir(task)

    pairs = await asyncio.gather(*(one(task) for task in tasks))
    overall: list[float] = []
    by_domain: dict[str, list[float]] = defaultdict(list)
    for task, result in pairs:
        score = float(result.judge_score or 0.0)
        overall.append(score)
        domain = getattr(task, "domain", None) or "unknown"
        by_domain[domain].append(score)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "pass1": _mean(overall),
        "n": float(len(overall)),
    }
    for domain, scores in sorted(by_domain.items()):
        metrics[f"pass1_{domain}"] = _mean(scores)
        metrics[f"n_{domain}"] = float(len(scores))
    return metrics


async def evaluate_and_log(
    engine: InferenceEngine,
    judge: RubricJudge | None,
    tasks,
    step: int,
    policy_version: int,
    dataset: str = "diligence",
    max_concurrency: int = 8,
) -> None:
    """Fire-and-forget wrapper: eval, then wandb.log against the step that *launched* it."""
    try:
        if dataset == "tau2" or judge is None:
            metrics = await evaluate_pass1(engine, tasks, max_concurrency=max_concurrency)
        else:
            metrics = await evaluate(
                engine, judge, tasks, max_concurrency=max_concurrency
            )
        logger.info("EVAL launched at step %d (policy %d): %s", step, policy_version, metrics)
        payload = {f"eval/{k}": v for k, v in metrics.items()}
        payload["eval/launched_at_step"] = float(step)
        payload["eval/policy_version"] = float(policy_version)
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(payload)
    except Exception:
        logger.exception("eval failed (launched at step %d)", step)
