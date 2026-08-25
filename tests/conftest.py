"""Shared test helpers.

The config is nested (see train/config.py), so tests build it through dotted overrides
rather than flat kwargs. `make_config` keeps call sites to one line while still going
through the real construction path -- including every group's __post_init__ validation,
which is the whole point of the config being typed.
"""

from train.config import Config


def make_config(**overrides) -> Config:
    """Build a Config from flat kwargs, mapped to their dotted paths.

        make_config(batch_size=4, mini_batch_size=2)

    Unknown names raise here rather than being silently ignored, so a test that renames a
    field fails loudly instead of quietly testing the default.
    """
    dotted = {
        # trainer
        "batch_size": "trainer.batch_size",
        "mini_batch_size": "trainer.mini_batch_size",
        "total_steps": "trainer.total_steps",
        "n_trainer_gpus": "trainer.n_trainer_gpus",
        "compile_trainer": "trainer.compile_trainer",
        # trainer.optimizer
        "learning_rate": "trainer.optimizer.learning_rate",
        "weight_decay": "trainer.optimizer.weight_decay",
        "max_grad_norm": "trainer.optimizer.max_grad_norm",
        # trainer.algorithm
        "max_staleness": "trainer.algorithm.max_staleness",
        "clip_ratio_low": "trainer.algorithm.clip_ratio_low",
        "clip_ratio_high": "trainer.algorithm.clip_ratio_high",
        "adv_ema_decay": "trainer.algorithm.adv_ema_decay",
        "use_sod": "trainer.algorithm.use_sod",
        # generator.engine
        "n_rollout_gpus": "generator.engine.n_rollout_gpus",
        "rollout_backend": "generator.engine.backend",
        "store_capacity": "generator.engine.store_capacity",
        "weight_sync_bucket_mb": "generator.engine.weight_sync_bucket_mb",
        # generator.hint
        "error_hint_prompt": "generator.hint.prompt",
        "error_hint_concurrency": "generator.hint.concurrency",
        # judge
        "eval_interval": "judge.eval_interval",
        # data
        "dataset": "data.dataset",
    }
    unknown = set(overrides) - set(dotted)
    if unknown:
        raise KeyError(f"make_config has no dotted path for {unknown}; add it to conftest.")
    # CPU unit tests are single-process; default the trainer shard count so a batch_size
    # of 2 is valid without every call site repeating n_trainer_gpus=1.
    if "n_trainer_gpus" not in overrides:
        overrides = {**overrides, "n_trainer_gpus": 1}
    return Config.from_cli_overrides([f"{dotted[k]}={v}" for k, v in overrides.items()])
