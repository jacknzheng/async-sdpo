"""File + wandb run logging. CPU-only; writes under a tmp dir, never /log."""

from datetime import datetime
from pathlib import Path

from train.config import Config, to_yaml
from train.logger import make_run_name, setup_run_logging, wandb_run_config


def test_make_run_name_includes_dataset_hint_and_time():
    cfg = Config()
    name = make_run_name(cfg, now=datetime(2026, 8, 25, 12, 0, 0))
    assert name == "tau2-gold-20260825-120000"


def test_make_run_name_honours_explicit_override():
    cfg = Config.from_cli_overrides(["logging.run_name=my-run"])
    assert make_run_name(cfg, smoke=True, baseline=True) == "my-run"


def test_setup_writes_args_config_and_train_log(tmp_path: Path):
    cfg = Config.from_cli_overrides(
        [
            f"logging.log_dir={tmp_path}",
            "logging.run_name=unit-run",
            "logging.wandb_enabled=false",
        ]
    )
    argv = ["run.py", "--hint-prompt", "gold", "trainer.total_steps=50"]
    ctx = setup_run_logging(cfg, argv, rank=0)

    assert ctx.run_name == "unit-run"
    assert ctx.log_dir == tmp_path / "unit-run"
    args_text = (ctx.log_dir / "args.txt").read_text(encoding="utf-8")
    assert "--hint-prompt gold" in args_text
    assert "trainer.total_steps=50" in args_text
    assert "hint: gold" in args_text
    yaml_text = (ctx.log_dir / "config.yaml").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-8B" in yaml_text
    assert Config.from_dict_config(__import__("yaml").safe_load(yaml_text))
    train_log = ctx.log_dir / "train.log"
    assert train_log.exists()
    assert "unit-run" in train_log.read_text(encoding="utf-8")
    payload = wandb_run_config(cfg, ctx)
    assert payload["cli_args"] == argv
    assert payload["run_name"] == "unit-run"
    assert payload["data"]["dataset"] == "tau2"
