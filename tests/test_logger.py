"""File + wandb run logging. CPU-only; writes under a tmp dir, never /log."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from data.diagnostics import ARTIFACT_FILES, artifact_event
from train.config import Config, to_yaml
from train.logger import (
    evaluate_and_log,
    log_rollout_samples,
    make_run_name,
    rollout_sample_rows,
    setup_run_logging,
    wandb_run_config,
)
from train.models import Trajectory


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
    manifest = (ctx.log_dir / "ARTIFACTS.txt").read_text(encoding="utf-8")
    assert "console.log" in manifest
    assert "rollouts.jsonl" in manifest
    assert "sandbox.jsonl" in manifest
    assert "evaluations.jsonl" in manifest
    train_log = ctx.log_dir / "train.log"
    assert train_log.exists()
    assert "unit-run" in train_log.read_text(encoding="utf-8")
    for filename in ARTIFACT_FILES.values():
        assert (ctx.log_dir / filename).exists()
    artifact_event("api_failures", "test_failure", provider="test", status_code=503)
    failure = json.loads(
        (ctx.log_dir / "api_failures.jsonl").read_text(encoding="utf-8")
    )
    assert failure["event"] == "test_failure"
    assert failure["provider"] == "test"
    assert failure["status_code"] == 503
    payload = wandb_run_config(cfg, ctx)
    assert payload["cli_args"] == argv
    assert payload["run_name"] == "unit-run"
    assert payload["data"]["dataset"] == "tau2"


def test_pass1_records_one_failed_task_without_aborting_eval(monkeypatch):
    tasks = [
        SimpleNamespace(task_id="ok", domain="retail"),
        SimpleNamespace(task_id="bad", domain="airline"),
    ]

    class Engine:
        async def generate_tir(self, task):
            if task.task_id == "bad":
                raise RuntimeError("user simulator unavailable")
            return SimpleNamespace(
                judge_score=1.0,
                text="successful tau2 transcript",
                prompt_token_ids=[1, 2],
                response_token_ids=[3],
            )

    events = []
    monkeypatch.setattr(
        "train.logger.artifact_event",
        lambda channel, event, **fields: events.append((channel, event, fields)),
    )
    metrics = asyncio.run(
        evaluate_and_log(
            Engine(),
            None,
            tasks,
            step=25,
            policy_version=25,
            dataset="tau2",
            max_concurrency=2,
        )
    )
    assert metrics is not None
    assert metrics["pass1"] == 1.0
    assert metrics["n"] == 1.0
    assert metrics["eval_requested"] == 2.0
    assert metrics["eval_rollout_errors"] == 1.0
    eval_events = [
        (event, fields)
        for channel, event, fields in events
        if channel == "evaluations"
    ]
    assert [event for event, _ in eval_events] == [
        "evaluation_started",
        "evaluation_task_completed",
        "evaluation_task_failed",
        "evaluation_completed",
    ]
    completed = eval_events[1][1]
    assert completed["launched_at_step"] == 25
    assert completed["policy_version"] == 25
    assert completed["task_id"] == "ok"
    assert completed["pass1"] == 1.0
    assert completed["response_text"] == "successful tau2 transcript"


def test_diligence_eval_records_response_and_judge_score(monkeypatch):
    task = SimpleNamespace(task_id="d1", domain=None, query="Assess revenue.")

    class Engine:
        async def generate_tir(self, _task):
            return SimpleNamespace(
                text="Revenue increased.",
                prompt_token_ids=[1],
                response_token_ids=[2, 3],
            )

    class Judge:
        async def score_all(self, _pairs):
            return [
                SimpleNamespace(
                    score=0.75,
                    raw_score=3.0,
                    sections={},
                    error=None,
                )
            ]

    events = []
    monkeypatch.setattr(
        "train.logger.artifact_event",
        lambda channel, event, **fields: events.append((channel, event, fields)),
    )
    metrics = asyncio.run(
        evaluate_and_log(
            Engine(),
            Judge(),
            [task],
            step=50,
            policy_version=50,
            dataset="diligence",
            max_concurrency=1,
        )
    )

    assert metrics["judge_score"] == 0.75
    task_event = next(
        fields
        for channel, event, fields in events
        if (channel, event) == ("evaluations", "evaluation_task_completed")
    )
    assert task_event["launched_at_step"] == 50
    assert task_event["response_text"] == "Revenue increased."
    assert task_event["score"] == 0.75


def test_rollout_sample_rows_include_prompt_hint_and_output():
    batch = [
        Trajectory(
            task_id="t1",
            prompt_token_ids=[1],
            response_token_ids=[2],
            rollout_logprobs=[0.0],
            policy_version=1,
            hint_free="do not cite the figure",
            hint_bearing="the figure is 12",
            query="Assess the funding base.",
            response_text="Deposits grew.",
        )
    ]
    assert rollout_sample_rows(batch) == [
        [
            "t1",
            "Assess the funding base.",
            "Deposits grew.",
            "do not cite the figure",
            "the figure is 12",
        ]
    ]


def test_log_rollout_samples_is_a_noop_without_wandb():
    log_rollout_samples(None, [], step=1)
