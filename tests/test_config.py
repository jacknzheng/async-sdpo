"""Config validation tests.

The load-bearing one is the smoke-config test: the smoke preset deliberately uses a subset
of the box (1 rollout + 1 trainer GPU), and an over-strict GPU-split check once made
`python run.py --smoke` crash before doing anything -- on the box, not on the laptop.

Overrides go through Config.from_cli_overrides rather than dataclasses.replace(), because
replace() is shallow and cannot reach a nested field.
"""

import pytest
import yaml

from train.config import Config, to_yaml

# Kept in sync with run.py's --smoke preset. If that list changes and this does not, the
# test below stops guarding the thing it exists to guard.
SMOKE_OVERRIDES = [
    "trainer.total_steps=10",
    "trainer.batch_size=4",
    "trainer.mini_batch_size=2",
    "trainer.n_trainer_gpus=1",
    "trainer.compile_trainer=false",
    "generator.engine.n_rollout_gpus=1",
    "generator.hint.backend=openrouter",
    "judge.eval_interval=5",
]


def test_smoke_overrides_do_not_raise():
    smoke = Config.from_cli_overrides(SMOKE_OVERRIDES)
    assert smoke.generator.engine.n_rollout_gpus == 1
    assert smoke.trainer.n_trainer_gpus == 1
    assert smoke.generator.hint.backend == "openrouter"
    assert smoke.reserved_gpus() == 2
    # A YAML "false" must land as a bool, not the string "false" (which is truthy).
    assert smoke.trainer.compile_trainer is False


def test_answer_free_uses_local_hint_engine():
    cfg = Config.from_cli_overrides(["generator.hint.prompt=answer_free"])
    assert cfg.uses_local_hint_engine()
    assert not Config().uses_local_hint_engine()


def test_default_split_uses_the_whole_box():
    cfg = Config()
    assert cfg.generator.engine.n_rollout_gpus == 4
    assert cfg.trainer.n_trainer_gpus == 4
    assert cfg.generator.hint.backend == "vllm"
    assert cfg.generator.hint.gpu == 8
    assert cfg.total_num_gpus == 9
    assert cfg.reserved_gpus() == 9


def test_vllm_hint_on_eight_gpus_rejected():
    with pytest.raises(ValueError, match="<= 8"):
        Config.from_cli_overrides(["total_num_gpus=8"])


def test_gpu_split_over_capacity_rejected():
    with pytest.raises(ValueError, match="<="):
        Config.from_cli_overrides(
            ["generator.engine.n_rollout_gpus=8", "trainer.n_trainer_gpus=4"]
        )


def test_hint_gpu_must_not_overlap_rollout_or_trainer():
    with pytest.raises(ValueError, match="overlaps"):
        Config.from_cli_overrides(["generator.hint.gpu=3"])


def test_unknown_hint_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        Config.from_cli_overrides(["generator.hint.backend=sglang"])


def test_zero_trainer_gpus_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        Config.from_cli_overrides(["trainer.n_trainer_gpus=0"])


def test_clip_window_must_contain_one():
    with pytest.raises(ValueError, match="1.0"):
        Config.from_cli_overrides(
            ["trainer.algorithm.clip_ratio_low=1.2", "trainer.algorithm.clip_ratio_high=1.4"]
        )


def test_default_rollout_backend_is_vllm():
    assert Config().generator.engine.backend == "vllm"


def test_unknown_rollout_backend_rejected():
    """A typo must fail at construction, not at the first weight sync 500 steps in."""
    with pytest.raises(ValueError, match="backend"):
        Config.from_cli_overrides(["generator.engine.backend=sglang-typo"])


# ---- validators that had no coverage before the nesting ----

def test_batch_size_must_divide_into_mini_batches():
    with pytest.raises(ValueError, match="divisible"):
        Config.from_cli_overrides(["trainer.batch_size=33", "trainer.mini_batch_size=16"])


def test_adv_ema_decay_must_be_a_fraction():
    with pytest.raises(ValueError, match="adv_ema_decay"):
        Config.from_cli_overrides(["trainer.algorithm.adv_ema_decay=1.5"])


def test_unknown_hint_prompt_rejected():
    with pytest.raises(ValueError, match="generator.hint.prompt"):
        Config.from_cli_overrides(["generator.hint.prompt=answer_maybe"])


def test_mixture_hint_prompt_accepted():
    cfg = Config.from_cli_overrides(["generator.hint.prompt=mixture"])
    assert cfg.generator.hint.prompt == "mixture"


def test_gold_and_step_hint_prompts_accepted():
    assert Config.from_cli_overrides(["generator.hint.prompt=gold"]).generator.hint.prompt == "gold"
    assert (
        Config.from_cli_overrides(["generator.hint.prompt=step_hint"]).generator.hint.prompt
        == "step_hint"
    )


def test_default_is_proven_tau2_8b_stack():
    cfg = Config()
    assert cfg.data.dataset == "tau2"
    assert cfg.model.model == "Qwen/Qwen3-8B"
    assert cfg.trainer.gradient_checkpointing is True
    assert cfg.trainer.mini_batch_size == 2
    assert cfg.trainer.batch_size == 16
    assert cfg.generator.engine.max_model_len == 16384
    assert cfg.generator.engine.disable_custom_all_reduce is True
    assert cfg.generator.hint.prompt == "gold"
    assert cfg.generator.hint.backend == "vllm"
    assert cfg.generator.hint.model == "Qwen/Qwen3.5-9B"
    assert cfg.generator.hint.gpu == 8
    assert cfg.generator.hint.max_tokens == 2048
    assert cfg.generator.hint.reasoning_enabled is False
    assert (
        cfg.data.user_llm
        == "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    )
    assert cfg.judge.model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert cfg.logging.log_dir == "/log"
    assert cfg.logging.resume_from is None
    assert cfg.logging.wandb_enabled is True
    assert cfg.trainer.algorithm.use_sod is True
    assert cfg.trainer.algorithm.sod_delta == 0.2
    assert cfg.data.search_mode == "fast"


def test_sod_eps_must_be_positive():
    with pytest.raises(ValueError, match="sod_eps"):
        Config.from_cli_overrides(["trainer.algorithm.sod_eps=0"])


def test_unknown_search_mode_rejected():
    with pytest.raises(ValueError, match="search_mode"):
        Config.from_cli_overrides(["data.search_mode=slow"])


def test_zero_hint_concurrency_rejected():
    with pytest.raises(ValueError, match="concurrency"):
        Config.from_cli_overrides(["generator.hint.concurrency=0"])


def test_zero_hint_retries_rejected():
    with pytest.raises(ValueError, match="max_retries"):
        Config.from_cli_overrides(["generator.hint.max_retries=0"])


def test_zero_hint_max_tokens_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        Config.from_cli_overrides(["generator.hint.max_tokens=0"])


def test_resume_checkpoint_accepts_latest_selector():
    cfg = Config.from_cli_overrides(["logging.resume_from=latest"])
    assert cfg.logging.resume_from == "latest"


# ---- the nesting machinery itself ----

def test_dotted_override_reaches_a_deeply_nested_field():
    cfg = Config.from_cli_overrides(["trainer.optimizer.learning_rate=1e-5"])
    assert cfg.trainer.optimizer.learning_rate == 1e-5


def test_unknown_key_is_rejected_rather_than_ignored():
    """Silently absorbing a typo'd key means the value never takes effect and nothing says so."""
    with pytest.raises(ValueError, match="no_such_field"):
        Config.from_cli_overrides(["trainer.no_such_field=1"])


def test_plus_prefix_rejected():
    with pytest.raises(ValueError, match=r"\+"):
        Config.from_cli_overrides(["+trainer.brand_new_field=1"])


def test_yaml_round_trip_is_lossless():
    """The resolved config is dumped into the run directory; it must rebuild exactly."""
    cfg = Config.from_cli_overrides(["trainer.optimizer.learning_rate=1e-5"])
    assert Config.from_dict_config(yaml.safe_load(to_yaml(cfg))) == cfg


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(Exception):
        cfg.trainer.batch_size = 999
