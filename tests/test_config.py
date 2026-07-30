"""Config validation tests.

The load-bearing one is the smoke-config test: dataclasses.replace() re-runs
__post_init__, and an over-strict GPU-split check there once made `python run.py --smoke`
crash before doing anything -- on the box, not on the laptop.
"""

from dataclasses import replace

import pytest

from config import CONFIG, Config


def test_smoke_replace_does_not_raise():
    smoke = replace(
        CONFIG, total_steps=10, batch_size=4, mini_batch_size=2, eval_interval=5,
        n_rollout_gpus=1, n_trainer_gpus=1
    )
    assert smoke.n_rollout_gpus == 1 and smoke.n_trainer_gpus == 1


def test_default_split_uses_the_whole_box():
    assert CONFIG.n_rollout_gpus + CONFIG.n_trainer_gpus == 8


def test_gpu_split_over_eight_rejected():
    with pytest.raises(ValueError, match="<= 8"):
        Config(n_rollout_gpus=8, n_trainer_gpus=4)


def test_zero_trainer_gpus_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        Config(n_rollout_gpus=4, n_trainer_gpus=0)


def test_clip_window_must_contain_one():
    with pytest.raises(ValueError, match="1.0"):
        Config(clip_ratio_low=1.2, clip_ratio_high=1.4)
