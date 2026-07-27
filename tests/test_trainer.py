"""Trainer tests on a tiny randomly-initialized model. CPU-only, no downloads.

The end-to-end test here is the one that proves the pieces actually compose: store ->
batch -> student/teacher forward -> loss -> optimizer step, with the loss decreasing.
"""

import asyncio

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from config import Config
from data.dataset import Task
from train.batch import build_batch
from train.store import Trajectory, TrajectoryStore
from train.trainer import SDPOTrainer, TrainerState, _slice_batch


class FakeTokenizer:
    """Minimal tokenizer: maps characters to ids. Avoids a tokenizer download."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [(ord(c) % 50) + 2 for c in text[:24]]}


def _model():
    config = AutoConfig.for_model(
        "llama", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config)


def _task(task_id="1") -> Task:
    # No hint on the Task -- hints are per-rollout and live on the Trajectory.
    return Task(task_id=task_id, query="Assess the funding base.", sections=[])


def _traj(task_id="1", version=0, n=4) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_token_ids=[5, 6, 7],
        response_token_ids=[10 + i for i in range(n)],
        rollout_logprobs=[-0.7] * n,
        policy_version=version,
        hint="consider deposit beta and repricing risk",
    )


def _trainer(config: Config | None = None) -> SDPOTrainer:
    config = config or Config(batch_size=4, mini_batch_size=2)
    return SDPOTrainer(
        model=_model(),
        tokenizer=FakeTokenizer(),
        store=TrajectoryStore(max_staleness=config.max_staleness),
        config=config,
        tasks_by_id={"1": _task("1"), "2": _task("2")},
        device="cpu",
    )


def test_train_step_runs_and_returns_metrics():
    trainer = _trainer()
    metrics = trainer.train_step([_traj(), _traj(task_id="2")])

    assert "loss" in metrics
    assert "teacher_minus_student_logp" in metrics
    assert metrics["step"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_train_step_actually_updates_weights():
    trainer = _trainer()
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train_step([_traj(), _traj(task_id="2")])
    after = list(trainer.model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_teacher_forward_produces_no_gradient():
    """The teacher is a stop-gradient target; grad must flow only through the student."""
    trainer = _trainer()
    batch = trainer.prepare_batch([_traj()]).to("cpu")
    max_r = batch.response_mask.size(1)

    teacher_lp = trainer._teacher_logprobs(batch, max_r)
    assert not teacher_lp.requires_grad

    student_lp = trainer._student_logprobs(batch, max_r)
    assert student_lp.requires_grad


def test_loss_decreases_over_repeated_steps():
    """The real integration check: repeatedly stepping on the same batch should pull the
    student toward the teacher, so the loss trends down.
    """
    trainer = _trainer(Config(batch_size=2, mini_batch_size=2, learning_rate=1e-3))
    batch = [_traj(), _traj(task_id="2")]

    first = trainer.train_step(batch)["loss"]
    for _ in range(12):
        last = trainer.train_step(batch)["loss"]

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_teacher_student_gap_is_reported_and_nonzero():
    """This is the metric that tells us whether the hint is doing anything at all."""
    trainer = _trainer()
    metrics = trainer.train_step([_traj()])
    assert metrics["teacher_minus_student_logp"] != 0.0


def test_mini_batching_matches_single_batch_gradient():
    """Gradient accumulation over mini-batches must equal the full-batch gradient."""
    trajectories = [_traj(), _traj(task_id="2"), _traj(), _traj(task_id="2")]

    single = _trainer(Config(batch_size=4, mini_batch_size=4))
    single.train_step(trajectories)
    single_params = [p.detach().clone() for p in single.model.parameters()]

    chunked = _trainer(Config(batch_size=4, mini_batch_size=2))
    chunked.train_step(trajectories)
    chunked_params = [p.detach().clone() for p in chunked.model.parameters()]

    for a, b in zip(single_params, chunked_params):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_empty_batch_is_a_noop():
    trainer = _trainer()
    assert trainer.train_step([]) == {}
    assert trainer.state.step == 0


def test_prepare_batch_uses_trajectory_hint():
    """The teacher conditions on the hint generated from THIS rollout's draft, which is
    carried on the trajectory -- there is no task- or config-level hint to fall back to."""
    trainer = _trainer()
    traj = _traj()
    traj.hint = "a much longer hint that tokenizes to more tokens than the default one"
    batch = trainer.prepare_batch([traj])
    # Teacher prompt is longer than the student prompt because the hint was prepended.
    assert batch.teacher_attention_mask.sum() > batch.student_attention_mask.sum()


def test_slice_batch_preserves_alignment():
    trajectories = [_traj(task_id="1"), _traj(task_id="2")]
    batch = build_batch(trajectories, [[1, 2], [1, 2, 3]], pad_token_id=0)
    sliced = _slice_batch(batch, 1, 2)

    assert sliced.task_ids == ["2"]
    assert sliced.student_input_ids.shape[0] == 1
    assert sliced.response_mask.shape[0] == 1


def test_policy_version_bump():
    trainer = _trainer()
    assert trainer.bump_policy_version() == 1
    assert trainer.bump_policy_version() == 2
    assert trainer.state.policy_version == 2


def test_checkpoint_roundtrip_preserves_ema_and_step():
    """The advantage EMA is training state -- losing it on resume silently re-tightens the
    clip bound exactly when training is least stable."""
    trainer = _trainer()
    trainer.train_step([_traj(), _traj(task_id="2")])
    trainer.bump_policy_version()
    saved = trainer.state_dict()

    restored = _trainer()
    restored.load_state_dict(saved)

    assert restored.state.step == trainer.state.step
    assert restored.state.policy_version == trainer.state.policy_version
    assert restored.state.clipper.ema == trainer.state.clipper.ema
    assert restored.state.clipper.step == trainer.state.clipper.step


def test_save_checkpoint_writes_loadable_state(tmp_path):
    """run.py's checkpoint file must round-trip through load_state_dict -- weights, step,
    and the EMA clipper. A RunPod interruption is survivable only if this works."""
    from run import save_checkpoint

    trainer = _trainer()
    trainer.train_step([_traj(), _traj(task_id="2")])
    save_checkpoint(trainer, str(tmp_path), 7)

    restored = _trainer()
    restored.load_state_dict(torch.load(tmp_path / "step_7" / "state.pt"))

    assert restored.state.step == trainer.state.step
    assert restored.state.clipper.ema == trainer.state.clipper.ema
    for a, b in zip(restored.model.parameters(), trainer.model.parameters()):
        torch.testing.assert_close(a, b)


def test_trainer_state_roundtrip():
    state = TrainerState()
    state.step = 7
    state.policy_version = 3
    state.clipper.update(torch.full((1, 4), 2.0), torch.ones(1, 4))

    restored = TrainerState()
    restored.load_state_dict(state.state_dict())
    assert restored.step == 7 and restored.policy_version == 3
    assert restored.clipper.ema == state.clipper.ema


def test_weight_buckets_cover_every_parameter_exactly_once():
    """Every parameter must ship, exactly once, or the rollout policy silently diverges
    from the trainer's."""
    trainer = _trainer()
    expected = [name for name, _ in trainer.model.named_parameters()]

    shipped = []
    for names, dtype_names, shapes, tensors in trainer.weight_buckets():
        assert len(names) == len(dtype_names) == len(shapes) == len(tensors)
        shipped.extend(names)

    assert shipped == expected


def test_weight_buckets_cast_to_bfloat16():
    """The rollout engine runs bf16; sending fp32 would fail the dtype check or double the
    bytes on the wire."""
    trainer = _trainer()
    for names, dtype_names, shapes, tensors in trainer.weight_buckets():
        assert all(d == "bfloat16" for d in dtype_names)
        assert all(t.dtype == torch.bfloat16 for t in tensors)


def test_weight_bucket_shapes_match_tensors():
    """Receivers allocate from this metadata; a mismatch corrupts the transfer."""
    trainer = _trainer()
    for names, dtype_names, shapes, tensors in trainer.weight_buckets():
        for shape, tensor in zip(shapes, tensors):
            assert list(tensor.shape) == shape


def test_small_bucket_size_produces_multiple_buckets():
    trainer = _trainer(Config(batch_size=2, mini_batch_size=2, weight_sync_bucket_mb=0))
    assert len(list(trainer.weight_buckets())) > 1


def test_send_weight_bucket_requires_setup():
    trainer = _trainer()
    with pytest.raises(RuntimeError, match="weight sync not initialized"):
        trainer.send_weight_bucket(["w"], [torch.zeros(2)])


def test_store_to_trainer_integration():
    """Full path: rollouts land in the store, trainer samples within K=3 and steps."""
    async def run():
        config = Config(batch_size=4, mini_batch_size=2, max_staleness=3)
        trainer = _trainer(config)
        store = trainer.store

        for i in range(3):
            await store.add(_traj(task_id=str((i % 2) + 1), version=0))
        await store.add(_traj(task_id="1", version=-10))  # too stale, must be evicted

        await store.set_policy_version(0)
        batch = await store.sample(8)
        return trainer.train_step(batch), store, len(batch)

    metrics, store, n = asyncio.run(run())
    assert n == 3, "the over-stale trajectory should have been evicted"
    assert store.stats.evicted_stale == 1
    assert "loss" in metrics and torch.isfinite(torch.tensor(metrics["loss"]))
