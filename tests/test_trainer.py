"""Trainer tests on a tiny randomly-initialized model. CPU-only, no downloads.

The end-to-end test here is the one that proves the pieces actually compose: store ->
batch -> student/teacher forward -> loss -> optimizer step, with the loss decreasing.
"""

import asyncio

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from conftest import make_config
from train.config import Config
from data.dataset import Task
from train.utils import build_batch
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
        hint_free="consider deposit beta and repricing risk",
    )


def _trainer(config: Config | None = None) -> SDPOTrainer:
    config = config or make_config(batch_size=4, mini_batch_size=2)
    return SDPOTrainer(
        model=_model(),
        tokenizer=FakeTokenizer(),
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

    teacher_lp, _, _ = trainer._teacher_logprobs(batch, max_r)
    assert not teacher_lp.requires_grad

    student_lp = trainer._student_logprobs(batch, max_r)
    assert student_lp.requires_grad


def test_loss_decreases_over_repeated_steps():
    """The real integration check: repeatedly stepping on the same batch should pull the
    student toward the teacher, so the loss trends down.
    """
    trainer = _trainer(make_config(batch_size=2, mini_batch_size=2, learning_rate=1e-3))
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

    single = _trainer(make_config(batch_size=4, mini_batch_size=4))
    single.train_step(trajectories)
    single_params = [p.detach().clone() for p in single.model.parameters()]

    chunked = _trainer(make_config(batch_size=4, mini_batch_size=2))
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
    traj.hint_free = "a much longer hint that tokenizes to more tokens than the default one"
    batch = trainer.prepare_batch([traj])
    # Teacher prompt is longer than the student prompt because the hint was appended.
    assert batch.teacher_attention_mask.sum() > batch.student_attention_mask.sum()


def test_slice_batch_preserves_alignment():
    trajectories = [_traj(task_id="1"), _traj(task_id="2")]
    batch = build_batch(trajectories, [[1, 2], [1, 2, 3]], pad_token_id=0)
    sliced = _slice_batch(batch, 1, 2)

    assert sliced.task_ids == ["2"]
    assert sliced.student_input_ids.shape[0] == 1
    assert sliced.response_mask.shape[0] == 1
    assert sliced.step_ids is not None
    assert sliced.step_ids.shape[0] == 1
    torch.testing.assert_close(sliced.step_ids[0], batch.step_ids[1])


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
    for bucket in trainer.weight_buckets():
        assert len(bucket.names) == len(bucket.dtype_names) == len(bucket.shapes) == len(
            bucket.tensors
        )
        shipped.extend(bucket.names)

    assert shipped == expected


def test_weight_buckets_cast_to_bfloat16():
    """The rollout engine runs bf16; sending fp32 would fail the dtype check or double the
    bytes on the wire."""
    trainer = _trainer()
    for bucket in trainer.weight_buckets():
        assert all(d == "bfloat16" for d in bucket.dtype_names)
        assert all(t.dtype == torch.bfloat16 for t in bucket.tensors)


def test_weight_bucket_shapes_match_tensors():
    """Receivers allocate from this metadata; a mismatch corrupts the transfer."""
    trainer = _trainer()
    for bucket in trainer.weight_buckets():
        for shape, tensor in zip(bucket.shapes, bucket.tensors):
            assert list(tensor.shape) == shape


def test_small_bucket_size_produces_multiple_buckets():
    trainer = _trainer(make_config(batch_size=2, mini_batch_size=2, weight_sync_bucket_mb=0))
    assert len(list(trainer.weight_buckets())) > 1


def test_weight_bucket_names_survive_torch_compile():
    """torch.compile wraps the model in an OptimizedModule whose parameters are named
    `_orig_mod.*`; `unwrapped` must peel that or vLLM's receiver rejects every weight.
    Wrapper creation is lazy (no trace until a forward), so this is cheap on CPU."""
    trainer = SDPOTrainer(
        model=torch.compile(_model()),
        tokenizer=FakeTokenizer(),
        config=make_config(batch_size=4, mini_batch_size=2),
        tasks_by_id={"1": _task("1")},
        device="cpu",
    )
    shipped = [n for bucket in trainer.weight_buckets() for n in bucket.names]
    assert shipped and all(not n.startswith("_orig_mod.") for n in shipped)
    assert trainer._fsdp is None  # compiled but not sharded -> no FSDP handle


def test_unwrapped_peels_compile_wrapper_under_fsdp():
    """The full production wrapping is torch.compile(fully_shard(hf)). Parameter names
    must come out bare, and the FSDP root (used to detect sharding) must be found through
    the compile wrapper -- an isinstance check on the outer OptimizedModule misses it.

    Unlike DDP, fully_shard mutates in place rather than wrapping, so there is no
    `module.` level to peel; this pins that names stay clean for vLLM's receiver."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29517", world_size=1, rank=0
    )
    try:
        model = _model()
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        trainer = SDPOTrainer(
            model=torch.compile(model),
            tokenizer=FakeTokenizer(),
            config=make_config(batch_size=4, mini_batch_size=2),
            tasks_by_id={"1": _task("1")},
            device="cpu",
        )
        assert trainer._fsdp is model
        shipped = [n for bucket in trainer.weight_buckets() for n in bucket.names]
        assert shipped and all(
            not n.startswith(("_orig_mod.", "module.")) for n in shipped
        )
    finally:
        dist.destroy_process_group()


def test_weight_buckets_gather_sharded_params_to_full_shape():
    """Under FSDP2 every parameter is a DTensor holding only this rank's shard. If
    weight_buckets ships that raw, vLLM receives a FRAGMENT of each weight -- which does
    not crash, it silently generates garbage. Every shipped tensor must be full-shape."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29519", world_size=1, rank=0
    )
    try:
        model = _model()
        full_shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        assert any(isinstance(p.data, DTensor) for p in model.parameters())

        trainer = SDPOTrainer(
            model=model,
            tokenizer=FakeTokenizer(),
            config=make_config(batch_size=4, mini_batch_size=2),
            tasks_by_id={"1": _task("1")},
            device="cpu",
        )
        for bucket in trainer.weight_buckets():
            for name, tensor, shape in zip(bucket.names, bucket.tensors, bucket.shapes):
                assert not isinstance(tensor, DTensor), f"{name} shipped as a shard"
                assert tuple(tensor.shape) == full_shapes[name], name
                assert tuple(shape) == full_shapes[name], name
    finally:
        dist.destroy_process_group()


def test_send_weight_bucket_requires_setup():
    from train.backends.backend import WeightBucket

    trainer = _trainer()
    bucket = WeightBucket(
        names=["w"], dtype_names=["bfloat16"], shapes=[[2]], tensors=[torch.zeros(2)]
    )
    with pytest.raises(RuntimeError, match="weight sync not initialized"):
        trainer.send_weight_bucket(bucket)


def test_store_to_trainer_integration():
    """Full path: rollouts land in the store, trainer samples within K=3 and steps."""
    async def run():
        config = make_config(batch_size=4, mini_batch_size=2, max_staleness=3)
        trainer = _trainer(config)
        # The store is the producer's, not the trainer's -- run.py owns it on rank 0 and
        # hands train_step() a plain list of trajectories.
        store = TrajectoryStore(max_staleness=config.trainer.algorithm.max_staleness)

        await store.add(_traj(task_id="1", version=-10))  # too stale, must be evicted
        for i in range(3):
            await store.add(_traj(task_id=str((i % 2) + 1), version=0))

        await store.set_policy_version(0)
        batch = await store.get_batch(3)
        return trainer.train_step(batch), store, len(batch)

    metrics, store, n = asyncio.run(run())
    assert n == 3, "the over-stale trajectory should have been evicted"
    assert store.stats.evicted_stale == 1
    assert "loss" in metrics and torch.isfinite(torch.tensor(metrics["loss"]))


def test_mixture_prepare_batch_attaches_bearing_teacher():
    trainer = _trainer(make_config(batch_size=2, mini_batch_size=2, error_hint_prompt="mixture"))
    traj = _traj()
    traj.hint_free = "short"
    traj.hint_bearing = "a much longer answer-bearing hint with more tokens"
    batch = trainer.prepare_batch([traj])
    assert batch.teacher_bearing_input_ids is not None
    assert batch.teacher_bearing_input_ids.size(1) > batch.teacher_input_ids.size(1)


def test_mixture_teacher_is_mean_of_two_arms():
    """Averaging KLs == one SDPO pass on 0.5 (log t_b + log t_f), clipper off."""
    trainer = _trainer(make_config(batch_size=2, mini_batch_size=2, error_hint_prompt="mixture"))
    traj = _traj()
    traj.hint_free = "short"
    traj.hint_bearing = "a much longer answer-bearing hint with more tokens"
    batch = trainer.prepare_batch([traj]).to("cpu")
    max_r = batch.response_mask.size(1)

    student_lp = trainer._student_logprobs(batch, max_r)
    mix, lp_free, lp_bearing = trainer._teacher_logprobs(batch, max_r)
    assert lp_free is not None and lp_bearing is not None
    torch.testing.assert_close(mix, 0.5 * (lp_free + lp_bearing))

    from train.loss import sdpo_loss

    mixed_loss, _ = sdpo_loss(
        student_lp, mix, batch.rollout_logprobs, batch.response_mask, clipper=None
    )
    free_loss, _ = sdpo_loss(
        student_lp, lp_free, batch.rollout_logprobs, batch.response_mask, clipper=None
    )
    bearing_loss, _ = sdpo_loss(
        student_lp, lp_bearing, batch.rollout_logprobs, batch.response_mask, clipper=None
    )
    torch.testing.assert_close(mixed_loss, 0.5 * (free_loss + bearing_loss))


def test_mixture_train_step_logs_unmixed_gaps():
    trainer = _trainer(make_config(batch_size=2, mini_batch_size=2, error_hint_prompt="mixture"))
    traj = _traj()
    traj.hint_bearing = "different bearing hint"
    metrics = trainer.train_step([traj])
    assert "teacher_free_minus_student_logp" in metrics
    assert "teacher_bearing_minus_student_logp" in metrics
