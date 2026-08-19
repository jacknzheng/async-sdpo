"""Backend selection and the trainer/transport seam.

These are the regression tests for the abstraction that keeps a specific inference engine
out of `train/trainer.py`. The valuable one is
`test_trainer_delegates_every_bucket_to_the_transport`: it is the reason the trainer no
longer imports vLLM, and it runs entirely on CPU with a fake transport.
"""

import pytest
import torch

from train.backends import BACKEND_NAMES, get_backend
from train.backends.backend import InferenceEngine, WeightBucket, WeightTransport


def test_vllm_backend_resolves_to_a_pair():
    engine_cls, transport_cls = get_backend("vllm")
    assert engine_cls.__name__ == "VLLMRolloutEngine"
    assert transport_cls.__name__ == "NCCLWeightTransport"


def test_unknown_backend_raises_with_available_names():
    with pytest.raises(ValueError, match="unknown rollout backend"):
        get_backend("sglang")


def test_vllm_is_a_registered_name():
    assert "vllm" in BACKEND_NAMES


def test_backend_classes_satisfy_the_protocols():
    """runtime_checkable Protocols check method presence, which is exactly the guard we
    want: a backend missing `finish_update` fails here rather than mid-run."""
    engine_cls, transport_cls = get_backend("vllm")
    # Constructed without start(), so no vLLM import is triggered.
    from train.config import Config

    assert isinstance(engine_cls(Config()), InferenceEngine)
    assert isinstance(transport_cls(), WeightTransport)


# ---- WeightBucket ----


def test_weight_bucket_rejects_ragged_fields():
    """Parallel lists that drift silently misalign the transfer -- the receiver allocates
    from `shapes` while the sender ships `tensors`."""
    with pytest.raises(ValueError, match="parallel lists"):
        WeightBucket(
            names=["a", "b"],
            dtype_names=["bfloat16"],
            shapes=[[1]],
            tensors=[torch.zeros(1)],
        )


def test_weight_bucket_len_is_parameter_count():
    bucket = WeightBucket(
        names=["a"], dtype_names=["bfloat16"], shapes=[[2]], tensors=[torch.zeros(2)]
    )
    assert len(bucket) == 1


# ---- the trainer/transport seam ----


class FakeTransport:
    """Records what the trainer hands it. Stands in for any real backend."""

    def __init__(self) -> None:
        self.setup_args = None
        self.sent: list[WeightBucket] = []
        self.torn_down = False

    def setup(self, master_address, master_port, world_size) -> None:
        self.setup_args = (master_address, master_port, world_size)

    def send_bucket(self, bucket) -> None:
        self.sent.append(bucket)

    def teardown(self) -> None:
        self.torn_down = True


def _trainer_with(transport):
    """A tiny CPU trainer wired to `transport`. Mirrors tests/test_trainer.py's helper."""
    from conftest import make_config
    from tests.test_trainer import FakeTokenizer, _model, _task
    from train.store import TrajectoryStore
    from train.trainer import SDPOTrainer

    return SDPOTrainer(
        model=_model(),
        tokenizer=FakeTokenizer(),
        store=TrajectoryStore(max_staleness=3),
        config=make_config(batch_size=4, mini_batch_size=2),
        tasks_by_id={"1": _task("1")},
        device="cpu",
        transport=transport,
    )


def test_trainer_delegates_every_bucket_to_the_transport():
    """THE test for the decoupling: the trainer must ship every parameter through whatever
    transport it was given, in bucket order, with no engine-specific code of its own."""
    transport = FakeTransport()
    trainer = _trainer_with(transport)
    trainer.setup_weight_sync("127.0.0.1", 51216, 2)

    expected = [name for name, _ in trainer.unwrapped.named_parameters()]
    for bucket in trainer.weight_buckets():
        trainer.send_weight_bucket(bucket)

    assert transport.setup_args == ("127.0.0.1", 51216, 2)
    shipped = [n for bucket in transport.sent for n in bucket.names]
    assert shipped == expected


def test_send_before_setup_raises_even_with_a_transport():
    """The guard is on the rendezvous, not on transport presence: sending before every rank
    has joined blocks on a broadcast nobody is listening for."""
    trainer = _trainer_with(FakeTransport())
    bucket = WeightBucket(
        names=["w"], dtype_names=["bfloat16"], shapes=[[2]], tensors=[torch.zeros(2)]
    )
    with pytest.raises(RuntimeError, match="weight sync not initialized"):
        trainer.send_weight_bucket(bucket)


def test_setup_without_a_transport_raises():
    """DDP ranks 1..N-1 get no transport; if one ever tried to sync, say so plainly."""
    trainer = _trainer_with(None)
    with pytest.raises(RuntimeError, match="no weight transport"):
        trainer.setup_weight_sync("127.0.0.1", 51216, 2)
