"""The SDPO training loop.

Each step:

  1. Pull a batch from the trajectory store (evicting anything staler than K=3).
  2. Recompute student log-probs under the CURRENT weights (with gradient).
  3. Run the hinted teacher forward pass -- same weights, hint-prefixed context, no grad.
  4. Compute the SDPO loss: per-token reverse KL, advantage-clipped, IS-ratio-clipped.
  5. Step the optimizer, then periodically push weights to the rollout engines and bump
     the store's policy version.

Three things in here are load-bearing and easy to get subtly wrong:

* The teacher is the SAME MODEL, run under `torch.no_grad()` with a different context. It
  is not a copy, not an EMA, not a frozen snapshot. If you ever find yourself loading a
  second checkpoint here, the implementation has drifted from SDPO.
* `rollout_logprobs` come from the store, captured at generation time. Never recompute
  them -- that would make every IS ratio 1.0 and silently delete the off-policy correction.
* The EMA advantage clipper is training state. It must be checkpointed, and its EMA must be
  all-reduced across ranks so every rank clips identically.

The metric to watch is `teacher_minus_student_logp`. It is the mean per-token advantage
before clipping, i.e. how much better the hinted teacher predicts than the student. SDPO's
entire gradient is that gap. If it sits near zero the hint is too weak, the loss is ~0, and
training will not move regardless of how healthy the loss curve looks.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from config import Config
from data.dataset import Task, build_prompt
from train.batch import SDPOBatch, build_batch, gather_response_logprobs
from train.loss import EMAAdvantageClipper, sdpo_loss
from train.store import TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Everything that must survive a checkpoint/resume."""

    step: int = 0
    policy_version: int = 0
    clipper: EMAAdvantageClipper = field(default_factory=EMAAdvantageClipper)

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "policy_version": self.policy_version,
            "clipper": self.clipper.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step = state["step"]
        self.policy_version = state["policy_version"]
        self.clipper.load_state_dict(state["clipper"])


class SDPOTrainer:
    """Off-policy SDPO trainer.

    `model` is the policy. The same object serves as both student and teacher -- the only
    difference is the context each forward pass sees.
    """

    def __init__(
        self,
        model,
        tokenizer,
        store: TrajectoryStore,
        config: Config,
        tasks_by_id: dict[str, Task],
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.store = store
        self.config = config
        self.tasks_by_id = tasks_by_id
        self.device = device
        # Full activations at mini-batch x ~3.5k tokens exceed 80 GB; checkpointing trades
        # them for recompute. use_cache is incompatible with checkpointing (and useless
        # for full-sequence scoring anyway). Non-reentrant is the variant that composes
        # with DDP without requiring static_graph.
        self.unwrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.unwrapped.config.use_cache = False
        self.state = TrainerState(
            clipper=EMAAdvantageClipper(
                mult=config.adv_clip_mult,
                decay=config.adv_ema_decay,
                bias_correction=config.adv_ema_bias_correction,
            )
        )
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        # Set by setup_weight_sync(); None until the weight-transfer group is joined.
        self._transfer_group = None

    @property
    def unwrapped(self):
        """The bare HF model. Under DDP `self.model` is a wrapper whose parameter names
        gain a `module.` prefix -- vLLM's weight receiver and checkpoints need the real
        names, so anything that reads names or state dicts must go through here."""
        return getattr(self.model, "module", self.model)

    # ---- forward passes ----

    def _student_logprobs(self, batch: SDPOBatch, max_response: int) -> torch.Tensor:
        """Log-probs under current weights, WITH gradient. The only tensor grad flows through."""
        logits = self.model(
            input_ids=batch.student_input_ids,
            attention_mask=batch.student_attention_mask,
        ).logits
        return gather_response_logprobs(
            logits, batch.student_input_ids, batch.student_response_mask, max_response
        )

    @torch.no_grad()
    def _teacher_logprobs(self, batch: SDPOBatch, max_response: int) -> torch.Tensor:
        """Log-probs of the SAME model on the SAME response tokens, with the hint prepended.

        No gradient: the teacher is a stop-gradient target (the `stopgrad` in the SDPO loss).
        """
        logits = self.model(
            input_ids=batch.teacher_input_ids,
            attention_mask=batch.teacher_attention_mask,
        ).logits
        return gather_response_logprobs(
            logits, batch.teacher_input_ids, batch.teacher_response_mask, max_response
        )

    # ---- batch prep ----

    def prepare_batch(self, trajectories) -> SDPOBatch:
        """Tokenize hint-prefixed teacher prompts and assemble padded tensors.

        The hint is read from the trajectory, not from current config, so a mid-run hint
        change can't mix two different teachers into one batch.
        """
        teacher_prompt_ids = []
        for trajectory in trajectories:
            task = self.tasks_by_id[trajectory.task_id]
            teacher_prompt = build_prompt(task, hint=trajectory.hint)
            teacher_prompt_ids.append(
                self.tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return build_batch(trajectories, teacher_prompt_ids, pad_token_id=pad_id)

    # ---- the step ----

    def train_step(self, trajectories) -> dict[str, float]:
        """One optimizer step over a batch, run in mini-batches for memory."""
        if not trajectories:
            return {}

        self.model.train()
        batch = self.prepare_batch(trajectories).to(self.device)
        max_response = batch.response_mask.size(1)

        self.optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        n_micro = 0

        starts = list(range(0, len(trajectories), self.config.mini_batch_size))
        for start in starts:
            end = min(start + self.config.mini_batch_size, len(trajectories))
            micro = _slice_batch(batch, start, end)

            # Under DDP every backward all-reduces gradients across ranks; during
            # accumulation only the LAST chunk's backward should sync. no_sync must wrap
            # the forward too -- DDP records the sync decision at forward time.
            is_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            sync_context = (
                self.model.no_sync() if is_ddp and start != starts[-1] else nullcontext()
            )
            with sync_context:
                student_lp = self._student_logprobs(micro, max_response)
                teacher_lp = self._teacher_logprobs(micro, max_response)

                loss, metrics = sdpo_loss(
                    student_log_probs=student_lp,
                    teacher_log_probs=teacher_lp,
                    rollout_log_probs=micro.rollout_logprobs,
                    response_mask=micro.response_mask,
                    clipper=self.state.clipper,
                    clip_low=self.config.clip_ratio_low,
                    clip_high=self.config.clip_ratio_high,
                    one_sided=self.config.use_one_sided_clip,
                    one_sided_max=self.config.one_sided_clip_max,
                )

                # Scale so the summed gradient equals the full-batch mean.
                (loss / len(starts)).backward()

            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + value
            n_micro += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.state.step += 1

        # The EMA must agree across ranks or they clip differently and slowly diverge.
        self._sync_clipper()

        averaged = {k: v / max(n_micro, 1) for k, v in accumulated.items()}
        averaged["grad_norm"] = float(grad_norm)
        averaged["step"] = float(self.state.step)
        averaged["mean_batch_staleness"] = float(
            (self.state.policy_version - batch.policy_versions.float()).mean()
        )
        return averaged

    def _sync_clipper(self) -> None:
        """All-reduce the advantage EMA so every trainer rank clips identically."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        tensor = torch.tensor([self.state.clipper.ema], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        self.state.clipper.sync_(float(tensor.item()))

    # ---- weight sync (vLLM native NCCL transfer) ----

    def setup_weight_sync(self, master_address: str, master_port: int, world_size: int) -> None:
        """Join the weight-transfer group as the SENDER at rank 0.

        `world_size` counts vLLM GPU workers plus one for the trainer -- each rollout engine
        contributes `tensor_parallel_size` ranks, not one rank per engine. This blocks until
        every rank has joined, so the rollout engines must already be initializing their
        side (the orchestrator issues both concurrently).
        """
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        logger.info(
            "weight-sync: opening rendezvous at %s:%d, waiting for %d receiver(s)",
            master_address, master_port, world_size - 1,
        )
        self._transfer_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            )
        )
        logger.info("weight-sync: rendezvous complete, all %d ranks joined", world_size)

    def weight_buckets(self):
        """Yield (names, dtype_names, shapes, tensors) buckets of the current policy.

        Bucketing amortizes per-transfer overhead across many small tensors. Weights are
        cast to bf16 for transfer, matching the rollout engine's dtype -- sending fp32 to a
        bf16 engine would either fail the dtype check or silently double the bytes on the
        wire.
        """
        max_bytes = self.config.weight_sync_bucket_mb * 1024 * 1024
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0

        for name, param in self.unwrapped.named_parameters():
            bucket.append((name, param))
            bucket_bytes += param.numel() * param.element_size()
            if bucket_bytes >= max_bytes:
                yield _pack_bucket(bucket)
                bucket, bucket_bytes = [], 0
        if bucket:
            yield _pack_bucket(bucket)

    def send_weight_bucket(self, names: list[str], tensors: list[torch.Tensor]) -> None:
        """Broadcast one bucket over the weight-sync group.

        The receivers post their matching broadcasts from the metadata in their
        `update_weights` request, so send order must match `names` exactly. Their RPC must
        already be in flight when this is called, or this blocks on a broadcast nobody is
        listening for.
        """
        if self._transfer_group is None:
            raise RuntimeError("weight sync not initialized; call setup_weight_sync first")

        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )

        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=zip(names, tensors),
            trainer_args=NCCLTrainerSendWeightsArgs(group=self._transfer_group, packed=True),
        )

    def bump_policy_version(self) -> int:
        """Advance the policy version after pushing weights to the rollout engines.

        Called by the orchestrator AFTER the weight push completes. Trajectories already in
        the store get one step staler; anything past K=3 is evicted on the next sample.
        """
        self.state.policy_version += 1
        return self.state.policy_version

    # ---- checkpointing ----

    def state_dict(self) -> dict:
        return {
            "model": self.unwrapped.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainer": self.state.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.unwrapped.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.state.load_state_dict(state["trainer"])


def _pack_bucket(bucket: list[tuple[str, torch.Tensor]]):
    """Cast one bucket to bf16 and extract the metadata the receivers need."""
    names, tensors = [], []
    for name, param in bucket:
        tensor = param.data
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        names.append(name)
        tensors.append(tensor)
    dtype_names = [str(t.dtype).split(".")[-1] for t in tensors]
    shapes = [list(t.shape) for t in tensors]
    return names, dtype_names, shapes, tensors


def _slice_batch(batch: SDPOBatch, start: int, end: int) -> SDPOBatch:
    """Slice a batch along the batch dimension for gradient accumulation."""
    return SDPOBatch(
        student_input_ids=batch.student_input_ids[start:end],
        student_attention_mask=batch.student_attention_mask[start:end],
        student_response_mask=batch.student_response_mask[start:end],
        teacher_input_ids=batch.teacher_input_ids[start:end],
        teacher_attention_mask=batch.teacher_attention_mask[start:end],
        teacher_response_mask=batch.teacher_response_mask[start:end],
        rollout_logprobs=batch.rollout_logprobs[start:end],
        response_mask=batch.response_mask[start:end],
        task_ids=batch.task_ids[start:end],
        policy_versions=batch.policy_versions[start:end],
    )


def log_metrics(metrics: dict[str, float], store: TrajectoryStore) -> None:
    """Emit one line per step, leading with the diagnostics that actually matter."""
    combined = {**metrics, **store.metrics()}
    gap = combined.get("teacher_minus_student_logp", 0.0)
    logger.info(
        "step %d | loss %.4f | teacher-student gap %+.4f | adv_clip %.1f%% | "
        "ratio_clip lo %.1f%% hi %.1f%% | staleness %.2f | grad %.3f",
        int(combined.get("step", 0)),
        combined.get("loss", 0.0),
        gap,
        100 * combined.get("adv_clip_frac", 0.0),
        100 * combined.get("ratio_clip_frac_low", 0.0),
        100 * combined.get("ratio_clip_frac_high", 0.0),
        combined.get("store_mean_staleness", 0.0),
        combined.get("grad_norm", 0.0),
    )
    if abs(gap) < 1e-3:
        logger.warning(
            "teacher-student gap is ~0 (%.6f): the hint is not making the teacher better "
            "than the student, so the SDPO gradient has vanished and training will not "
            "move. Try config.error_hint_prompt='answer_bearing' for a stronger teacher "
            "(see data/hint.py).",
            gap,
        )
