"""All hyperparameters for the off-policy SDPO run, in one place.

Reproduces the setup from Trajectory's "Scaling SDPO" field report
(https://trajectory.ai/field-notes/scaling-sdpo) on the paperinstruments/diligence-bench
financial-diligence benchmark.

Every value that came from the blog is marked BLOG. Values the blog left unspecified are
marked OURS, so it is always clear what is reproduction and what is our own choice.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- model ----
    model: str = "Qwen/Qwen3-4B"
    # Tiny model for the end-to-end smoke test before committing GPUs to a real run.
    smoke_model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"

    # ---- GPU topology (8 total) ----
    # 4 rollout + 4 trainer. Rollout GPUs come first (vLLM's workers take cuda:0..N-1),
    # trainer rank r pins cuda:{n_rollout_gpus + r}. TP=4 is a power of two, so it divides
    # Qwen3's attention heads evenly.
    n_rollout_gpus: int = 4
    # Trainer is DDP: Qwen3-4B (~8 GB bf16 + ~32 GB Adam state) fits on one 80 GB H100,
    # so each rank holds a full copy and only gradients are synced -- FSDP's sharding
    # would buy nothing here and add failure modes.
    n_trainer_gpus: int = 4
    # The hinted teacher needs no GPU of its own: it is the SAME weights as the student,
    # and every DDP rank already holds a full copy, so each rank runs its own teacher
    # forward locally under no_grad.
    n_teacher_gpus: int = 0

    # ---- batching ----
    batch_size: int = 32
    mini_batch_size: int = 16
    # BLOG: group size 1 is the headline finding -- production gives one trajectory per
    # user, and SDPO's loss is well-defined for a single trajectory.
    group_size: int = 1
    # BLOG: failed trajectories still produce gradients, so they are retained. Dropping
    # them is what forces GRPO to discard nearly every rollout on hard tasks.
    keep_failures: bool = True

    # ---- off-policy staleness ----
    # BLOG: accuracy is flat from K=0 through K=3 then degrades; K=3 buys ~2x wall-clock.
    # A trajectory older than this many trainer steps is evicted rather than trained on.
    max_staleness: int = 3

    # ---- DAPO decoupled clipping of the IS ratio ----
    # The ratio r = pi_current / pi_rollout is centered at 1.0, so the window must contain
    # 1.0 or every unchanged token gets clipped. eps_low=0.2 / eps_high=0.4 gives the
    # asymmetric "clip-higher" window that is the point of DAPO.
    clip_ratio_low: float = 0.8   # 1 - 0.2
    clip_ratio_high: float = 1.4  # 1 + 0.4  (BLOG uses symmetric 0.2; the asymmetry is ours)
    # The reference SDPO implementations instead apply a one-sided max-only clamp on the
    # distillation branch (TRL default 2.0). Kept available for parity checks.
    use_one_sided_clip: bool = False
    one_sided_clip_max: float = 2.0

    # ---- advantage clipping via EMA ----
    # BLOG: "bounds the per-token advantage at a fixed multiple of its running mean" at 3x.
    # The blog never says which statistic, over what window, or how it is initialized --
    # the decay and bias correction below are OURS. See train/loss.py for the mechanism.
    adv_clip_mult: float = 3.0
    adv_ema_decay: float = 0.99
    adv_ema_bias_correction: bool = True

    # BLOG: dropped in favor of advantage clipping -- a reference-model KL penalty roughly
    # doubles forward-pass compute and biases the policy toward whatever reference is chosen.
    use_kl_penalty: bool = False

    # ---- optimizer ----
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    total_steps: int = 500

    # ---- rollout / generation ----
    temperature: float = 1.0
    top_p: float = 1.0
    max_prompt_tokens: int = 2048
    max_response_tokens: int = 1536
    gpu_memory_utilization: float = 0.85
    # Push weights to the rollout engines every N trainer steps and bump policy_version.
    weight_sync_interval: int = 1
    # Rendezvous for vLLM's native NCCL weight-transfer group. Rank 0 is the trainer
    # (sender); every vLLM GPU worker joins as a receiver.
    weight_sync_host: str = "127.0.0.1"
    weight_sync_port: int = 51216
    weight_sync_timeout_s: float = 180.0
    # Bucket size for weight transfer -- amortizes per-transfer overhead across the many
    # small tensors in a transformer. Larger buckets mean fewer round trips but more peak
    # memory held in the bf16 staging copies.
    weight_sync_bucket_mb: int = 512
    # Capacity of the trajectory store; oldest-first eviction so a stalled trainer cannot
    # OOM the box while rollout workers keep producing.
    store_capacity: int = 512

    # ---- data ----
    dataset_name: str = "paperinstruments/diligence-bench"
    dataset_split: str = "test"  # the only split the dataset ships
    n_heldout: int = 30          # 120 train / 30 held-out
    split_seed: int = 0

    # ---- hint (error-conditioned, generated per rollout) ----
    # The hint is prepended ONLY to the teacher's context; the student never sees it. It is
    # generated by an LLM that reads the draft this rollout produced against the rubric and
    # names where THAT draft fell short, so two rollouts of one task get different hints.
    # A rollout whose hint cannot be generated is DROPPED (see train/rollout.py): an
    # unhinted teacher equals the student and contributes ~zero gradient.
    #
    # The ablation axis -- may the hint state the answer?
    #   "answer_free"    -- reasoning angles only; no figures, no conclusions.
    #   "answer_bearing" -- may cite the missed criteria verbatim, figures included.
    # NOTE "answer_bearing" makes the teacher a model reading the answer key, which is
    # closer to supervised distillation than self-distillation. That is the point of the
    # arm, but it means a win there measures something different -- see data/hint.py.
    # `answer_free` is enforced by the prompt alone; there is no regex backstop.
    error_hint_prompt: str = "answer_free"
    error_hint_model: str = "deepseek/deepseek-v4-flash"
    error_hint_concurrency: int = 8  # bound on in-flight hint calls
    error_hint_timeout: float = 60.0

    # ---- judge (eval only -- never in the gradient) ----
    # Served via OpenRouter; needs OPENROUTER_API_KEY in the environment. The judge is off
    # the gradient path, so its cost is pure overhead -- v4-flash is cheap ($0.14/$0.28 per
    # M tokens) and its 1M context comfortably holds a rubric plus a long answer.
    judge_model: str = "deepseek/deepseek-v4-flash"
    judge_max_concurrency: int = 8
    judge_max_retries: int = 3
    judge_timeout: float = 120.0
    eval_interval: int = 25  # trainer steps between held-out evals

    # ---- logging ----
    log_interval: int = 1
    checkpoint_interval: int = 50
    output_dir: str = "runs/sdpo-diligence"

    def __post_init__(self) -> None:
        # <= 8 rather than == 8: dataclasses.replace() re-runs this check, and the smoke
        # config deliberately uses a subset of the box (1 rollout + 1 trainer).
        total = self.n_rollout_gpus + self.n_trainer_gpus + self.n_teacher_gpus
        if total > 8:
            raise ValueError(f"GPU split must sum to <= 8, got {total}")
        if self.n_rollout_gpus < 1 or self.n_trainer_gpus < 1:
            raise ValueError("need at least 1 rollout GPU and 1 trainer GPU")
        # batch_size % n_trainer_gpus is checked in run.py at launch, against the actual
        # torchrun world size -- tests use tiny batches with no GPUs at all.
        if not self.clip_ratio_low < 1.0 < self.clip_ratio_high:
            raise ValueError(
                f"clip window [{self.clip_ratio_low}, {self.clip_ratio_high}] must contain "
                "1.0, or unchanged tokens (ratio == 1) would be clipped"
            )
        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by mini_batch_size")
        if not 0.0 < self.adv_ema_decay < 1.0:
            raise ValueError("adv_ema_decay must be in (0, 1)")
        if self.error_hint_prompt not in ("answer_free", "answer_bearing"):
            raise ValueError(
                f"error_hint_prompt must be 'answer_free' or 'answer_bearing', "
                f"got {self.error_hint_prompt!r}"
            )
        if self.error_hint_concurrency < 1:
            raise ValueError("error_hint_concurrency must be >= 1")


CONFIG = Config()
