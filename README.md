# Off-policy SDPO on DiligenceBench

A reproduction of the training setup from Trajectory's field report
[Scaling SDPO](https://trajectory.ai/field-notes/scaling-sdpo), applied to
[`paperinstruments/diligence-bench`](https://huggingface.co/datasets/paperinstruments/diligence-bench).

## What SDPO is, in plain terms

Normal RL post-training gives a model one score at the end of a long answer ("that was
3/10") and asks it to work out what to change. That's a thin signal for a lot of text.

SDPO (Self-Distillation Policy Optimization) instead makes the model **its own teacher**.
You run the same model twice on the same task:

- The **student** sees only the question.
- The **teacher** is the _exact same model, same weights_, but with a **hint** secretly
  added to its prompt.

Knowing more, the teacher predicts better next-words. SDPO nudges the student's predictions
toward the teacher's, **at every single token** rather than once at the end. There is no
second model, no reward model, and no human labels — one model, prompted two ways.

**The student never sees the hint.** It only ever answers the bare question; the hint exists
purely to make the teacher a better predictor for one forward pass.

## What "off-policy" means here

Generating an answer is slow; a training step is fast. If the trainer waits for every
answer, the training GPUs sit idle. So generation runs ahead: answers land in a store, and
the trainer pulls from it whenever it's ready. By the time an answer is trained on, the
model has already changed a few times.

Naive off-policy correction (importance sampling) empirically collapses after 50 steps as
on rare tokens the correction ratio explodes to 50-100x and one token hijacks the update.
The fix is two clips: PPO-style ratio clipping, and advantage clipping at 3x a running average. 
We run at **K = 3**: flat accuracy from K=0 to K=3, degrading after, and ~2x wall-clock speedup
over synchronous training.

## The sign convention (read before editing `loss.py`)

This is the single most dangerous detail in the implementation. Both reference
implementations ([`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) and TRL's
`SDPOTrainer`) express the loss identically:

```python
log_ratio      = student_log_probs - teacher_log_probs
per_token_loss = log_ratio.detach() * student_log_probs
```

As a **loss to minimize** the coefficient is `(log π_student − log π_teacher)`. As an
**advantage** for a maximizing update it is the negation, `A_t = log π_teacher −
log π_student` — positive for tokens the teacher likes more.

**Flip it and you train the model to reinforce exactly the tokens the teacher disagrees
with.** It doesn't crash and it produces a perfectly plausible loss curve.
`tests/test_loss.py::test_gradient_moves_toward_teacher` is the guard: it runs a real
optimizer step and asserts the student's log-prob _increases_ on the token the teacher
prefers.

There is deliberately **no baseline, no `+1`, no k3 estimator, and no advantage whitening**.
The paper proves the baseline term vanishes identically; whitening would destroy the signal,
because unlike GRPO's relative advantages the _absolute sign_ here means "teacher agrees /
disagrees" and is the entire point.

## Configuration

| Setting            | Value                             | Source                      |
| ------------------ | --------------------------------- | --------------------------- |
| model              | `Qwen/Qwen3-8B`                   | spec                        |
| GPUs               | 4 rollout / 4 trainer (DDP)       | see below                   |
| batch / mini-batch | 32 / 16                           | spec                        |
| max staleness K    | 3                                 | blog + confirmed            |
| clip window        | `clip(r, 0.8, 1.4)`               | see below                   |
| advantage clip     | 3.0x EMA (decay 0.99)             | blog (3x); EMA spec is ours |
| group size         | 1, failures retained              | blog's headline finding     |
| KL penalty         | off                               | blog dropped it             |

**GPU split.** Rollout GPUs come first (vLLM's multiprocess workers pick `cuda:0..TP-1`
by worker index); each trainer rank pins `cuda:{n_rollout_gpus + rank}`. The trainer is
**DDP, not FSDP**: Qwen3-8B in pure bf16 (~16 GB weights + ~16 GB gradients + ~33 GB Adam
state ≈ 65 GB) fits on a single 80 GB H100 — tightly, which is why `run.py` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — so sharding buys nothing; each rank
holds a full copy and only gradients are all-reduced. The hinted teacher gets **no dedicated GPU**: it is the *same weights* as the
student, and every DDP rank already holds a full copy, so each rank runs its own teacher
forward locally under `no_grad`. That frees the 8th GPU for rollout (TP=4, a power of two,
divides attention heads evenly), which is the slow side and the reason this run is async
in the first place.

**Clip window.** The IS ratio `r = π_current/π_rollout` is centered at **1.0**, so the
window must contain 1.0 — a `[1.2, 1.4]` window would clip every _unchanged_ token. `1.4`
is DAPO's asymmetric "clip-higher" upper bound; `0.8` is the matching lower bound
(ε_low = 0.2, the blog's value).

**EMA advantage clipping.** The blog says only "a fixed multiple of its running mean" at 3x.
It never specifies which statistic, over what window, or how it's initialized. Our spec:

```
batch_mean = masked_mean(|A_t|)                     # mean magnitude, not signed mean
ema        = decay * ema + (1 - decay) * batch_mean
ema_hat    = ema / (1 - decay ** step)              # bias correction
A_clipped  = clamp(A_t, -3 * ema_hat, +3 * ema_hat)
```

Magnitude, not signed mean, because signed advantages are roughly symmetric and would
cancel toward zero. Bias correction is on because an EMA starting at 0 is badly biased low
for the first ~100 steps — exactly when training is least stable.

## Hints are error-conditioned

Every hint is written **per rollout** by an LLM (`deepseek/deepseek-v4-flash` on OpenRouter)
that reads the draft the student actually produced, compares it against the rubric, and names
where *that* draft fell short. Two rollouts of the same task get different hints: one that
missed the funding-risk angle is told about funding risk; one that missed the margin math is
told something else.

There is no static rubric-derived hint. An earlier version built hints from the rubric alone,
which meant every rollout of a task got a byte-identical hint no matter what the model wrote.

**The ablation** — may the hint state the answer? Both arms see the full rubric; only the
output differs:

```bash
python run.py                              # answer_free (default)
python run.py --hint-prompt answer_bearing # may cite the missed figures verbatim
```

`answer_free` is enforced by the prompt alone — there is no regex backstop, so a model that
ignores the instruction can leak a figure into the teacher's context. `answer_bearing` makes
the teacher a model reading the answer key, which is closer to supervised distillation than
self-distillation; a win there measures something different and should be reported as such.

**The main risk to watch.** SDPO's entire gradient is the teacher-student gap. If the hint is
too subtle for the teacher to outpredict the student, that gap goes to ~0, the loss goes to
~0, and training will not move — no matter how healthy the loss curve looks. `trainer.py` logs
`teacher_minus_student_logp` every step and warns loudly when it approaches zero.

A rollout whose hint cannot be generated is **dropped**, not trained with an empty hint: an
unhinted teacher is identical to the student and would contribute ~zero gradient. Watch
`RolloutEngine.hintless_dropped` — a sustained nonzero value means the hint model, not the
rollout engine, is the bottleneck on data production.

## Verification

Everything through the trainer is CPU-testable — no GPU, no model downloads:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers datasets rubric requests \
  python-dotenv pytest
.venv/bin/python -m pytest tests/ -q          # 123 tests
```

The test suite is fully offline. Only the judge reaches the network, and only at eval time
— it runs on OpenRouter, so `python run.py` and `--baseline` need a key in `.env` at the
repo root (gitignored; `run.py` loads it automatically):

```bash
echo 'OPENROUTER_API_KEY=sk-or-v1-...' > .env
```

What the tests actually pin down:

- **Sign correctness** — a real optimizer step moves the student toward the teacher.
  Verified by mutation: flipping the sign in `loss.py` fails this test _and_ the
  reference-equivalence test.
- **Reference equivalence** — our loss equals `log_ratio.detach() * student_log_probs`.
- **Log-prob alignment** — with an identical hint, student and teacher log-probs match
  exactly (catches both causal-shift and prompt-offset bugs).
- **Staleness boundary** — K=3 is trained on, K=4 is evicted.
- **Loss decreases** — repeated steps on a fixed batch reduce the loss on a real model.
- **Gradient accumulation** — mini-batched gradients equal full-batch gradients.
- **No answer leakage** — asserts no generated hint contains a digit, across all 150 rows.
- **Weight-sync deadlock ordering** — asserts the receive-side RPC is dispatched _before_
  the trainer broadcasts. Also mutation-verified: restoring the naive
  await-then-broadcast order makes the test hang and fail.

## Weight sync

Uses vLLM's **native** `vllm.distributed.weight_transfer` NCCL API, matching the approach
in `healthbench-rl`. The engine is built with `WeightTransferConfig(backend="nccl")`, and
each update is a transaction:

```
pause_generation(mode="keep") → start_weight_update()
  → per bucket: post receive RPC, then trainer_send_weights, then join
→ finish_weight_update() → reset_prefix_cache() → resume_generation()
```

vLLM owns the group lifecycle, receivers post their own matching broadcasts from the
names/dtypes/shapes metadata in the request, and bucketing is native via `packed=True`.
No worker extension and no private `load_weights` call, so none of it rides on unversioned
internals.

Details that matter:

- **Ranks are per GPU worker, not per engine.** Each engine contributes
  `tensor_parallel_size` ranks; rank 0 is the trainer (sender). Getting this wrong hangs
  the rendezvous.
- **Ordering.** The receive-side `update_weights` RPC must be in flight _before_ the
  trainer enters `trainer_send_weights`, or the sender blocks on a broadcast nobody is
  listening for. Pinned by a test — and mutation-verified: reversing the order makes it
  hang and fail.
- **`mode="keep"`** freezes in-flight rollouts instead of aborting them, so a long
  generation survives the swap and resumes under the new weights.
- **Prefix cache is reset** after each update. It holds KV computed under the _old_
  weights; reusing it would splice two policy versions into one trajectory and make
  `policy_version` a lie, corrupting the staleness accounting.
- **bf16 casting** on the send side, matching the rollout engine's dtype.

## vLLM version pin

Written against **vLLM 0.26.x — pin it exactly.** The native weight-transfer API
(`init_weight_transfer_engine` / `start_weight_update` / `update_weights` /
`finish_weight_update`) does **not** exist in 0.11 or 0.12. Also:

- `logprobs_mode` defaults to `"raw_logprobs"` (pre temperature/top-p). We set
  `"processed_logprobs"` because the IS ratio's denominator must come from the same
  distribution the sampler actually drew from — otherwise every ratio is off by exactly
  the sampling transform. Silent and systematic, not a crash.
- `SamplingParams.flat_logprobs` must stay `False`, or `.logprobs` stops being a plain list.

On the 8-GPU box:

```bash
python run.py --baseline               # zero-shot held-out judge score (the number to beat)
python run.py --smoke                  # 10 steps on Qwen3-0.6B, 2 GPUs, single process
torchrun --nproc-per-node=4 run.py     # the real run: 4 vLLM GPUs + 4 DDP trainer ranks
```

The real run must be launched with `torchrun` (one process per trainer GPU); rank 0 owns
all the async machinery and broadcasts each batch, ranks 1–3 just train their shard.
Checkpoints (model + optimizer + EMA-clipper state) land in `runs/sdpo-diligence/step_N/`
every `checkpoint_interval` steps, plus `step_final/` at the end — RunPod pods get
interrupted, and a run that saves nothing never happened.

During a real run, watch in order: (1) `teacher-student gap` is clearly non-zero,
(2) clip fractions are a modest minority of tokens, (3) `staleness` sits at or below 3,
(4) held-out judge score rises above the zero-shot baseline.

**Cold start & compile caches.** When `/workspace` (RunPod's volume disk, which survives
pod stop/start) exists, `run.py` automatically points every slow-to-rebuild cache at it:

| env var                  | default                          | what it saves on re-boot        |
| ------------------------ | -------------------------------- | ------------------------------- |
| `VLLM_CACHE_ROOT`        | `/workspace/.cache/vllm`         | vLLM's torch.compile artifacts  |
| `TORCHINDUCTOR_CACHE_DIR`| `/workspace/.cache/torchinductor`| trainer's compiled kernels      |
| `TRITON_CACHE_DIR`       | `/workspace/.cache/triton`       | trainer's Triton kernels        |
| `HF_HOME`                | `/workspace/hf`                  | ~16 GB model download           |

All are `setdefault`, so anything you export yourself wins. The FIRST run on a fresh pod
pays everything once — model download, vLLM engine compile, and the trainer's
`torch.compile` on its first step (the trainer compiles with `dynamic=True` because padded
batch shapes differ every step). Every boot after that reuses the caches and starts in
seconds, not tens of minutes. If training misbehaves, `compile_trainer=False` in
`config.py` is the first debug lever — compile + DDP + gradient checkpointing + varying
shapes is the most fragile stack in here (`--smoke` already runs uncompiled for this
reason).

**RunPod checklist.** Put the `.env` with `OPENROUTER_API_KEY` at the repo root, pin
`vllm==0.26.0` in the image, and give the container a large `/dev/shm` — vLLM's
multiprocess tensor-parallel workers communicate through it. Also put the *venv itself*
on `/workspace` (e.g. `uv venv /workspace/venv && uv pip install --python
/workspace/venv/bin/python -r requirements.txt`): packages download and build once on the
first boot and every later boot is a seconds-fast no-op.

## Deviations from the blog, and why

| Blog                              | Here                                     | Why                            |
| --------------------------------- | ---------------------------------------- | ------------------------------ |
| Tau-Retail (multi-turn, tool-use) | diligence-bench (single-turn, long-form) | requested                      |
| symmetric PPO clip ε=0.2          | DAPO decoupled 0.8 / 1.4                 | requested                      |
| hint = canonical answer           | hint = answer-free reasoning nudge       | requested                      |
| no reward model at all            | rubric judge for **eval only**           | judge is never in the gradient |

## References

- [Scaling SDPO](https://trajectory.ai/field-notes/scaling-sdpo) — Trajectory, June 2026
- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) —
  Hübotter et al., arXiv:2601.20802
- [`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) — official implementation
