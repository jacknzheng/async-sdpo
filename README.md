# Off-policy SDPO (tau2 + DiligenceBench)

A reproduction of the training setup from Trajectory's field report
[Scaling SDPO](https://trajectory.ai/field-notes/scaling-sdpo), applied to two
tool-using evals:

- **tau2** (default) — Sierra `banking_knowledge` + `retail` + `airline`. Multi-turn
  TIR with a user simulator; held-out metric is pass^1.
- **DiligenceBench** — [`paperinstruments/diligence-bench`](https://huggingface.co/datasets/paperinstruments/diligence-bench).
  Multi-turn TIR with Parallel `web_search`, scored by a rubric judge (eval only).

Launch with the scripts in `scripts/`. Do not start from a raw `python run.py` on an
8×H100 unless you are iterating on a one-off override.

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

**There is no reward in the gradient.** Tau2 pass^1 and the diligence rubric judge are
eval diagnostics. The training signal is the teacher−student logp gap
(`teacher_minus_student_logp`). If that gap is ~0, training is a no-op even when the loss
curve looks healthy. This is the thing people mean by "reward variance collapsed" in this
repo.

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

## Run it (8×H100)

Default stack: `Qwen/Qwen3.8-27B`, 4 vLLM rollout GPUs + 4 FSDP2 trainer ranks, max
staleness K=3, `vllm==0.26.x` and `torch==2.11.0+cu128` pinned exactly. Python 3.12.
Do not let uv pull torch 2.13 (CUDA 13) — it will not load on a 12.8 driver.

### Secrets

Put a gitignored `.env` at the repo root (`run.py` loads it before anything that reads
keys). Never commit it.

```bash
OPENROUTER_API_KEY=...   # hints, diligence judge, tau2 user simulator
WANDB_API_KEY=...
HF_TOKEN=...             # model download
PARALLEL_API_KEY=...     # diligence web_search only
```

### Install

```bash
uv sync --extra knowledge          # tau2 + banking retrieval; also enough for diligence
bash scripts/setup_tau2_sandbox.sh # banking_knowledge shell tool; skip only if you will
                                   # never touch banking. This script *probes namespaces*,
                                   # not just `which`.
uv run pytest tests/ -q            # offline; no GPU
```

`setup_tau2_sandbox.sh` installs `@anthropic-ai/sandbox-runtime@0.0.23` plus
`ripgrep`, `bubblewrap`, `socat`. Retail and airline do not need this.

**Known failure: `which` is a false green.** tau2 only checks that `srt` / `rg` /
`bwrap` / `socat` exist on PATH, so env construction succeeds. The banking `shell`
tool then runs inside `srt` → `bwrap` → `unshare`. GPU pods (RunPod / Baseten /
default Docker) often install those binaries but **seccomp-block nested namespaces**.
Every shell call then fails at runtime with:

```
bwrap: Creating new namespace failed: Operation not permitted
```

That is container policy, not a missing package. Recreate the pod/container with
`--privileged`, or `--security-opt seccomp=unconfined --security-opt apparmor=unconfined`.
On RunPod: Edit Pod → extra flags → `--privileged`, then Start. On Baseten: request a
privileged / unconfined-seccomp workstation. Confirm with
`bwrap --ro-bind / / --dev /dev /bin/echo bwrap-ok` and `srt -c 'echo srt-ok'`.
`run.py` runs this probe at tau2 startup and exits if it fails — do not train through
a fleet of zero-reward banking episodes. BM25 / dense search still work; only shell
is dead. Treat `gold_banking` as the canary before the three-domain `gold` run.

### Launch

```bash
# Prove the loop on one box before scaling out. Tiny model, 10 steps, not a result.
bash scripts/run_taubench.sh smoke
bash scripts/run_diligencebench.sh smoke

# Zero-shot held-out numbers to beat.
bash scripts/run_taubench.sh baseline
bash scripts/run_diligencebench.sh baseline

# Training ablations (one 8×H100 per arm is the intended fleet).
bash scripts/run_taubench.sh gold
bash scripts/run_taubench.sh step_hint
bash scripts/run_taubench.sh gold_banking

bash scripts/run_diligencebench.sh answer_free
bash scripts/run_diligencebench.sh answer_bearing
bash scripts/run_diligencebench.sh mixture
```

Dotted overrides after the mode, e.g. `bash scripts/run_taubench.sh gold trainer.total_steps=200`.
Logs land in `/log/<run_name>/` (`train.log`, `console.log`, `args.txt`, `config.yaml`);
if `/log` is not writable the scripts fall back to `./log`. Checkpoints go to
`runs/sdpo-tau2/` or `runs/sdpo-diligence/`. Wandb projects: `sdpo-tau2` / `sdpo-diligence`.

The real run is `torchrun --nproc-per-node=4` (one process per trainer GPU). Rank 0 owns
rollout, the store, eval, and wandb; ranks 1–3 only train. `--smoke` and `--baseline` are
single-process.

### Host disk (not just RunPod)

When `/workspace` exists (RunPod volume), `run.py` points compile/HF caches there. On any
other host (Baseten, a raw workstation) that directory often does not exist — export the
same vars at a persistent disk yourself, or every boot re-compiles:

| env var                   | RunPod default                     | what it saves                 |
| ------------------------- | ---------------------------------- | ----------------------------- |
| `VLLM_CACHE_ROOT`         | `/workspace/.cache/vllm`           | vLLM torch.compile artifacts  |
| `TORCHINDUCTOR_CACHE_DIR` | `/workspace/.cache/torchinductor`  | trainer compiled kernels      |
| `TRITON_CACHE_DIR`        | `/workspace/.cache/triton`         | trainer Triton kernels        |
| `HF_HOME`                 | `/workspace/hf`                    | model weights                 |

All are `setdefault`. Also: large `/dev/shm` for vLLM TP workers; pin `vllm==0.26.0`;
put the venv on persistent disk so the image is not rebuilt every boot.

**Baseten / workstation bringup.** vLLM 0.26.0 is not in `pyproject.toml` — `uv pip install
vllm==0.26.0` after `uv sync`. That wheel may pull CUDA-13 torchvision/torchaudio; re-pin
torchvision to cu128 and drop torchaudio. tau2 UserSimulator needs
`OPENAI_API_KEY=$OPENROUTER_API_KEY` and `OPENAI_API_BASE=https://openrouter.ai/api/v1`
(`run.py` sets these if only the OpenRouter key is present). Weight-sync on these images
wants `NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1`. Kill leftover vLLM EngineCore processes
between runs (they pin GPU 0 VRAM). `asyncio.to_thread` weight-sync must
`torch.cuda.set_device` on the trainer GPU or NCCL binds to cuda:0 (a vLLM worker).

If training misbehaves, `trainer.compile_trainer=false` is the first debug lever.
`--smoke` already runs uncompiled.

## Ablations

### Tau2 (`scripts/run_taubench.sh`)

| Mode           | Teacher hint                                      | Notes                                      |
| -------------- | ------------------------------------------------- | ------------------------------------------ |
| `gold`         | Sierra gold docs / canonical tool trajectory      | Main arm. No hint LLM. Cheap.              |
| `step_hint`    | OpenRouter names the single next correct action   | Gold + transcript in, one action out.      |
| `gold_banking` | Same as `gold`, `banking_knowledge` only          | Sandbox-stress arm.                        |
| `baseline`     | n/a                                               | Zero-shot pass^1 on ~87 held-out tasks.    |
| `smoke`        | gold, tiny model                                  | Sanity check.                              |

Eval metric: `pass1` overall and per domain. Binary, so eval "reward variance" is low by
construction. A fleet of zeros on banking is almost always the sandbox, not the loss.

### Diligence (`scripts/run_diligencebench.sh`)

Needs `data.dataset=diligence` and `data.n_heldout=30` (the script sets both). Parallel
`web_search` (`data.search_mode=fast`). Tool-result tokens are masked out of the SDPO
loss; agent tokens between searches are SOD-reweighted.

| Mode             | Teacher hint                                      | Notes                                      |
| ---------------- | ------------------------------------------------- | ------------------------------------------ |
| `answer_free`    | Must not state figures / conclusions              | Main arm. Prompt-enforced, no regex.       |
| `answer_bearing` | May cite missed rubric facts verbatim             | Stronger teacher; closer to distillation.  |
| `mixture`        | 50/50 KL mix of the two                           | Both hints must succeed or the rollout drops. |
| `baseline`       | n/a                                               | Zero-shot rubric-judge score, 30 held-out. |
| `smoke`          | answer_free, tiny model                           | Sanity check.                              |

Both diligence arms see the full rubric; only the output is constrained. A win on
`answer_bearing` measures something different from `answer_free` and should be reported
as such.

## Configuration

Defaults live in `train/config.py`. Trailing dotted CLI args override them
(`trainer.optimizer.learning_rate=1e-5`). Every field must already exist on the
dataclasses — `+new.field=...` is rejected.

| Setting            | Value                             | Notes                       |
| ------------------ | --------------------------------- | --------------------------- |
| model              | `Qwen/Qwen3.8-27B`                | smoke: `Qwen/Qwen3-0.6B`    |
| GPUs               | 4 rollout / 4 trainer (FSDP2)     | see below                   |
| batch / mini-batch | 16 / 4                            | must divide trainer world   |
| max staleness K    | 3                                 | blog + confirmed            |
| clip window        | `clip(r, 0.8, 1.4)`               | must contain 1.0            |
| advantage clip     | 3.0x EMA (decay 0.99)             | blog (3x); EMA spec is ours |
| group size         | 1, failures retained              | blog's headline finding     |
| KL penalty         | off                               | blog dropped it             |
| SOD                | on (`eps=1e-6`, `delta=0.2`)      | no-op on single-step trajs  |
| total steps        | 500                               | eval every 25               |

**GPU split.** Rollout GPUs come first (vLLM's multiprocess workers pick `cuda:0..TP-1`
by worker index); each trainer rank pins `cuda:{n_rollout_gpus + rank}`. 27B in bf16 does
not fit on one 80 GB H100, so the trainer is **FSDP2**, not DDP: each transformer block is
its own shard unit, then the root. 8B *would* fit as a full copy (~16 GB weights + 16 GB
grads + 33 GB Adam ≈ 65 GB, hence `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) —
if 27B OOMs after shrinking `mini_batch_size`, `model.model=Qwen/Qwen3-8B` is the fallback.
The hinted teacher gets **no dedicated GPU**: it is the same weights as the student, run
under `no_grad` on each trainer rank. That keeps TP=4 on the rollout side (a power of two).

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

On diligence (and tau2 `step_hint`), every hint is written **per rollout** by an LLM
(`stealth/ox-alpha` on OpenRouter) that reads the draft the student actually produced.
Two rollouts of the same task get different hints. Tau2 `gold` skips the LLM and injects
Sierra gold / the canonical tool trajectory instead.

A rollout whose hint cannot be generated is **dropped**, not trained with an empty hint:
an unhinted teacher is identical to the student and would contribute ~zero gradient. Watch
`store_hint_dropped_percent` (and `store.stats.hint_dropped`). A sustained nonzero value
means the hint model, not the rollout engine, is the bottleneck on data production.

## What to watch (and what to fix)

During a real run, in order:

1. **Teacher−student gap is clearly nonzero.** `trainer.py` logs
   `teacher_minus_student_logp` every step and warns when `|gap| < 1e-3`. That is a dead
   gradient. Typical causes: hint too timid (`answer_free`), hint LLM failing / empty,
   gold suffix not actually landing, sandbox producing empty transcripts, SOD + loss mask
   eating every token. Stronger teacher = `answer_bearing` / `gold`; do **not** add
   GRPO-style whitening or a baseline term.
2. **Clip fractions are a modest minority of tokens.** All-clipped means the off-policy
   correction is a no-op; all-unclipped with exploding ratios is the failure the clips
   exist to prevent.
3. **Staleness ≤ 3.** If the store is starving, rollout is too slow (sandbox, search,
   OpenRouter) or too many hints are dropping. A freeze after ~K steps with GPUs at 0%
   is the producer/consumer deadlock: the staleness manager must admit `batch_size`
   groups per step, not `mini_batch_size`. That is wired in `AsyncStalenessManager`.
4. **Held-out metric beats the zero-shot baseline.** Tau2: `pass1`. Diligence:
   `judge_score` plus `factual-accuracy` / `analytical-reasoning` / `risk-awareness`.

**Sandbox.** Two different failures, do not mix them up:

1. **Missing binaries** (`srt` / `rg` / `bwrap` / `socat`) — `SandboxRuntimeError` at
   env construction. Run `bash scripts/setup_tau2_sandbox.sh`.
2. **Namespaces blocked** (the usual GPU-pod case) — binaries exist, `which` is green,
   then `bwrap: Creating new namespace failed: Operation not permitted`. Recreate the
   pod `--privileged` / `seccomp=unconfined`. `run.py` probes this at startup and
   exits. Until the pod is recreated, isolate banking (`gold_banking`) rather than
   taking down the three-domain `gold` run; BM25/dense still work.

**Do not** flip the SDPO sign, change the clip window to exclude 1.0, filter
zero-variance groups (`group_size=1`, `keep_failures=True` is the blog finding), or
"simplify" weight-sync order (receive RPC must be in flight before `trainer_send_weights`).

## Verification

Everything through the trainer is CPU-testable — no GPU, no model downloads:

```bash
uv sync
uv run pytest tests/ -q
```

The test suite is fully offline. Only the judge / hint LLM / tau2 user sim / Parallel
search reach the network, and only on a real run.

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
- **Weight-sync deadlock ordering** — asserts the receive-side RPC is dispatched _before_
  the trainer broadcasts. Also mutation-verified: restoring the naive
  await-then-broadcast order makes the test hang and fail.

## Backends

The inference engine sits behind two protocols in `train/backends/backend.py`, so a new
backend is a new file plus a config value rather than an edit to the trainer:

| protocol | side | who holds it | vLLM implementation |
| --- | --- | --- | --- |
| `InferenceEngine` | receive | orchestrator (`run.py`) | `VLLMRolloutEngine` |
| `WeightTransport` | send | trainer (`train/trainer.py`) | `NCCLWeightTransport` |

They are **always chosen together** — `train.backends.get_backend(name)` returns the pair,
selected by `generator.engine.backend` (default `"vllm"`). The receiver derives its
broadcasts from metadata the sender chose, so a mismatched pair would hang the rendezvous
rather than fail loudly; pairing them at the selector makes that unrepresentable.

`train/backends/vllm.py` is the only module in the package that imports vLLM, and every
import there is lazy, so the package stays importable (and the full test suite runs) on a
laptop without vLLM.

Adding a backend means implementing both protocols. **Read the invariants at the top of
`train/backends/backend.py` first** — particularly processed-logprobs, which a new engine
can violate silently: SGLang's logprob semantics differ from vLLM's and must be verified,
not assumed.

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

## Deviations from the blog, and why

| Blog                              | Here                                     | Why                            |
| ---------------------------------- | ---------------------------------------- | ------------------------------ |
| Tau-Retail only                    | tau2 (banking+retail+airline) + diligence | both requested                 |
| Qwen3-8B DDP                       | Qwen3.8-27B FSDP2                        | 27B does not fit per-rank      |
| symmetric PPO clip ε=0.2           | DAPO decoupled 0.8 / 1.4                 | requested                      |
| hint = canonical answer            | gold / step_hint / answer_free|bearing   | per-dataset ablations          |
| no reward model at all             | rubric judge + tau2 pass^1, **eval only** | neither is in the gradient    |

## References

- [Scaling SDPO](https://trajectory.ai/field-notes/scaling-sdpo) — Trajectory, June 2026
- [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802) —
  Hübotter et al., arXiv:2601.20802
- [`lasgroup/SDPO`](https://github.com/lasgroup/SDPO) — official implementation
- [tau2-bench](https://github.com/sierra-research/tau2-bench) — Sierra
