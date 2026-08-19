from __future__ import annotations

from dataclasses import dataclass

import torch

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over positions where `mask` is 1, guarding against an empty mask."""
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def importance_ratio(
    student_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Off-policy IS ratio r_t = pi_current(a_t) / pi_rollout(a_t).

    `rollout_log_probs` must be the log-probs recorded by the rollout engine AT
    GENERATION TIME under policy pi_{theta-K}. Recomputing them under current weights
    would silently make every ratio 1.0 and defeat the entire off-policy correction.

    The log-difference is clamped to +/-20 before exponentiating (matching the reference
    implementation) purely to avoid inf/overflow on pathological tokens; the real bounding
    is done by the clip below.
    """
    log_diff = (student_log_probs - rollout_log_probs).detach()
    return torch.exp(log_diff.clamp(min=-20.0, max=20.0))


def clip_ratio(
    ratio: torch.Tensor,
    clip_low: float,
    clip_high: float,
    one_sided: bool = False,
    one_sided_max: float = 2.0,
) -> torch.Tensor:
    """DAPO decoupled clipping of the IS ratio.

    The ratio is centered at 1.0, so the window must straddle 1.0 -- a window like
    [1.2, 1.4] would clip even tokens whose probability did not change at all. DAPO's
    contribution is making the window ASYMMETRIC (clip-higher): a larger upper bound
    leaves more room for probability mass to grow on tokens the teacher favors, while the
    tighter lower bound still contains the rare-token blowups that collapse naive IS.

    `one_sided` reproduces the reference SDPO implementations instead, which clamp only
    the maximum on the distillation branch.
    """
    if one_sided:
        return ratio.clamp(max=one_sided_max)
    return ratio.clamp(min=clip_low, max=clip_high)


@dataclass
class EMAAdvantageClipper:
    """Bounds per-token advantage at a fixed multiple of its running mean magnitude, prevents advantage explosion from rogue tokens.

        batch_mean = masked_mean(|A_t|)                         # this step, response tokens
        ema        = decay * ema + (1 - decay) * batch_mean
        ema_hat    = ema / (1 - decay ** step)                  # bias correction
        A_clipped  = clamp(A_t, -mult * ema_hat, +mult * ema_hat)
    """

    mult: float = 3.0
    decay: float = 0.99
    bias_correction: bool = True
    ema: float = 0.0
    step: int = 0

    def bound(self) -> float:
        """Current clip bound, or inf before the first update (no data yet -> no clipping)"""
        if self.step == 0:
            return float("inf")
        ema_hat = self.ema
        if self.bias_correction:
            ema_hat = self.ema / (1.0 - self.decay**self.step) # in early steps, 0.99 ** 0 is large, which counteracts self.ema being very small (since its biased and initialized at 0)
            # INTUITION: self.ema = 0.01 / (1-(0.99)^1) = 1, this makes our self.ema less biased towards zero
            # eventually ema_hat approaches ema, as the denom -> 1
        return self.mult * ema_hat

    def update(self, advantages: torch.Tensor, mask: torch.Tensor) -> float:
        """Fold this batch's mean |A| into the EMA and return the new clip bound."""
        batch_mean = masked_mean(advantages.abs(), mask).item()
        self.ema = self.decay * self.ema + (1.0 - self.decay) * batch_mean # 0.99 * old_advantages + 0.01 * new_advantages = EMA advantage
        self.step += 1
        return self.bound()

    def clip(self, advantages: torch.Tensor) -> torch.Tensor:
        b = self.bound()
        if b == float("inf"):
            return advantages
        return advantages.clamp(min=-b, max=b) # clip advantages according to our current EMA

    def sync_(self, ema: float) -> None:
        """Overwrite the EMA with an all-reduced value so every rank clips identically."""
        self.ema = ema

    def state_dict(self) -> dict:
        return {"mult": self.mult, "decay": self.decay,
                "bias_correction": self.bias_correction, "ema": self.ema, "step": self.step}

    def load_state_dict(self, state: dict) -> None:
        self.mult = state["mult"]
        self.decay = state["decay"]
        self.bias_correction = state["bias_correction"]
        self.ema = state["ema"]
        self.step = state["step"]


def sdpo_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor | None,
    response_mask: torch.Tensor,
    clipper: EMAAdvantageClipper | None = None,
    clip_low: float = 0.8,
    clip_high: float = 1.4,
    one_sided: bool = False,
    one_sided_max: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Off-policy SDPO loss for one batch.

    Args:
        student_log_probs: [B, T] log pi_current(a_t|s_t). REQUIRES GRAD -- this is the
            only tensor gradient flows through.
        teacher_log_probs: [B, T] log pi_current(a_t|s_t, c), hinted, no grad.
        rollout_log_probs: [B, T] log pi_rollout(a_t|s_t) captured at generation time.
            None means on-policy (K=0), which skips the IS correction entirely.
        response_mask: [B, T] 1 on generated tokens, 0 on prompt/padding.
        clipper: EMA advantage clipper, updated in place. None disables advantage clipping.

    Returns:
        (scalar loss, metrics dict). `metrics["teacher_minus_student_logp"]` is the
        diagnostic to watch: it is the mean advantage before clipping, and if it sits near
        zero the hint is too weak for the teacher to beat the student, so the gradient
        vanishes and training will not move regardless of what the loss curve looks like.
    """
    advantages = (teacher_log_probs - student_log_probs).detach()

    metrics: dict[str, float] = {
        "teacher_minus_student_logp": masked_mean(advantages, response_mask).item(),
        "adv_abs_mean": masked_mean(advantages.abs(), response_mask).item(),
    }

    if clipper is not None:
        bound = clipper.update(advantages, response_mask) # update based on new advantages
        clipped_adv = clipper.clip(advantages) # then clip the advantages
        was_clipped = (advantages.abs() > bound).float() # all tokens where advantage was clipped
        metrics["adv_clip_bound"] = bound
        metrics["adv_clip_frac"] = masked_mean(was_clipped, response_mask).item()
        metrics["adv_ema"] = clipper.ema
        advantages = clipped_adv

    # produces a surrogate loss objective - the gradient of which we use monte carlo sampling to approximate the dense KL
    # this basically means we don't need to materialize the full [B, T, vocab_size] tensor which would be enormous! 
    per_token_loss = -advantages * student_log_probs

    if rollout_log_probs is not None:
        ratio = importance_ratio(student_log_probs, rollout_log_probs)
        clipped_ratio = clip_ratio(ratio, clip_low, clip_high, one_sided, one_sided_max)
        low_frac = (ratio < clip_low).float()
        high_frac = (ratio > clip_high).float()
        metrics["ratio_mean"] = masked_mean(ratio, response_mask).item()
        metrics["ratio_clip_frac_low"] = masked_mean(low_frac, response_mask).item()
        metrics["ratio_clip_frac_high"] = masked_mean(high_frac, response_mask).item()
        per_token_loss = per_token_loss * clipped_ratio

    # surrogate gradient = 1/n ∑ d(log student) (log student - log teacher) - summing over just all tokens in the seq
    # dense gradient = ∑ ∑ d(log student) (log student - log teacher) - summing over all tokens + log_probs of all possible tokens in the seq
    loss = masked_mean(per_token_loss, response_mask) # masked mean averages the per_token_loss over the actual unmasked tokens (i.e. ignoring prompt tokens)
    metrics["loss"] = loss.item()
    metrics["n_response_tokens"] = response_mask.sum().item()
    return loss, metrics
