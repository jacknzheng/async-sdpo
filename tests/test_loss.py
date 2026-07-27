"""CPU-only tests for the SDPO loss. No GPU, no network, no model downloads.

The important one is `test_gradient_moves_toward_teacher`. A flipped sign in the loss does
not crash and does not look wrong on a loss curve -- it just trains the model to reinforce
the tokens the teacher disagrees with. That test is the only thing standing between us and
a silently inverted training run.
"""

import torch
import torch.nn.functional as F

from train.loss import (
    EMAAdvantageClipper,
    clip_ratio,
    importance_ratio,
    masked_mean,
    sdpo_advantage,
    sdpo_loss,
)


def test_advantage_sign_is_teacher_minus_student():
    """Advantage is positive exactly when the teacher likes the token more."""
    student = torch.tensor([[-2.0, -1.0]])
    teacher = torch.tensor([[-1.0, -3.0]])  # likes token 0 more, token 1 less
    adv = sdpo_advantage(student, teacher)
    assert adv[0, 0] > 0
    assert adv[0, 1] < 0
    assert not adv.requires_grad, "advantage must be detached"


def test_gradient_moves_toward_teacher():
    """THE critical test: one optimizer step must RAISE the student's logprob on the
    token the teacher prefers, and LOWER it on the token the teacher dislikes.

    Built as a real optimization on real logits rather than an assertion about gradient
    signs, so it catches a flipped sign anywhere in the chain (advantage, negation, or
    ratio), not just in one function.
    """
    torch.manual_seed(0)
    vocab, favored, disfavored = 8, 3, 5

    logits = torch.zeros(1, 2, vocab, requires_grad=True)
    actions = torch.tensor([[favored, disfavored]])

    # Teacher strongly prefers `favored` at position 0 and strongly dislikes `disfavored`
    # at position 1. Held fixed across the step.
    teacher_lp = torch.tensor([[-0.05, -6.0]])
    mask = torch.ones(1, 2)

    def student_logprobs_from(lg):
        return torch.gather(F.log_softmax(lg, dim=-1), 2, actions.unsqueeze(-1)).squeeze(-1)

    before = student_logprobs_from(logits).detach().clone()

    opt = torch.optim.SGD([logits], lr=1.0)
    loss, metrics = sdpo_loss(
        student_log_probs=student_logprobs_from(logits),
        teacher_log_probs=teacher_lp,
        rollout_log_probs=None,  # on-policy: isolate the SDPO term
        response_mask=mask,
        clipper=None,
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    after = student_logprobs_from(logits).detach()

    assert after[0, 0] > before[0, 0], (
        "SIGN ERROR: student logprob DECREASED on the token the teacher prefers. "
        "The advantage/loss negation is inverted -- see the sign block in train/loss.py."
    )
    assert after[0, 1] < before[0, 1], (
        "SIGN ERROR: student logprob INCREASED on the token the teacher dislikes."
    )
    assert metrics["teacher_minus_student_logp"] != 0.0


def test_matches_reference_implementation():
    """Our loss must equal the reference form `log_ratio.detach() * student_log_probs`
    (lasgroup/SDPO and TRL, which agree byte-for-byte), up to the mean aggregation.
    """
    torch.manual_seed(0)
    student = torch.randn(4, 6).requires_grad_(True)
    teacher = torch.randn(4, 6)
    mask = torch.ones(4, 6)

    log_ratio = (student - teacher).detach()
    reference = masked_mean(log_ratio * student, mask)

    ours, _ = sdpo_loss(student, teacher, None, mask, clipper=None)
    torch.testing.assert_close(ours, reference)


def test_zero_loss_when_teacher_equals_student():
    """Reverse KL of a distribution with itself is 0, so the advantage vanishes."""
    torch.manual_seed(0)
    lp = torch.randn(2, 5).requires_grad_(True)
    mask = torch.ones(2, 5)
    loss, metrics = sdpo_loss(lp, lp.detach().clone(), None, mask, clipper=None)
    torch.testing.assert_close(loss, torch.zeros(()))
    assert abs(metrics["teacher_minus_student_logp"]) < 1e-6


def test_masked_tokens_do_not_contribute():
    """Prompt/padding positions must not affect the loss regardless of their values."""
    torch.manual_seed(0)
    student = torch.randn(1, 4).requires_grad_(True)
    teacher = torch.randn(1, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    loss_a, _ = sdpo_loss(student, teacher, None, mask, clipper=None)

    poisoned = teacher.clone()
    poisoned[0, 2:] = 1e4  # garbage in the masked-out region
    loss_b, _ = sdpo_loss(student, poisoned, None, mask, clipper=None)

    torch.testing.assert_close(loss_a, loss_b)


# ---- DAPO ratio clipping ----

def test_clip_window_contains_one():
    """A ratio of exactly 1.0 (token probability unchanged) must pass through untouched.
    This is the property a [1.2, 1.4] window would violate.
    """
    r = torch.tensor([1.0])
    torch.testing.assert_close(clip_ratio(r, 0.8, 1.4), torch.tensor([1.0]))


def test_extreme_ratios_clip_to_bounds():
    """The 50x-100x blowups that collapse naive off-policy SDPO land on the bounds."""
    r = torch.tensor([50.0, 100.0, 0.01, 1.0, 1.2])
    out = clip_ratio(r, 0.8, 1.4)
    torch.testing.assert_close(out, torch.tensor([1.4, 1.4, 0.8, 1.0, 1.2]))


def test_one_sided_clip_matches_reference():
    """Reference SDPO clamps max only, leaving small ratios alone."""
    r = torch.tensor([50.0, 0.01])
    out = clip_ratio(r, 0.8, 1.4, one_sided=True, one_sided_max=2.0)
    torch.testing.assert_close(out, torch.tensor([2.0, 0.01]))


def test_importance_ratio_is_one_when_policy_unchanged():
    lp = torch.randn(3, 4)
    torch.testing.assert_close(importance_ratio(lp, lp.clone()), torch.ones(3, 4))


def test_importance_ratio_does_not_overflow():
    """Pathological log-differences must not produce inf and poison the batch."""
    student = torch.tensor([[500.0, -500.0]])
    rollout = torch.tensor([[-500.0, 500.0]])
    r = importance_ratio(student, rollout)
    assert torch.isfinite(r).all()


# ---- EMA advantage clipping ----

def test_ema_converges_to_constant_stream():
    """Fed a constant mean |A|, the bias-corrected EMA converges to it and the bound to 3x."""
    clipper = EMAAdvantageClipper(mult=3.0, decay=0.99, bias_correction=True)
    adv = torch.full((4, 8), 2.0)
    mask = torch.ones(4, 8)
    for _ in range(500):
        bound = clipper.update(adv, mask)
    assert abs(clipper.ema / (1 - 0.99**clipper.step) - 2.0) < 0.05
    assert abs(bound - 6.0) < 0.15


def test_bias_correction_matters_early():
    """Without correction the bound is far too tight on step 1 -- exactly when training is
    least stable. With it, the first step already reflects the observed magnitude.
    """
    adv, mask = torch.full((1, 4), 2.0), torch.ones(1, 4)

    corrected = EMAAdvantageClipper(mult=3.0, decay=0.99, bias_correction=True)
    uncorrected = EMAAdvantageClipper(mult=3.0, decay=0.99, bias_correction=False)
    b_corrected = corrected.update(adv, mask)
    b_uncorrected = uncorrected.update(adv, mask)

    assert abs(b_corrected - 6.0) < 1e-6
    assert b_uncorrected < 0.1  # 3 * 0.01 * 2.0
    assert b_corrected > b_uncorrected * 50


def test_no_clipping_before_first_update():
    """With no data yet the bound is infinite, so nothing is clipped."""
    clipper = EMAAdvantageClipper()
    adv = torch.tensor([[1e6, -1e6]])
    assert clipper.bound() == float("inf")
    torch.testing.assert_close(clipper.clip(adv), adv)


def test_clip_bounds_outliers_symmetrically():
    clipper = EMAAdvantageClipper(mult=3.0, decay=0.99, bias_correction=True)
    clipper.update(torch.full((1, 4), 1.0), torch.ones(1, 4))  # bound -> 3.0
    out = clipper.clip(torch.tensor([[100.0, -100.0, 1.0]]))
    torch.testing.assert_close(out, torch.tensor([[3.0, -3.0, 1.0]]))


def test_clipper_state_survives_checkpoint_roundtrip():
    """The EMA is training state, not a parameter -- resuming without it would silently
    reset the clip bound and re-introduce the early-training tightness.
    """
    clipper = EMAAdvantageClipper(mult=3.0, decay=0.99)
    for _ in range(17):
        clipper.update(torch.full((2, 3), 1.5), torch.ones(2, 3))

    restored = EMAAdvantageClipper()
    restored.load_state_dict(clipper.state_dict())

    assert restored.step == clipper.step == 17
    assert restored.ema == clipper.ema
    assert restored.bound() == clipper.bound()


def test_sync_overwrites_ema_for_rank_agreement():
    """All-reduced EMA must be adoptable, or ranks clip differently and diverge."""
    clipper = EMAAdvantageClipper(decay=0.99)
    clipper.update(torch.full((1, 2), 1.0), torch.ones(1, 2))
    clipper.sync_(0.5)
    assert clipper.ema == 0.5


# ---- integration of the two clips ----

def test_metrics_report_clip_fractions():
    """Both clip fractions are reported, since a high fraction is the signal that
    staleness or the learning rate is too aggressive.
    """
    torch.manual_seed(0)
    student = torch.randn(2, 5).requires_grad_(True)
    teacher = torch.randn(2, 5)
    rollout = student.detach() - 5.0  # forces large ratios
    mask = torch.ones(2, 5)

    _, metrics = sdpo_loss(
        student, teacher, rollout, mask,
        clipper=EMAAdvantageClipper(), clip_low=0.8, clip_high=1.4,
    )
    assert metrics["ratio_clip_frac_high"] > 0.9
    assert "adv_clip_bound" in metrics and "teacher_minus_student_logp" in metrics


def test_loss_is_finite_under_extreme_staleness():
    """The whole point of the two clips: heavy-tailed ratios must not produce inf/nan."""
    torch.manual_seed(0)
    student = torch.randn(4, 16).requires_grad_(True)
    teacher = torch.randn(4, 16) * 10
    rollout = torch.randn(4, 16) * 10
    mask = torch.ones(4, 16)

    loss, _ = sdpo_loss(student, teacher, rollout, mask, clipper=EMAAdvantageClipper())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
