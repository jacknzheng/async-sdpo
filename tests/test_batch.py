"""Tests for batch assembly and log-prob alignment. CPU-only, no model download.

Two silent-failure modes are covered here, both of which produce a plausible loss curve
while training nothing useful:

1. The causal shift: logits[:, t] predicts input_ids[:, t+1]. Off by one and every token
   is scored against its neighbour's distribution.
2. Student/teacher alignment: the teacher's prompt is longer (it carries the hint), so the
   same response token sits at different absolute positions in the two sequences. If the
   packing doesn't correct for that, the per-token KL compares mismatched positions.
"""

import pytest
import torch

from train.utils import build_batch, gather_response_logprobs

# NOTE: the chunked `_TargetLogProbs` autograd function is gone -- gather_response_logprobs
# now computes response log-probs directly. The two tests that exercised its chunking and
# bf16->fp32 behaviour were removed with it.
from train.store import Trajectory


def _traj(task_id="1", prompt=(1, 2, 3), response=(4, 5), version=0) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        prompt_token_ids=list(prompt),
        response_token_ids=list(response),
        rollout_logprobs=[-0.5] * len(response),
        policy_version=version,
    )


def test_student_and_teacher_masks_select_same_token_count():
    """Different prompt lengths, same number of response tokens selected."""
    trajectories = [_traj(prompt=(1, 2, 3), response=(7, 8, 9))]
    teacher_prompts = [[90, 91, 92, 93, 94, 95]]  # hint makes it longer
    batch = build_batch(trajectories, teacher_prompts, pad_token_id=0)

    assert batch.student_response_mask.sum() == batch.teacher_response_mask.sum() == 3


def test_response_tokens_land_at_correct_offsets():
    trajectories = [_traj(prompt=(1, 2, 3), response=(7, 8))]
    teacher_prompts = [[90, 91, 92, 93, 94]]
    batch = build_batch(trajectories, teacher_prompts, pad_token_id=0)

    student_selected = batch.student_input_ids[batch.student_response_mask.bool()]
    teacher_selected = batch.teacher_input_ids[batch.teacher_response_mask.bool()]
    assert student_selected.tolist() == [7, 8]
    assert teacher_selected.tolist() == [7, 8]


def test_padding_is_masked_out():
    trajectories = [_traj(response=(1, 2, 3, 4)), _traj(task_id="2", response=(5,))]
    teacher_prompts = [[9], [9]]
    batch = build_batch(trajectories, teacher_prompts, pad_token_id=0)

    assert batch.response_mask[0].sum() == 4
    assert batch.response_mask[1].sum() == 1
    assert batch.rollout_logprobs[1, 1:].abs().sum() == 0  # padded region is zeroed


def test_rollout_logprobs_are_preserved_verbatim():
    """These come from the rollout engine and must not be recomputed or reordered."""
    trajectory = _traj(response=(4, 5, 6))
    trajectory.rollout_logprobs = [-0.1, -0.2, -0.3]
    batch = build_batch([trajectory], [[9]], pad_token_id=0)
    assert batch.rollout_logprobs[0, :3].tolist() == pytest.approx([-0.1, -0.2, -0.3])


def test_mismatched_teacher_prompt_count_raises():
    with pytest.raises(ValueError, match="teacher prompts"):
        build_batch([_traj(), _traj()], [[1]], pad_token_id=0)


def test_gather_recovers_known_logprobs_exactly():
    """Build logits whose log-softmax is known, then assert we extract the right entries.

    This is the causal-shift test: the log-prob of response token at position t must come
    from the logits at position t-1.
    """
    vocab, seq = 10, 5
    torch.manual_seed(0)
    logits = torch.randn(1, seq, vocab)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    # Response occupies the last two positions (tokens 4 and 5).
    response_mask = torch.tensor([[0, 0, 0, 1, 1]])

    packed = gather_response_logprobs(logits, input_ids, response_mask, max_response_len=2)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = torch.tensor([
        log_probs[0, 2, 4],  # logits at pos 2 predict token at pos 3 (id 4)
        log_probs[0, 3, 5],  # logits at pos 3 predict token at pos 4 (id 5)
    ])
    torch.testing.assert_close(packed[0], expected)


def test_gather_is_left_packed_regardless_of_offset():
    """The same response at different prompt offsets must pack to the same columns --
    this is what makes student and teacher log-probs elementwise comparable.
    """
    vocab = 12
    torch.manual_seed(0)
    shared = torch.randn(1, 3, vocab)  # logits covering the response region

    # Short prompt: response at positions 1-2.
    short_logits = torch.cat([torch.randn(1, 1, vocab), shared], dim=1)
    short_ids = torch.tensor([[1, 7, 8, 9]])
    short_mask = torch.tensor([[0, 1, 1, 1]])

    # Long prompt: same response, positions 3-5.
    long_logits = torch.cat([torch.randn(1, 3, vocab), shared], dim=1)
    long_ids = torch.tensor([[1, 2, 3, 7, 8, 9]])
    long_mask = torch.tensor([[0, 0, 0, 1, 1, 1]])

    # Make the predicting-position logits identical across the two so the extracted
    # logprobs must match if (and only if) the offset handling is right.
    long_logits[0, 2] = short_logits[0, 0]

    short_packed = gather_response_logprobs(short_logits, short_ids, short_mask, 3)
    long_packed = gather_response_logprobs(long_logits, long_ids, long_mask, 3)

    torch.testing.assert_close(short_packed, long_packed)


def test_gather_handles_ragged_batch():
    vocab = 8
    torch.manual_seed(0)
    logits = torch.randn(2, 6, vocab)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 0, 0, 0]])
    response_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0]])

    packed = gather_response_logprobs(logits, input_ids, response_mask, max_response_len=4)
    assert packed.shape == (2, 4)
    assert packed[1, 1:].abs().sum() == 0  # shorter row zero-padded


def test_end_to_end_alignment_with_real_model():
    """Full path on a real (tiny, randomly-initialized) causal LM: build a batch, run both
    forward passes, gather, and confirm student/teacher log-probs are comparable and that
    an identical hint reproduces identical log-probs.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()

    trajectories = [_traj(prompt=(1, 2, 3), response=(10, 11, 12))]
    teacher_prompts = [[20, 21, 22, 23, 24, 1, 2, 3]]  # hint + same query
    batch = build_batch(trajectories, teacher_prompts, pad_token_id=0)

    with torch.no_grad():
        student_logits = model(
            input_ids=batch.student_input_ids, attention_mask=batch.student_attention_mask
        ).logits
        teacher_logits = model(
            input_ids=batch.teacher_input_ids, attention_mask=batch.teacher_attention_mask
        ).logits

    max_r = batch.response_mask.size(1)
    student_lp = gather_response_logprobs(
        student_logits, batch.student_input_ids, batch.student_response_mask, max_r
    )
    teacher_lp = gather_response_logprobs(
        teacher_logits, batch.teacher_input_ids, batch.teacher_response_mask, max_r
    )

    assert student_lp.shape == teacher_lp.shape == (1, 3)
    assert torch.isfinite(student_lp).all() and torch.isfinite(teacher_lp).all()
    # Different contexts => different distributions (the gradient signal exists at all).
    assert not torch.allclose(student_lp, teacher_lp)


def test_identical_contexts_give_identical_logprobs():
    """Sanity check on the alignment: if the teacher prompt equals the student prompt, the
    two gathers must agree exactly. If they don't, the offset handling is wrong.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "llama", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).eval()

    trajectories = [_traj(prompt=(1, 2, 3), response=(10, 11))]
    batch = build_batch(trajectories, [[1, 2, 3]], pad_token_id=0)  # same prompt

    with torch.no_grad():
        s = model(input_ids=batch.student_input_ids,
                  attention_mask=batch.student_attention_mask).logits
        t = model(input_ids=batch.teacher_input_ids,
                  attention_mask=batch.teacher_attention_mask).logits

    max_r = batch.response_mask.size(1)
    student_lp = gather_response_logprobs(
        s, batch.student_input_ids, batch.student_response_mask, max_r)
    teacher_lp = gather_response_logprobs(
        t, batch.teacher_input_ids, batch.teacher_response_mask, max_r)

    torch.testing.assert_close(student_lp, teacher_lp)
