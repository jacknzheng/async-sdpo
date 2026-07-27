"""Turning a list of trajectories into padded tensors for the SDPO loss.

Pure tensor/tokenizer plumbing, separated from trainer.py so it can be unit-tested on CPU
without a model, a GPU, or vLLM.

The subtle part is the teacher forward pass. The student and teacher share weights but see
different contexts: the student's prompt is the bare query, the teacher's has the hint
prepended. Those prompts have different lengths, so the response tokens sit at different
offsets in the two sequences. Aligning student and teacher log-probs on the SAME response
tokens is what makes the per-token KL meaningful -- an off-by-one here silently compares
each student token against the teacher's distribution for a neighbouring position, which
produces a plausible-looking loss curve and trains nothing useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from train.store import Trajectory


@dataclass
class SDPOBatch:
    """Padded tensors for one training batch.

    `student_input_ids` and `teacher_input_ids` hold different sequences (the teacher's has
    the hint prepended), but their `*_response_mask`s select the same response tokens, so
    gathered log-probs align 1:1 across the two.
    """

    student_input_ids: torch.Tensor      # [B, S]
    student_attention_mask: torch.Tensor # [B, S]
    student_response_mask: torch.Tensor  # [B, S] 1 on response tokens
    teacher_input_ids: torch.Tensor      # [B, T]
    teacher_attention_mask: torch.Tensor # [B, T]
    teacher_response_mask: torch.Tensor  # [B, T] 1 on the SAME response tokens
    rollout_logprobs: torch.Tensor       # [B, R] captured at generation time
    response_mask: torch.Tensor          # [B, R] 1 on real response tokens, 0 on padding
    task_ids: list[str]
    policy_versions: torch.Tensor        # [B] for staleness diagnostics

    def to(self, device: torch.device | str) -> SDPOBatch:
        move = lambda t: t.to(device)  # noqa: E731
        return SDPOBatch(
            student_input_ids=move(self.student_input_ids),
            student_attention_mask=move(self.student_attention_mask),
            student_response_mask=move(self.student_response_mask),
            teacher_input_ids=move(self.teacher_input_ids),
            teacher_attention_mask=move(self.teacher_attention_mask),
            teacher_response_mask=move(self.teacher_response_mask),
            rollout_logprobs=move(self.rollout_logprobs),
            response_mask=move(self.response_mask),
            task_ids=self.task_ids,
            policy_versions=move(self.policy_versions),
        )


def _pad(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max((len(s) for s in sequences), default=0)
    return torch.tensor(
        [s + [pad_value] * (width - len(s)) for s in sequences], dtype=torch.long
    )


def build_batch(
    trajectories: list[Trajectory],
    teacher_prompt_ids: list[list[int]],
    pad_token_id: int,
) -> SDPOBatch:
    """Assemble padded student/teacher tensors from finished trajectories.

    `teacher_prompt_ids[i]` is the hint-prefixed prompt for `trajectories[i]`, tokenized by
    the caller (trainer.py owns the tokenizer). The same response tokens are appended to
    both prompts.
    """
    if len(trajectories) != len(teacher_prompt_ids):
        raise ValueError(
            f"{len(trajectories)} trajectories but {len(teacher_prompt_ids)} teacher prompts"
        )

    student_seqs, teacher_seqs = [], []
    student_starts, teacher_starts, response_lens = [], [], []

    for trajectory, teacher_prompt in zip(trajectories, teacher_prompt_ids):
        response = trajectory.response_token_ids
        student_starts.append(len(trajectory.prompt_token_ids))
        teacher_starts.append(len(teacher_prompt))
        response_lens.append(len(response))
        student_seqs.append(list(trajectory.prompt_token_ids) + list(response))
        teacher_seqs.append(list(teacher_prompt) + list(response))

    student_input_ids = _pad(student_seqs, pad_token_id)
    teacher_input_ids = _pad(teacher_seqs, pad_token_id)

    student_attention_mask = torch.zeros_like(student_input_ids)
    teacher_attention_mask = torch.zeros_like(teacher_input_ids)
    student_response_mask = torch.zeros_like(student_input_ids)
    teacher_response_mask = torch.zeros_like(teacher_input_ids)

    for i, (s_start, t_start, r_len) in enumerate(
        zip(student_starts, teacher_starts, response_lens)
    ):
        student_attention_mask[i, : s_start + r_len] = 1
        teacher_attention_mask[i, : t_start + r_len] = 1
        student_response_mask[i, s_start : s_start + r_len] = 1
        teacher_response_mask[i, t_start : t_start + r_len] = 1

    max_response = max(response_lens, default=0)
    rollout_logprobs = torch.zeros(len(trajectories), max_response)
    response_mask = torch.zeros(len(trajectories), max_response)
    for i, trajectory in enumerate(trajectories):
        n = len(trajectory.rollout_logprobs)
        rollout_logprobs[i, :n] = torch.tensor(trajectory.rollout_logprobs)
        response_mask[i, :n] = 1.0

    return SDPOBatch(
        student_input_ids=student_input_ids,
        student_attention_mask=student_attention_mask,
        student_response_mask=student_response_mask,
        teacher_input_ids=teacher_input_ids,
        teacher_attention_mask=teacher_attention_mask,
        teacher_response_mask=teacher_response_mask,
        rollout_logprobs=rollout_logprobs,
        response_mask=response_mask,
        task_ids=[t.task_id for t in trajectories],
        policy_versions=torch.tensor([t.policy_version for t in trajectories]),
    )


def gather_response_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    max_response_len: int,
) -> torch.Tensor:
    """Extract per-token log-probs of the realized response tokens, left-packed to [B, R].

    Causal LM shift: `logits[:, t]` predicts `input_ids[:, t+1]`, so the log-prob of the
    token at position t comes from position t-1's logits. Getting this shift wrong is the
    other classic silent bug in this file -- it also trains, and also learns nothing.

    Returns [B, max_response_len] with each row left-packed so column j is the j-th
    response token, regardless of where the response sits in the padded sequence. That
    packing is what lets student and teacher log-probs be compared elementwise even though
    their prompts differ in length.
    """
    log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_logprobs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # response_mask marks response positions in the *unshifted* sequence; after the shift,
    # position t of token_logprobs corresponds to input position t+1.
    shifted_mask = response_mask[:, 1:]

    batch_size = logits.size(0)
    packed = torch.zeros(
        batch_size, max_response_len, dtype=token_logprobs.dtype, device=logits.device
    )
    for i in range(batch_size):
        selected = token_logprobs[i][shifted_mask[i].bool()]
        n = min(selected.numel(), max_response_len)
        packed[i, :n] = selected[:n]
    return packed
