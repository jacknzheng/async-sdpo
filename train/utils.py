from __future__ import annotations

import torch
from dataclasses import dataclass
from train.models import Trajectory

@dataclass
class SDPOBatch: 

    student_input_ids: torch.Tensor # (batch, num_prompt_tokens+num_response_tokens)
    student_attention_mask: torch.Tensor # (batch, num_prompt_tokens+num_response_tokens)
    student_response_mask: torch.Tensor # (batch, num_prompt_tokens+num_response_tokens) but 1 on response tokens
    teacher_input_ids: torch.Tensor # (batch, privileged info + prompt tokens + response tokens)
    teacher_attention_mask: torch.Tensor # (batch, privileged info + prompt tokens + response tokens)
    teacher_response_mask: torch.Tensor # (batch, privileged info + prompt tokens + response tokens) but 1 on response tokens
    rollout_logprobs: torch.Tensor # (batch, response tokens) <- used for KL between rollout and actor models
    response_mask: torch.Tensor # (batch, response tokens)
    task_ids: list[str] 
    policy_versions: torch.Tensor # (batch,) staleness
    # Packed step index per trained token. -1 on pad. All 0 when step_spans is empty
    # (single-turn / no TIR) so SOD is a no-op.
    step_ids: torch.Tensor | None = None
    # Second teacher (answer-bearing) for the mixture arm. None for single-hint runs.
    teacher_bearing_input_ids: torch.Tensor | None = None
    teacher_bearing_attention_mask: torch.Tensor | None = None
    teacher_bearing_response_mask: torch.Tensor | None = None

    def to(self, device:torch.device):
        # move batch to GPU
        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return SDPOBatch(
            student_input_ids=self.student_input_ids.to(device),
            student_attention_mask=self.student_attention_mask.to(device),
            student_response_mask=self.student_response_mask.to(device),
            teacher_input_ids=self.teacher_input_ids.to(device),
            teacher_attention_mask=self.teacher_attention_mask.to(device),
            teacher_response_mask=self.teacher_response_mask.to(device),
            rollout_logprobs=self.rollout_logprobs.to(device),
            response_mask=self.response_mask.to(device),
            task_ids=self.task_ids,  # list[str] -- nothing to move
            policy_versions=self.policy_versions.to(device),
            step_ids=_move(self.step_ids),
            teacher_bearing_input_ids=_move(self.teacher_bearing_input_ids),
            teacher_bearing_attention_mask=_move(self.teacher_bearing_attention_mask),
            teacher_bearing_response_mask=_move(self.teacher_bearing_response_mask),
        )
        

def packed_step_ids(
    loss_mask: list[int],
    step_spans: list[tuple[int, int]],
    n_tokens: int,
) -> list[int]:
    """Step index for each trained token, in the same packed order as rollout logprobs.

    `step_spans` index the unpacked response (sampled + injected). Injected positions
    (`loss_mask=0`) are dropped, matching `gather_response_logprobs`. Empty spans mean
    the whole trained sequence is step 0.
    """
    loss = list(loss_mask) if loss_mask else [1] * n_tokens
    if len(loss) != n_tokens:
        raise ValueError(f"loss_mask length {len(loss)} != {n_tokens} response tokens")
    span_of = [-1] * n_tokens
    for k, (start, end) in enumerate(step_spans):
        for j in range(max(start, 0), min(end, n_tokens)):
            span_of[j] = k
    ids: list[int] = []
    for j, mask in enumerate(loss):
        if int(mask) != 1:
            continue
        step = span_of[j]
        ids.append(0 if step < 0 else step)
    return ids


def _pad(sequences: list[list[int]], pad_value:int) -> torch.Tensor: 
    # pad up to the longest seq_len


    # find the longest sequence in the batch
    longest_seq_len = max((len(s) for s in sequences), default=0)
    
    # pad up to the longest sequence (i.e. longest sequence length), so [current token_ids] + [pad tokens]
    return torch.tensor(
        [sequence + [pad_value] * (longest_seq_len - len(sequence)) for sequence in sequences], dtype=torch.long
    )

def build_batch(trajectories: list[Trajectory], teacher_prompt_ids: list[list[int]], pad_token_id:int): 
    
    """
    Trajectories: trajectories we fetch from store.py using the batch() function, returns returns a list of the Trajectory object that comes from the rollout engine
    Teacher_prompt_ids: hint-prefixed prompt
    Pad_token_id: for when we pad to create the attention_mask

    Because the trainer owns the tokenizer, that means teacher_prompt_ids must be passed in (prompt_token_ids + tokenized(hint))
    """
    # need to create masking for student and teacher

    if len(trajectories) != len(teacher_prompt_ids): # the number of trajectories = the number of teacher prompts should match
        raise ValueError(
            f"{len(trajectories)} trajectories but {len(teacher_prompt_ids)} teacher prompts"
        )
    
    # first define where the masking indexes go per batch
    # attention mask = masking out the padding tokens - for forward passing on the student rollout by the teacher
    # response mask = masking out the prompt and padding tokens - for training on
    student_seqs, teacher_seqs = [] ,[]
    student_starts, teacher_starts, response_lens = [], [], []

    for trajectory, teacher_prompt in zip(trajectories, teacher_prompt_ids):
        response = trajectory.response_token_ids
        
        # here is where the prompt tokens are:
        student_starts.append(len(trajectory.prompt_token_ids)) 
        teacher_starts.append(len(teacher_prompt))

        # here is how long the response tokens are (they come after the prompt tokens)
        response_lens.append(len(response))

        # here are the actual token ids
        student_seqs.append(list(trajectory.prompt_token_ids) + list(response))
        teacher_seqs.append(list(teacher_prompt) + list(response))

    student_input_ids = _pad(student_seqs, pad_token_id)
    teacher_input_ids = _pad(teacher_seqs, pad_token_id)
    
    student_attention_mask = torch.zeros_like(student_input_ids)
    teacher_attention_mask = torch.zeros_like(teacher_input_ids)
    student_response_mask = torch.zeros_like(student_input_ids)
    teacher_response_mask = torch.zeros_like(teacher_input_ids)

    # for the attention mask - we just want to mask out padding
    # for the response mask - we want to mask out padding + prompt
    for i, (student_start, teacher_start, response_len) in enumerate(zip(student_starts, teacher_starts, response_lens)):
        
        # attention mask marks REAL TOKENS (prompt + response), zero only on trailing pad
        student_attention_mask[i, : student_start + response_len] = 1
        teacher_attention_mask[i, : teacher_start + response_len] = 1

        # response mask is everything from end of the prompt -> end of the response_len
        # that the POLICY sampled. Env-injected tool results / user turns are 0 so they
        # stay in the causal context (attention_mask) but do not enter the loss.
        loss = list(trajectories[i].loss_mask) if trajectories[i].loss_mask else [1] * response_len
        if len(loss) != response_len:
            raise ValueError(
                f"{trajectories[i].task_id}: loss_mask length {len(loss)} != {response_len}"
            )
        student_response_mask[i, student_start:student_start+response_len] = torch.tensor(loss)
        teacher_response_mask[i, teacher_start:teacher_start+response_len] = torch.tensor(loss)

    # longest response length from all sequences, gives you an idea of what to mask up to
    max_response_len = max(response_lens, default=0)

    # all we want to do here is create rollout_log_probs to get student_log_probs
    # we just don't want to be training on padding tokens
    rollout_logprobs = torch.zeros(len(trajectories), max_response_len)
    response_mask = torch.zeros(len(trajectories), max_response_len)
    step_ids = torch.full((len(trajectories), max_response_len), -1, dtype=torch.long)
    
    for i, trajectory in enumerate(trajectories):
        n = len(trajectory.rollout_logprobs)
        loss = list(trajectory.loss_mask) if trajectory.loss_mask else [1] * n
        if len(loss) != n:
            raise ValueError(
                f"{trajectory.task_id}: loss_mask length {len(loss)} != {n}"
            )
        # Pack trained tokens only so this lines up with gather_response_logprobs,
        # which left-packs positions where the (holed) student/teacher response mask is 1.
        packed_lp = [p for p, m in zip(trajectory.rollout_logprobs, loss) if int(m) == 1]
        packed_steps = packed_step_ids(loss, list(trajectory.step_spans), n)
        if len(packed_steps) != len(packed_lp):
            raise ValueError(
                f"{trajectory.task_id}: packed {len(packed_lp)} logprobs but "
                f"{len(packed_steps)} step ids"
            )
        k = len(packed_lp)
        response_mask[i, :k] = 1.0
        if packed_lp:
            rollout_logprobs[i, :k] = torch.tensor(packed_lp)
            step_ids[i, :k] = torch.tensor(packed_steps, dtype=torch.long)

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
        step_ids=step_ids,
    )

def gather_response_logprobs(response_logits:torch.Tensor, response_token_ids: torch.Tensor, response_mask:torch.Tensor, max_response_len:int) -> torch.Tensor:
    """
    response_logits = the logits for the input_ids, where the logit at position t corresponds to the token at t+1
    response_token_ids = the token ids of the response
    response_mask = the full input_id, with the mask over just the response
    max_response_len = the longest response length, which is what we padded the response_mask to

    Returns (batch, max_response_len) of LOG-PROBS (not raw logits), left-packed: entry [i, k] is the log-prob of the i'th sequence's k'th response token, zero-padded after.
    """
    # I'm basically trying to do what I just did with rollout_logprobs - get the logprobs I want to train on, and mask the rest

    # what we want to do is pluck out the actual token that was picked from the logits, remember that with our surrogate loss objective we don't do KL over the entire token distribution
    # logit[k] predicts token[k+1]
    targets = response_token_ids[:, 1:] # we have logprobs for all these tokens

    # logits are (B, T, V)
    logits = response_logits[:, :-1] # here are the logits for the corresponding tokens
    # [token0, token1, ...]

    batch_size, _, _ = logits.shape

    # convert to float32 to retain precision
    # so when we do a forward pass on the rollout trajectories, we need to compare the logprobs
    # logsoftmax = x - ∑_j exp(x_j)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    # gather along the VOCAB dim: at each position pick the score of the one token that was actually sampled. (B, T, V) indexed by (B, T, 1) -> (B, T, 1) -> squeeze -> (B, T).
    gathered_logprobs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Same slice as `targets` so the mask lines up with it element for element: entry k of
    # both refers to the token at position k+1.
    shifted_mask = response_mask[:, 1:]

    # now output the logprobs with the mask over them
    train_batch = torch.zeros(batch_size, max_response_len, dtype=gathered_logprobs.dtype, device=logits.device)
    for i in range(batch_size):
        # loop through batches and apply masking
        response = gathered_logprobs[i][shifted_mask[i].bool()]
        # we now have just the response logprobs in 'response'
        
        # find how long response is, we'll fill up train_batch with our response tokens
        n = min(response.numel(), max_response_len)

        # fill up the train_batch tensor with the responses, padded up to max response len in the batch
        train_batch[i, :n] = response[:n]
    
    # returns just the response, padded up to the longest response length in the batch
    # [R0,...Rn, 0, ..., 0]
    return train_batch

def response_logprobs_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    max_response_len: int,
) -> torch.Tensor:
    """LM-head only the positions that predict a response token, then pack left.

    Equivalent to `gather_response_logprobs(lm_head(hidden), ...)` without materializing
    logits for prompt/hint/pad rows.
    """
    pred_mask = response_mask[:, 1:].bool()
    packed_h = hidden[:, :-1][pred_mask]
    packed_targets = input_ids[:, 1:][pred_mask]
    packed_logits = lm_head(packed_h)
    gathered = packed_logits.gather(-1, packed_targets.unsqueeze(-1)).squeeze(-1)
    token_logp = gathered.float() - packed_logits.float().logsumexp(dim=-1)

    batch_size = hidden.size(0)
    out = torch.zeros(
        batch_size, max_response_len, dtype=token_logp.dtype, device=hidden.device
    )
    if token_logp.numel() == 0:
        return out

    # dest column = how many True mask entries precede this one in the row
    dest_col = (pred_mask.cumsum(dim=-1) - 1)[pred_mask]
    row_idx, _ = pred_mask.nonzero(as_tuple=True)
    keep = dest_col < max_response_len
    out[row_idx[keep], dest_col[keep]] = token_logp[keep]
    return out