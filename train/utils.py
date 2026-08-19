"""
Implementation of fully async training in SkyRL.

For details, see https://docs.skyrl.ai/docs/tutorials/fully_async.

High-level notes:
- The global_step in each training loop iteration denotes the "current step being worked on", so
`global_step - 1` is the number of steps the model has finished training.
- We do not do any cross-epoch asynchrony here, so all the control logics like
  generation workers and data buffer are initialized per-epoch. The async dataloader
  and staleness manager are also reset / validated at the end of each epoch.
"""

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set, Tuple, Dict

import torch
from loguru import logger
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from train.config import Config
from train.store import Trajectory

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

    def to(self, device:torch.device):
        # move batch to GPU
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
            policy_versions=self.policy_versions.to(device)
        )
        

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
        
        # attention mask is everything up to the end of the last student_prompt token
        student_attention_mask[i, student_start:] = 1 # index into batch i (i.e. i'th sequence), 
        teacher_attention_mask[i, teacher_start:] = 1

        # response mask is everything from end of the prompt -> end of the response_len
        student_response_mask[i, student_start:student_start+response_len] = 1
        teacher_response_mask[i, teacher_start:teacher_start+response_len] = 1

    # longest response length from all sequences, gives you an idea of what to mask up to
    max_response_len = max(response_lens, default=0)

    # all we want to do here is create rollout_log_probs to get student_log_probs
    # we just don't want to be training on padding tokens
    rollout_logprobs = torch.zeros(len(trajectories), max_response_len)
    response_mask = torch.zeros(len(trajectories), max_response_len)
    
    for i, trajectory in enumerate(trajectories):
        n = len(trajectory.rollout_logprobs)
        response_mask[i, :n] = 1
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

def gather_response_logprobs(response_logits:torch.Tensor, response_token_ids: torch.Tensor, response_mask:torch.Tensor, max_len:int) -> torch.Tensor:
    """
    response_logits = the logits for the input_ids, where the logit at position t corresponds to the token at t+1
    response_token_ids = the token ids of the response
    response_mask = the full input_id, with the mask over just the response
    max_len = the longest response length, which is what we padded the response_mask to
    """
    # I'm basically trying to do what I just did with rollout_logprobs - get the logprobs I want to train on, and mask the rest

    # what we want to do is pluck out the actual token that was picked from the logits, remember that with our surrogate loss objective we don't do KL over the entire token distribution
    targets = response_token_ids[:, 1:] # we have logprobs for all these tokens
    
    # logits are (B, T, V)
    logits = response_logits[:, :-1] # here are the logits for the corresponding tokens
    # [token0, token1, ...]

    batch_size, _, _ = logits.shape

    # the gather operation requires the same shape, aside from the dimension you're gathering along
    gathered_logits = torch.gather(logits, dim=-1, index=targets)

    # logits are shifted to the right by one - since logit_t corresponds to token_id_{t+1}
    # mask =         [0,0,0,1,1,1,1,1]
    # shifted_mask = [0,0,1,1,1,1,1,1]
    shifted_mask = response_mask[:, 1:] # shifts to left by one

    # now output the logits with the mask over them
    train_batch = torch.zeros(batch_size, max_len, dtype=gathered_logits.dtype, device=logits.device)
    for i in range(batch_size):
        # loop through batches and apply masking
        response = gathered_logits[i][shifted_mask[i].bool()]
        # we now have just the response logits in 'response'
        
        # find how long response is, we'll fill up train_batch with our response tokens
        n = min(response.numel(), max_len)

        # fill up the train_batch tensor with the responses, padded up to max response len in the batch
        train_batch[i, :n] = response[:n]
    
    # returns just the response, padded up to the longest response length in the batch
    # [R0,...Rn, 0, ..., 0]
    return train_batch