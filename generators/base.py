"""
Copied from SkyRL
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import torch

TrainingPhase = Literal["train", "eval"]

@dataclass
class GeneratedOutputGroup:
    """
    The GeneratorOutput for a single group of rollouts, along with the metadata.

    Attributes:
        generator_output (GeneratorOutput): The GeneratorOutput for a single group of rollouts.
            That is, the output to the same prompt, but `n_samples_per_prompt` of it.

        uid (str): The uid of the group. Underlyingly, it is the index of the data in train_dataloader.dataset.

        global_step_when_scheduled (int): The global step when the group was scheduled for generation,
            used for validating the staleness control.

        group_completion_time_s (Optional[float]): Wall-clock time (seconds) for the whole group to
            finish generation, i.e. the time for the slowest trajectory in the group to complete.
            None if generation timing was not captured.

        prompts (Optional[List[Any]]): The generator input prompts (in OpenAI message format) for this
            group, one per trajectory (parallel to ``generator_output["response_ids"]``). Retained so
            the trajectory logger can render prompt + response, mirroring the synchronous trainer which
            logs ``generator_input["prompts"]``. None if not captured.
    """

    generator_output: GeneratorOutput
    uid: str
    global_step_when_scheduled: int
    group_completion_time_s: Optional[float] = None
    prompts: Optional[List[Any]] = None

@dataclass
class TrajectoryID:
    instance_id: str  # Unique identifier for the instance in the dataset
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"

@dataclass
class BatchMetadata:
    global_step: int
    training_phase: TrainingPhase


class GeneratorInput(TypedDict):
    prompts: List[str]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    # Wall-clock generation time (seconds) for each trajectory, with one entry per
    # trajectory in the input batch (i.e. per ``agent_loop`` call). Used by the fully
    # async trainer to compute per-group / intra-group completion-time metrics.
    trajectory_generation_times: Optional[List[float]]
    # Engine and env time splits of ``trajectory_generation_times``, one list entry per trajectory,
    # e.g. {"llm": [...], "env": [...]}. trajectory_time_splits is None if any trajectory did not
    # record its split.
    trajectory_time_splits: Optional[Dict[str, List[float]]]
    rollout_expert_indices: Optional[List[List[List[List[int]]]]]  # [batch_size, seq_len, layer_num, topk]
    # Applicable only for step-wise training
    is_last_step: Optional[List[bool]]
    # Per-row env metrics (one dict per row in the flattened batch). Used by
    # ``dump_per_dataset_eval_results`` to surface env-specific info (e.g. RLM's
    # ``rlm_metadata``) in eval JSONL dumps.
    env_metrics: Optional[List[Dict[str, Any]]]


class MetricsOutput(TypedDict):
    avg_score: Optional[float]
    pass_at_n: Optional[float]
    mean_positive_reward: Optional[float]


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError
