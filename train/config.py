from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf
from enum import Enum
from typing import Any, List, Optional, Annotated, Type, TypeVar
from dataclasses import fields
import dataclasses
import typing
from enum import Enum

def validate_dict_keys_against_dataclass(datacls: Type[Any], d: dict):
    """
    Validate the keys of a dict against fields of a dataclass.

    Args:
        datacls: The dataclass class to validate
    """
    valid_fields = {f.name for f in fields(datacls)}
    if invalid_keys := set(d.keys() - valid_fields):
        raise ValueError(f"Invalid fields {invalid_keys} for {datacls.__name__}. Valid fields are {valid_fields}.")


def _resolve_class_type(type_annotation: Any) -> Optional[Type]:
    """Extract the concrete non-plain class type from a type annotation.

    Handles plain types, Optional[T], Union[T, None], and Annotated[T, ...].
    Returns None if no dataclass or Enum type can be resolved.
    """
    origin = typing.get_origin(type_annotation)  # get outermost wrapper type

    if origin is Union:
        # if its a Union (commonly Optional), then unwrap and continue
        for arg in typing.get_args(type_annotation):
            if arg is type(None):
                continue  # unwrap, drop the None field in the Optional: Union[str, None]
            resolved = _resolve_class_type(arg)  # recurse, find the class without outer wrapping
            if resolved is not None:
                return resolved
        return None  # if Union[None, None] - its None

    if origin is Annotated:
        # Annotated[str, "must be uppercase"] -> get rid of the metadata at the last index
        return _resolve_class_type(typing.get_args(type_annotation)[0])

    # Plain class check
    if isinstance(type_annotation, type) and (  # check its not an instance, must be a class
        # if it is a class - is it a dataclass or an Enum?
        dataclasses.is_dataclass(type_annotation) or issubclass(type_annotation, Enum)
    ):
        return type_annotation  # return the class

    return None

T = TypeVar("T")


# takes in a class type, outputs a class instance
def build_nested_dataclass(datacls: Type[T], d: dict) -> T:
    """Recursively build a dataclass from a dict, handling nested dataclasses.

    Supports fields typed as standard python types, plain dataclasses,
    Optional[DataclassType], Union[DataclassType, None], and Annotated[...] wrappers.
    Non-dataclass fields (primitives, dicts, lists, etc.) are passed through as-is.
    """
    validate_dict_keys_against_dataclass(datacls, d)  # check d's keys line up with the class
    kwargs = {}
    for f in fields(datacls):  # {k: v}
        if f.name not in d:
            continue
        value = d[f.name]  # find the value associated to the class field: f.name
        nested_cls = _resolve_class_type(f.type)  # unwrap Optional, Annotated, etc. get the base class
        if nested_cls is not None:
            if isinstance(value, dict) and dataclasses.is_dataclass(nested_cls):
                kwargs[f.name] = build_nested_dataclass(nested_cls, value)
            elif issubclass(nested_cls, Enum):
                kwargs[f.name] = nested_cls(value)
            else:
                kwargs[f.name] = value
        else:
            kwargs[f.name] = value
    return datacls(**kwargs)  # output instance of the datacls class, with the kwargs


@dataclass(frozen=True)
class BaseConfig(ABC):
    @classmethod
    def from_dict_config(cls, cfg: Union[DictConfig, dict]) -> "BaseConfig":
        raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
        return build_nested_dataclass(cls, raw)


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    model: str = "Qwen/Qwen3.8-27B"
    smoke_model: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class SamplingParams(BaseConfig):
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1536


HINT_PROMPTS = (
    "answer_free",
    "answer_bearing",
    "mixture",
    "gold",
    "step_hint",
)


@dataclass(frozen=True)
class HintConfig(BaseConfig):
    prompt: str = "gold"
    model: str = "stealth/ox-alpha"
    concurrency: int = 8
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.prompt not in HINT_PROMPTS:
            raise ValueError(
                f"generator.hint.prompt must be one of {HINT_PROMPTS}, got {self.prompt!r}"
            )
        if self.concurrency < 1:
            raise ValueError("generator.hint.concurrency must be >= 1")


@dataclass(frozen=True)
class InferenceEngineConfig(BaseConfig):
    backend: str = "vllm"
    n_rollout_gpus: int = 4 
    gpu_memory_utilization: float = 0.85
    max_prompt_tokens: int = 8192
    max_model_len: int = 16384
    store_capacity: int = 512
    weight_sync_interval: int = 1
    weight_sync_host: str = "127.0.0.1"
    weight_sync_port: int = 51216
    weight_sync_timeout_s: float = 180.0
    weight_sync_bucket_mb: int = 512

    def __post_init__(self) -> None:
        from train.backends import BACKEND_NAMES

        if self.backend not in BACKEND_NAMES:
            raise ValueError(
                f"generator.engine.backend must be one of {BACKEND_NAMES}, got {self.backend!r}"
            )


@dataclass(frozen=True)
class GeneratorConfig(BaseConfig):
    engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    hint: HintConfig = field(default_factory=HintConfig)
    max_steps: int = 30

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("generator.max_steps must be >= 1")


@dataclass(frozen=True)
class OptimizerConfig(BaseConfig):
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 10


@dataclass(frozen=True)
class AlgorithmConfig(BaseConfig):
    clip_ratio_low: float = 0.8
    clip_ratio_high: float = 1.4
    use_one_sided_clip: bool = False
    one_sided_clip_max: float = 2.0

    adv_clip_mult: float = 3.0
    adv_ema_decay: float = 0.99
    adv_ema_bias_correction: bool = True

    use_kl_penalty: bool = False

    max_staleness: int = 3

    group_size: int = 1
    keep_failures: bool = True

    # SOD (step-wise OPD reweighting) for multi-step TIR. No-op when a trajectory
    # has a single step. w_k = min((d_1+eps)/(d_k+eps), 1+delta); see train/loss.py.
    use_sod: bool = True
    sod_eps: float = 1e-6
    sod_delta: float = 0.2

    def __post_init__(self) -> None:
        if not self.clip_ratio_low < 1.0 < self.clip_ratio_high:
            raise ValueError(
                f"clip window [{self.clip_ratio_low}, {self.clip_ratio_high}] must contain "
                "1.0, or unchanged tokens (ratio == 1) would be clipped"
            )
        if not 0.0 < self.adv_ema_decay < 1.0:
            raise ValueError("adv_ema_decay must be in (0, 1)")
        if self.sod_eps <= 0.0:
            raise ValueError("sod_eps must be > 0")
        if self.sod_delta < 0.0:
            raise ValueError("sod_delta must be >= 0")

@dataclass(frozen=True)
class FSDPConfig(BaseConfig):
    """FSDP2 sharding options. Only consulted when the world size is > 1."""

    cpu_offload: bool = False
    """Park parameters and optimizer state in host RAM when idle. Unlocks models that
    otherwise will not fit, at a large throughput cost -- try plain sharding first, and
    reach for this only after an OOM that lowering mini_batch_size did not fix."""


@dataclass(frozen=True)
class TrainerConfig(BaseConfig):
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    n_trainer_gpus: int = 4
    batch_size: int = 16             # divisible by 4
    eval_batch_size: int = 16
    mini_batch_size: int = 4

    gradient_checkpointing: bool = True
    
    total_steps: int = 500

    # only for non fully-async training setups
    epochs: int = 10

    seed: int = 1234

    compile_trainer: bool = True

    def __post_init__(self) -> None:
        if self.n_trainer_gpus < 1:
            raise ValueError("need at least 1 trainer GPU")
        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by mini_batch_size")
        if self.batch_size % self.n_trainer_gpus != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by n_trainer_gpus "
                f"({self.n_trainer_gpus}) so each FSDP rank gets an equal shard"
            )


@dataclass(frozen=True)
class DataConfig(BaseConfig):
    # "tau2" (banking_knowledge + retail + airline) or "diligence".
    dataset: str = "tau2"
    dataset_name: str = "paperinstruments/diligence-bench"
    dataset_split: str = "test"
    domains: list[str] = field(default_factory=lambda: ["banking_knowledge", "retail", "airline"])
    # Banking has no official split -- carve this many held-out of 97. Diligence
    # tests pass n_heldout=30 explicitly; set data.n_heldout=30 when dataset=diligence.
    n_heldout: int = 27
    split_seed: int = 0
    retrieval: str = "alltools-qwen"
    user_llm: str = "openrouter/stealth/ox-alpha"
    # Parallel Search (diligence TIR only). https://docs.parallel.ai/search/search-quickstart
    search_mode: str = "fast"
    search_timeout: float = 30.0
    search_max_chars: int = 12000

    def __post_init__(self) -> None:
        if self.dataset not in ("tau2", "diligence"):
            raise ValueError(f"data.dataset must be 'tau2' or 'diligence', got {self.dataset!r}")
        if self.retrieval != "alltools-qwen":
            raise ValueError(
                f"data.retrieval must be 'alltools-qwen' (Qwen embeddings); got {self.retrieval!r}"
            )
        if self.search_mode not in ("turbo", "fast", "basic", "advanced"):
            raise ValueError(
                f"data.search_mode must be turbo/fast/basic/advanced, got {self.search_mode!r}"
            )
        if self.search_timeout <= 0.0:
            raise ValueError("data.search_timeout must be > 0")
        if self.search_max_chars < 1:
            raise ValueError("data.search_max_chars must be >= 1")


@dataclass(frozen=True)
class JudgeConfig(BaseConfig):
    model: str = "stealth/ox-alpha"
    max_concurrency: int = 8
    max_retries: int = 3
    timeout: float = 120.0
    eval_interval: int = 25


@dataclass(frozen=True)
class LoggingConfig(BaseConfig):
    log_interval: int = 1
    checkpoint_interval: int = 50
    output_dir: str = "runs/sdpo-tau2"
    # Rank-0 writes train.log / args.txt / config.yaml here. Default `/log` on the
    # 8xH100 box; falls back to ./log if that path is not writable.
    log_dir: str = "/log"
    run_name: str = ""
    wandb_project: str = "sdpo-tau2"
    wandb_entity: str = ""
    wandb_enabled: bool = True


@dataclass(frozen=True)
class Config(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    total_num_gpus: int = 8
    num_workers: int = 4 * total_num_gpus

    def __post_init__(self) -> None:
        total = self.generator.engine.n_rollout_gpus + self.trainer.n_trainer_gpus
        if total > self.total_num_gpus:
            raise ValueError(f"GPU split must sum to <= {self.total_num_gpus}, got {total}")
        if self.generator.engine.n_rollout_gpus < 1 or self.trainer.n_trainer_gpus < 1:
            raise ValueError("need at least 1 rollout GPU and 1 trainer GPU")

    @classmethod
    def from_cli_overrides(cls, args: Optional[List[str]] = None) -> "Config":
        if not args:
            return cls()
        for arg in args:
            if arg.startswith("+"):
                raise ValueError(
                        f"The '+' prefix for adding new config fields is not supported: {arg!r}. "
                        "Every field must exist on the dataclasses in train/config.py."
                    )
        return cls.from_dict_config(OmegaConf.from_cli(args))

def to_yaml(cfg: Config) -> str:
    return yaml.dump(asdict(cfg), sort_keys=False)