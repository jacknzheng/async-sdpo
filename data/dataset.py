"""Loading and splitting paperinstruments/diligence-bench and tau2 domains.

Diligence-bench ships 150 rows in a single split named "test". We carve a
deterministic 120/30 split when `data.dataset=diligence`.

Tau2 mixes three domains into one pool. Retail and airline have official
train/test splits; banking_knowledge does not, so we carve 70/27. Task ids
collide across domains (`"0"` exists in both retail and airline), so every uid
is `"{domain}:{task.id}"`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import Dataset

from rubric import Criterion, Rubric

### BUILD TASKS AND PROMPTS
@dataclass
class Task:
    """One training row, with everything the rollout and (optional) judge need.

    Diligence fills `query` / `sections` / `criteria`. Tau2 fills `domain` and
    `tau2_task`; `query` is a readable dump of the user scenario for logging.
    """

    task_id: str
    query: str
    # Raw rubric sections as they appear in the dataset, kept because the `rubric` package
    # flattens sections and drops criterion ids on parse (see `section_spans`).
    sections: list[dict]
    # Parsed criteria in flattened order -- the order the judge reports verdicts in.
    criteria: list[Criterion] = field(default_factory=list)
    section_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    domain: str | None = None
    tau2_task: Any = None

    def to_rubric(self) -> Rubric:
        return Rubric(self.criteria)


def _build_task(row: dict) -> Task:
    sections = row["rubric"]["sections"] # index into the sections to get the list of arrays with the criteria

    # Rubric.validate_and_create_criteria natively accepts the {"sections": [...]} shape,
    # flattening nested criteria in order -- no reshaping needed.
    parsed = Rubric.from_dict(row["rubric"])

    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for section in sections:
        n = len(section.get("criteria", []))
        spans[section["id"]] = (cursor, cursor + n)
        cursor += n

    if cursor != len(parsed.rubric):
        raise ValueError(
            f"task {row['datasetId']}: section spans cover {cursor} criteria but the "
            f"parsed rubric has {len(parsed.rubric)} -- flattening order changed"
        )

    return Task(
        task_id=str(row["datasetId"]),
        query=row["query"],
        sections=sections,
        criteria=parsed.rubric,
        section_spans=spans,
    )


class TaskDataset(Dataset):
    """
    Wrap a list of Tasks so StatefulDataLoader can iterate them.
    Each item is 
        {"uid": task_id, "task": Task}
        
    `uid` is what AsyncDataLoader records as consumed/filtered across checkpoint resume. 
    
    collate_fn returns the list of dicts unchanged 
    - AsyncDataLoader reads `batch[0]["uid"]`.
    """
    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = list(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict:
        task = self.tasks[idx]
        return {"uid": task.task_id, "task": task}

    @staticmethod
    def collate_fn(batch: list[dict]) -> list[dict]:
        # used to collect samples into a batch, but we already store them in a batch so its fine!
        return batch


def load_tasks(
    dataset_name: str = "paperinstruments/diligence-bench",
    split: str = "test",
) -> list[Task]:
    """Load all 150 diligence-bench rows as Task objects."""
    from datasets import load_dataset  # imported lazily; heavy and network-bound

    rows = load_dataset(dataset_name, split=split)
    return [_build_task(row) for row in rows]


def split_tasks(
    tasks: list[Task], n_heldout: int = 30, seed: int = 0
) -> tuple[list[Task], list[Task]]:
    """Deterministic train/held-out split.

    Sorted by task_id before shuffling so the split depends only on the seed, not on
    whatever order the dataset happens to load in.
    """
    if n_heldout >= len(tasks):
        raise ValueError(f"n_heldout={n_heldout} must be < {len(tasks)} tasks")

    ordered = sorted(tasks, key=lambda t: _split_sort_key(t.task_id))
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    return shuffled[n_heldout:], shuffled[:n_heldout]


def _split_sort_key(task_id: str):
    """Numeric ids sort as ints; namespaced tau2 ids (`retail:0`) sort as strings."""
    try:
        return (0, int(task_id))
    except ValueError:
        return (1, task_id)


def namespace_id(domain: str, task_id: str) -> str:
    return f"{domain}:{task_id}"


def wrap_tau2_task(domain: str, tau2_task: Any) -> Task:
    """Lift a tau2 Task into our Task, namespacing the id so domains cannot collide."""
    scenario = getattr(tau2_task, "user_scenario", None)
    instructions = getattr(scenario, "instructions", scenario) if scenario is not None else ""
    return Task(
        task_id=namespace_id(domain, str(tau2_task.id)),
        query=str(instructions),
        sections=[],
        domain=domain,
        tau2_task=tau2_task,
    )


def load_tau2_split(
    domains: tuple[str, ...] = ("banking_knowledge", "retail", "airline"),
    n_heldout: int = 27,
    split_seed: int = 0,
) -> tuple[list[Task], list[Task]]:
    """Official retail/airline splits plus a carved banking_knowledge 70/27.

    Combined: 174 train / 87 held-out.
    """
    try:
        from tau2.registry import registry
    except ImportError as exc:
        raise ImportError(
            "tau2 is not installed. `uv sync --extra tau2` (retail/airline) or "
            "`uv sync --extra knowledge` (also banking_knowledge)."
        ) from exc

    train: list[Task] = []
    heldout: list[Task] = []
    for domain in domains:
        loader = registry.get_tasks_loader(domain)
        if domain in ("retail", "airline"):
            train.extend(wrap_tau2_task(domain, t) for t in loader("train"))
            heldout.extend(wrap_tau2_task(domain, t) for t in loader("test"))
            continue
        if domain == "banking_knowledge":
            wrapped = [wrap_tau2_task(domain, t) for t in loader()]
            carved_train, carved_held = split_tasks(wrapped, n_heldout=n_heldout, seed=split_seed)
            train.extend(carved_train)
            heldout.extend(carved_held)
            continue
        raise ValueError(
            f"unsupported tau2 domain {domain!r}; expected banking_knowledge, retail, or airline"
        )
    return train, heldout


def load_split(data_config) -> tuple[list[Task], list[Task]]:
    """Train / held-out split for whichever dataset the config names."""
    if data_config.dataset == "diligence":
        tasks = load_tasks(data_config.dataset_name, data_config.dataset_split)
        return split_tasks(tasks, data_config.n_heldout, data_config.split_seed)
    return load_tau2_split(
        domains=tuple(data_config.domains),
        n_heldout=data_config.n_heldout,
        split_seed=data_config.split_seed,
    )


HINT_SEPARATOR = "\n\n"


def build_prompt(task: Task, hint: str | None = None) -> str:
    """
    Student = query only. Teacher = query then hint (hint is a true prefix-cache suffix).

    This allows us to do prefix caching during training to save on the teacher forward pass.
    Tau2 rollouts do not use this -- their student prompt is the chat-templated conversation.
    """

    base = f"Answer the following financial diligence question.\n\n{task.query}"
    if hint:
        return f"{base}{HINT_SEPARATOR}{hint}"
    return base
