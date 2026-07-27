"""Loading and splitting paperinstruments/diligence-bench.

The dataset ships 150 rows in a single split named "test", with no reference answers and
no train/test division. Fields are `datasetId`, `query`, and `rubric`, where every rubric
has exactly three sections -- factual-accuracy (73% of weight), analytical-reasoning (12%),
and risk-awareness (15%) -- holding 23-50 weighted criteria.

We carve a deterministic 120/30 split. The held-out 30 are never trained on, so the gap
between train and held-out judge scores is our overfitting signal -- which matters here
because the rubric is effectively an answer key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from rubric import Criterion, Rubric


@dataclass
class Task:
    """One diligence-bench row, with everything the rollout and judge need."""

    task_id: str
    query: str
    # Raw rubric sections as they appear in the dataset, kept because the `rubric` package
    # flattens sections and drops criterion ids on parse (see `section_spans`).
    sections: list[dict]
    # Parsed criteria in flattened order -- the order the judge reports verdicts in.
    criteria: list[Criterion] = field(default_factory=list)
    # section id -> (start, end) index range into `criteria`. The rubric package's
    # `Criterion` model keeps only `weight` and `requirement`, so section membership and
    # criterion ids are lost on parse; we record the spans here to still report per-section
    # scores. factual-accuracy carries 73% of the weight, so watching it move on its own
    # is worth the bookkeeping.
    section_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    # NOTE: no `hint` field. Hints are generated per-rollout from the draft the model
    # actually produced (data/hint.py), so they belong to the Trajectory, not the Task --
    # a task-level hint would be identical across every rollout, which is exactly the
    # static behavior this replaced.

    def to_rubric(self) -> Rubric:
        return Rubric(self.criteria)


def _build_task(row: dict) -> Task:
    sections = row["rubric"]["sections"]

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


def load_tasks(
    dataset_name: str = "paperinstruments/diligence-bench",
    split: str = "test",
) -> list[Task]:
    """Load all 150 rows as Task objects. Hints are generated per-rollout, not here."""
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

    ordered = sorted(tasks, key=lambda t: int(t.task_id))
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    return shuffled[n_heldout:], shuffled[:n_heldout]


def build_prompt(task: Task, hint: str | None = None) -> str:
    """Render the prompt for one task.

    `hint=None` gives the STUDENT prompt (bare query). Passing `trajectory.hint` gives the
    TEACHER prompt. This asymmetry is the whole of SDPO: same weights, different context.
    """
    if hint:
        return (
            f"{hint}\n"
            "Answer the following financial diligence question.\n\n"
            f"{task.query}"
        )
    return f"Answer the following financial diligence question.\n\n{task.query}"
