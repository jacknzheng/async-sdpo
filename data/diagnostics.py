"""Structured per-run diagnostic artifacts.

The normal Python logs remain concise and human-readable. These JSONL streams
preserve machine-readable failure context, full rollouts, and engine/training
lifecycle events for post-mortems. Callers deliberately choose the fields they
record so API keys and environment variables never enter an artifact.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARTIFACT_FILES = {
    "api_failures": "api_failures.jsonl",
    "evaluations": "evaluations.jsonl",
    "rollouts": "rollouts.jsonl",
    "sandbox": "sandbox.jsonl",
    "training": "training.jsonl",
    "vllm": "vllm.jsonl",
}


def configure_artifact_logging(log_dir: Path, *, rank: int = 0) -> None:
    """Create rank-0 JSONL artifact files and attach thread-safe handlers."""
    if rank != 0:
        return
    for channel, filename in ARTIFACT_FILES.items():
        path = log_dir / filename
        path.touch(exist_ok=True)
        artifact_logger = logging.getLogger(f"sdpo.artifact.{channel}")
        for handler in list(artifact_logger.handlers):
            handler.close()
            artifact_logger.removeHandler(handler)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        artifact_logger.addHandler(handler)
        artifact_logger.setLevel(logging.INFO)
        artifact_logger.propagate = False


def artifact_event(channel: str, event: str, **fields: Any) -> None:
    """Append one JSON object if the channel was configured for this process."""
    artifact_logger = logging.getLogger(f"sdpo.artifact.{channel}")
    if not artifact_logger.handlers:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "pid": os.getpid(),
        **fields,
    }
    artifact_logger.info(
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    )
