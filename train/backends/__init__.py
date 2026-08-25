"""Backend selection: name -> (engine class, transport class).

A plain dict, not a decorator registry. The engine and the transport are chosen TOGETHER
and always from the same backend, because their wire formats have to agree -- the receiver
derives its broadcasts from metadata the sender chose. Pairing them here makes a mismatched
combination unrepresentable rather than a runtime hang.

Backends are imported lazily inside `get_backend` so that `import train.backends` does not
drag in vLLM (or any future engine's dependencies) on a machine that lacks them. Only the
backend actually named in config is imported.

To add a backend: subclass `InferenceEngine` and `WeightTransport` from
`train/backends/backend.py` in a new module here and add one line to `_BACKENDS`. Read the
invariants at the top of `train/backends/backend.py` first -- particularly the
processed-logprobs requirement, which a new engine can violate silently.
"""

from __future__ import annotations

# name -> (module attribute path). Kept as strings so nothing is imported until asked for.
_BACKENDS = {
    "vllm": ("train.backends.vllm", "VLLMRolloutEngine", "NCCLWeightTransport"),
}

BACKEND_NAMES = tuple(_BACKENDS)


def get_backend(name: str) -> tuple[type, type]:
    """Return `(engine_cls, transport_cls)` for a backend name.

    Raises ValueError on an unknown name -- config validation calls this indirectly via
    BACKEND_NAMES, so a typo fails at construction rather than at the first weight sync.
    """
    try:
        module_path, engine_attr, transport_attr = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown rollout backend {name!r}; available: {', '.join(BACKEND_NAMES)}"
        ) from None

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, engine_attr), getattr(module, transport_attr)
