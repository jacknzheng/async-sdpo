"""Guard: the evaluate_simulation() call in data/tau_harness.py must supply every
required (no-default) parameter of the pinned tau2 evaluate_simulation signature.
This would have caught the missing required `solo_mode` (tau2-bench a2c0247)."""
import ast
import inspect
import pathlib

import pytest

tau2 = pytest.importorskip("tau2")
from tau2.evaluator.evaluator import evaluate_simulation  # noqa: E402


def _harness_call_keywords():
    src = pathlib.Path(__file__).resolve().parents[1] / "data" / "tau_harness.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "evaluate_simulation"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("no evaluate_simulation(...) call found in data/tau_harness.py")


def test_harness_supplies_all_required_params():
    sig = inspect.signature(evaluate_simulation)
    required = {
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    supplied = _harness_call_keywords()
    missing = required - supplied
    assert not missing, f"evaluate_simulation() call missing required params: {missing}"


def test_solo_mode_is_supplied():
    assert "solo_mode" in _harness_call_keywords()
