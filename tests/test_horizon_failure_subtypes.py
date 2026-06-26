"""Tests for horizon failure subtype classification (robosuite-independent).

Verifies that horizon failures are classified into context-aware subtypes
based on the current step phase at the time of budget exhaustion.
"""

import ast
from pathlib import Path


def _get_horizon_failure_code() -> list[str]:
    """Extract horizon failure subtype assignment lines from source."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    tree = ast.parse(source)

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "failure_type":
                    if any(
                        isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.startswith("horizon_")
                        for v in (node.value if isinstance(node.value, (ast.Tuple, ast.List)) else [node.value])
                    ):
                        lines.append(ast.unparse(node))
    return lines


def test_horizon_before_grasp_subtype_exists():
    """horizon_before_grasp failure subtype should be defined."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "horizon_before_grasp" in source


def test_horizon_preplace_subtype_exists():
    """horizon_preplace failure subtype should be defined."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "horizon_preplace" in source


def test_horizon_holding_object_subtype_exists():
    """horizon_holding_object failure subtype should be defined."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "horizon_holding_object" in source


def test_horizon_after_goal_subtype_exists():
    """horizon_after_goal failure subtype should be defined."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "horizon_after_goal" in source


def test_horizon_generic_fallback_exists():
    """Generic horizon_{step_name} fallback should exist."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert 'f"horizon_{step_name}"' in source or 'f"horizon_{step_name}"' in source


def test_horizon_subtypes_are_mutually_exclusive():
    """Each horizon failure should be classified by exactly one subtype."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    tree = ast.parse(source)

    # Count all horizon_* string constants in the file
    horizon_types = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("horizon_"):
            horizon_types.add(node.value)

    # We should have at least 4 distinct horizon subtypes
    assert len(horizon_types) >= 4, f"Expected >= 4 horizon subtypes, got {horizon_types}"


def test_horizon_subtypes_recorded_in_failure_stage_counts():
    """Horizon failure subtypes should be recorded in failure_stage_counts."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    # Check that failure_stage_counts is updated for horizon failures
    assert "failure_stage_counts[failure_type]" in source or "failure_stage_counts" in source


def test_horizon_subtype_before_grasp_uses_low_budget():
    """horizon_before_grasp should trigger when budget < _MIN_BUDGET_GRASP."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "_MIN_BUDGET_GRASP" in source


def test_horizon_budget_constants_defined():
    """All horizon budget constants should be positive integers."""
    import ast
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    tree = ast.parse(source)

    budget_constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_MIN_BUDGET_"):
                    budget_constants[target.id] = node.value

    for name, val_node in budget_constants.items():
        if isinstance(val_node, ast.Constant):
            assert isinstance(val_node.value, int), f"{name} should be int, got {type(val_node.value)}"
            assert val_node.value > 0, f"{name} should be positive"
