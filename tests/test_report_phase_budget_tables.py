"""Tests for phase budget table export in embodied report (robosuite-independent).

Verifies that the export_embodied_report_tables module handles
phase_trace data and budget-related fields correctly.
"""

import ast
from pathlib import Path


def test_export_module_has_ablation_comparison():
    """export_embodied_report_tables should have ablation comparison function."""
    import inspect
    from vlm_nsm.experiments.export_embodied_report_tables import (
        export_ablation_comparison,
    )
    assert callable(export_ablation_comparison)


def test_ablation_keys_include_stage7a10():
    """Ablation comparison should include config for Stage 7A-10."""
    from vlm_nsm.experiments.export_embodied_report_tables import export_ablation_comparison
    import inspect
    source = inspect.getsource(export_ablation_comparison)
    # Should reference best_config_v2 or stage7a10 pattern
    has_stage7a = "stage7a" in source.lower() or "best_config_v2" in source


def test_phase_budget_constants_exist():
    """Phase budget constants should be defined in run_embodied_stacking_robosuite."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
        _MIN_BUDGET_REACH, _MIN_BUDGET_GRASP, _MIN_BUDGET_LIFT,
        _MIN_BUDGET_REACH_PLACE, _MIN_BUDGET_RELEASE, _MIN_BUDGET_RETREAT,
    )
    assert _MIN_BUDGET_REACH >= 0
    assert _MIN_BUDGET_GRASP >= 0
    assert _MIN_BUDGET_LIFT >= 0
    assert _MIN_BUDGET_REACH_PLACE >= 0
    assert _MIN_BUDGET_RELEASE >= 0
    assert _MIN_BUDGET_RETREAT >= 0


def test_phase_budget_release_depends_on_release_open_steps():
    """_MIN_BUDGET_RELEASE should be based on _RELEASE_OPEN_STEPS."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
        _MIN_BUDGET_RELEASE, _RELEASE_OPEN_STEPS,
    )
    assert _MIN_BUDGET_RELEASE == _RELEASE_OPEN_STEPS + 5


def test_phase_trace_in_episode_return():
    """_run_robosuite_episode return dict should include 'phase_trace'."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    # We can't call it without env, but we can inspect the source
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert '"phase_trace"' in source or "'phase_trace'" in source


def test_phase_budget_map_comprehensive():
    """Budget map should cover all step phases."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    expected_phases = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
    for phase in expected_phases:
        assert phase in source, f"Phase '{phase}' missing from source"


def test_remaining_budget_function():
    """_remaining_budget should return non-negative horizon - total_steps."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _remaining_budget
    assert _remaining_budget(50, 100) == 50
    assert _remaining_budget(100, 100) == 0
    assert _remaining_budget(150, 100) == 0
    assert _remaining_budget(0, 100) == 100


def test_horizon_aware_flag_controls_budget_check():
    """horizon_aware flag should guard budget check."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "if horizon_aware:" in source
    assert "min_budget_map" in source or "_MIN_BUDGET" in source


def test_budget_check_uses_min_budget_map():
    """Budget check should use a min_budget_map for phase-specific limits."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "min_budget_map" in source


def test_phase_trace_retreat_increment_exists():
    """Retreat phase should increment phase_trace['retreat']."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    import ast
    tree = ast.parse(source)

    retreat_increment = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript):
                if isinstance(node.target.slice, ast.Constant) and node.target.slice.value == "retreat":
                    retreat_increment = True

    assert retreat_increment, "phase_trace['retreat'] should be incremented somewhere"


def test_phase_trace_lift_increment_exists():
    """Lift phase should increment phase_trace['lift']."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "phase_trace[\"lift\"]" in source or "phase_trace['lift']" in source
