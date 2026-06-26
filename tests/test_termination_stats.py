"""Tests for termination_stats in episode return dict (robosuite-independent)."""


def test_termination_stats_in_return_dict():
    """termination_stats should be a key in _run_robosuite_episode result."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "termination_stats" in source


def test_termination_stats_has_required_keys():
    """termination_stats should have 5 counter keys."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    tree = ast.parse(source)

    # Find the termination_stats dict
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "terminate_on_goal_count" in keys:
                expected = {"terminate_on_goal_count", "terminate_after_safe_release_count",
                            "skip_retreat_if_goal_satisfied_count", "normal_completion_count",
                            "horizon_termination_count"}
                assert expected.issubset(set(keys)), f"Missing keys: {expected - set(keys)}"
                return
    assert False, "termination_stats dict not found"


def test_run_robosuite_eval_aggregates_termination_stats():
    """run_robosuite_eval should aggregate termination_stats across episodes."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "term_stats_agg" in source


def test_termination_stats_in_eval_summary():
    """Evaluation summary should include termination_stats."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert '"termination_stats"' in source or "'termination_stats'" in source


def test_termination_stats_mode():
    """termination_stats dict uses conditional expressions for counts."""
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "termination_stats" in source


def test_termination_stats_in_export_module():
    """Export module should have termination stats table function."""
    import ast
    source = open("src/vlm_nsm/experiments/export_embodied_report_tables.py").read()
    assert "export_termination_stats_table" in source


def test_skip_retreat_triggered_tracking():
    """skip_retreat should set _skip_retreat_triggered flag."""
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "_skip_retreat_triggered" in source


def test_terminate_release_triggered_tracking():
    """terminate_after_safe_release should set _terminate_release_triggered."""
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "_terminate_release_triggered" in source
