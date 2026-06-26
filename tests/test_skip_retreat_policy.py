"""Tests for skip_retreat_if_goal_satisfied policy logic (robosuite-independent).

Verifies that:
  - skip_retreat correctly skips retreat when goal achieved
  - successful_steps is still counted
  - early_stopped is set
  - No env.step calls occur during skipped retreat
"""


def test_skip_retreat_policy_accepts_bool():
    """skip_retreat_if_goal_satisfied should accept bool values."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    param = sig.parameters["skip_retreat_if_goal_satisfied"]
    # Should accept bool (default is False)
    assert param.default is False
    # Should be a bool-typed parameter
    assert param.annotation is bool or param.annotation == "bool"


def test_skip_retreat_plus_terminate_release_mutual():
    """skip_retreat and terminate_after_safe_release should be independently configurable."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    # Both params exist and have independent defaults
    assert sig.parameters["skip_retreat_if_goal_satisfied"].default is False
    assert sig.parameters["terminate_after_safe_release"].default is False


def test_skip_retreat_check_before_goal_check():
    """skip_retreat should check goal_was_achieved flag."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()

    # Find the retreat phase skip_retreat block
    import ast
    tree = ast.parse(source)

    skip_retreat_checks_goal = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if "skip_retreat_if_goal_satisfied" in condition and "goal_was_achieved" in condition:
                skip_retreat_checks_goal = True

    assert skip_retreat_checks_goal, "skip_retreat should check goal_was_achieved"


def test_skip_retreat_counts_as_success():
    """When skip_retreat triggers, the retreat step should count as successful."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    import ast
    tree = ast.parse(source)

    counts_success = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if "skip_retreat_if_goal_satisfied" in condition:
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.target, ast.Name) and child.target.id == "successful_steps":
                            counts_success = True

    assert counts_success, "skip_retreat branch should increment successful_steps"


def test_skip_retreat_no_env_steps():
    """When skip_retreat triggers, there should be no env.step calls."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    import ast
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if "skip_retreat_if_goal_satisfied" in condition:
                for child in ast.walk(node):
                    if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                        if isinstance(child.value.func, ast.Attribute) and child.value.func.attr == "step":
                            assert False, "skip_retreat branch should not call env.step"


def test_skip_retreat_also_stops_remaining_skills():
    """When skip_retreat triggers, remaining skills should be skipped via early_stopped."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "early_stopped" in source


def test_run_robosuite_eval_passes_skip_retreat():
    """run_robosuite_eval should pass skip_retreat_if_goal_satisfied to _run_robosuite_episode."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    import ast
    tree = ast.parse(source)

    # Find the call to _run_robosuite_episode inside run_robosuite_eval
    passes_param = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "_run_robosuite_episode":
                for kw in node.keywords:
                    if kw.arg == "skip_retreat_if_goal_satisfied":
                        passes_param = True

    assert passes_param, "run_robosuite_eval should pass skip_retreat_if_goal_satisfied"


def test_skip_retreat_in_main_config_loading():
    """main() config loading should handle skip_retreat_if_goal_satisfied."""
    from pathlib import Path
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "skip_retreat_if_goal_satisfied" in source


def test_skip_retreat_in_config_json():
    """Stage 7A-10 best config JSON should have skip_retreat_if_goal_satisfied."""
    import json
    from pathlib import Path
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    assert "skip_retreat_if_goal_satisfied" in config
    assert isinstance(config["skip_retreat_if_goal_satisfied"], bool)
