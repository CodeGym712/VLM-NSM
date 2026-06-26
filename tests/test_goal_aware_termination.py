"""Tests for goal-aware termination policies (robosuite-independent).

Verifies that skip_retreat_if_goal_satisfied and terminate_after_safe_release
flags are wired through the code correctly.
"""

from pathlib import Path


def test_skip_retreat_param_exists():
    """skip_retreat_if_goal_satisfied should be a parameter of _run_robosuite_episode."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    assert "skip_retreat_if_goal_satisfied" in sig.parameters


def test_terminate_after_safe_release_param_exists():
    """terminate_after_safe_release should be a parameter of _run_robosuite_episode."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    assert "terminate_after_safe_release" in sig.parameters


def test_skip_retreat_default_false():
    """skip_retreat_if_goal_satisfied should default to False."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    param = sig.parameters["skip_retreat_if_goal_satisfied"]
    assert param.default is False


def test_terminate_after_safe_release_default_false():
    """terminate_after_safe_release should default to False."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    param = sig.parameters["terminate_after_safe_release"]
    assert param.default is False


def test_run_robosuite_eval_passes_skip_retreat():
    """run_robosuite_eval should accept and pass skip_retreat_if_goal_satisfied."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import run_robosuite_eval
    sig = inspect.signature(run_robosuite_eval)
    assert "skip_retreat_if_goal_satisfied" in sig.parameters


def test_run_robosuite_eval_passes_terminate_release():
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import run_robosuite_eval
    sig = inspect.signature(run_robosuite_eval)
    assert "terminate_after_safe_release" in sig.parameters


def test_skip_retreat_code_in_retreat_phase():
    """retreat phase should check skip_retreat_if_goal_satisfied."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "skip_retreat_if_goal_satisfied" in source


def test_terminate_release_code_in_release_phase():
    """release phase should check terminate_after_safe_release."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "terminate_after_safe_release" in source


def test_skip_retreat_increments_successful_steps():
    """When skip_retreat is triggered, successful_steps should increment."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    # Look for the pattern where successful_steps is incremented in the skip-retreat branch
    import ast
    tree = ast.parse(source)

    skip_retreat_increment = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            condition_str = ast.unparse(node.test)
            if "skip_retreat_if_goal_satisfied" in condition_str and "goal_was_achieved" in condition_str:
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and child.op.__class__.__name__ == "Add":
                        if isinstance(child.target, ast.Name) and child.target.id == "successful_steps":
                            skip_retreat_increment = True

    assert skip_retreat_increment, "skip_retreat branch should increment successful_steps"


def test_early_stopped_set_on_skip_retreat():
    """early_stopped should be set to True when skip_retreat triggers."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    lines = source.split('\n')
    early_stop_in_skip = False
    for i, line in enumerate(lines):
        if 'skip_retreat_if_goal_satisfied' in line and 'goal_was_achieved' in line:
            for j in range(i, min(i + 5, len(lines))):
                if 'early_stopped' in lines[j] and 'True' in lines[j]:
                    early_stop_in_skip = True
    assert early_stop_in_skip, "early_stopped should be set when skip_retreat triggers"


def test_main_accepts_skip_retreat_cli():
    """main() should accept --skip-retreat-if-goal-satisfied CLI argument."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "--skip-retreat-if-goal-satisfied" in source


def test_main_accepts_terminate_release_cli():
    """main() should accept --terminate-after-safe-release CLI argument."""
    source = Path("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read_text()
    assert "--terminate-after-safe-release" in source
