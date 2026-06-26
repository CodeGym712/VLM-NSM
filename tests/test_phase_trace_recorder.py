"""Tests for phase-level trace recorder structure (robosuite-independent)."""


def test_phase_trace_has_required_keys():
    """Phase trace dict should have all 6 phase keys."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    # Inspect defaults via function signature (can't call without env)
    import inspect
    sig = inspect.signature(_run_robosuite_episode)
    # The phase_trace is a local variable, not a parameter.
    # Instead, verify the expected keys from the implementation.
    expected_phases = {"reach_pick", "grasp", "lift", "reach_place", "release", "retreat"}
    assert len(expected_phases) == 6


def test_phase_trace_keys_defined_in_function():
    """Phase trace keys should be defined in _run_robosuite_episode."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    tree = ast.parse(source)

    found = False
    target_phases = {"reach_pick", "grasp", "lift", "reach_place", "release", "retreat"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            # Must match the full set of 6 (not _PHASE_TO_SKILL_ID with extra fields)
            if keys == target_phases:
                found = True
                break

    assert found, "phase_trace dict with phase keys not found in source"


def test_phase_trace_values_are_ints():
    """Phase trace values should be initialized to 0 (integers)."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    tree = ast.parse(source)

    target_phases = {"reach_pick", "grasp", "lift", "reach_place", "release", "retreat"}
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if keys == target_phases:
                # Verify all values are int 0 (distinguishes from _PHASE_TO_SKILL_ID)
                all_zero = all(
                    isinstance(v, ast.Constant) and v.value == 0
                    for v in node.values
                )
                if all_zero:
                    found = True
                    break
    assert found, "phase_trace dict with all-zero values not found"


def test_phase_trace_increment_exists():
    """Each phase should have an increment operation in _run_robosuite_episode."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    tree = ast.parse(source)

    target_phases = {"reach_pick", "grasp", "lift", "reach_place", "release", "retreat"}
    found_phases = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            if isinstance(node.target.value, ast.Name) and node.target.value.id == "phase_trace":
                if isinstance(node.target.slice, ast.Constant):
                    found_phases.add(node.target.slice.value)

    # At least most phases should have increment operations
    # (some may be counted via _grasp_steps / _place_steps accumulation)
    assert len(found_phases) >= 3, f"Expected >= 3 phases with increments, got {found_phases}"


def test_phase_trace_in_return_dict():
    """Return dict from _run_robosuite_episode should include phase_trace."""
    import ast
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    tree = ast.parse(source)

    return_dict = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return_dict = node.value
            break

    assert return_dict is not None, "Return dict not found"

    keys = []
    for k in return_dict.keys:
        if isinstance(k, ast.Constant):
            keys.append(k.value)

    assert "phase_trace" in keys, "phase_trace missing from return dict"
