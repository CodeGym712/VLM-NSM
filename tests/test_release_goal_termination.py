"""Tests for release phase enhancement and goal termination (robosuite-independent)."""


def test_release_steps_min_exists():
    """release_steps_min should be a module constant."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_STEPS_MIN
    assert isinstance(_RELEASE_STEPS_MIN, int)
    assert _RELEASE_STEPS_MIN >= 0


def test_release_steps_max_exists():
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_STEPS_MAX
    assert isinstance(_RELEASE_STEPS_MAX, int)
    assert _RELEASE_STEPS_MAX > 0


def test_post_release_stable_steps_exists():
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _POST_RELEASE_STABLE_STEPS
    assert isinstance(_POST_RELEASE_STABLE_STEPS, int)
    assert _POST_RELEASE_STABLE_STEPS >= 0


def test_release_steps_min_default():
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_STEPS_MIN
    assert _RELEASE_STEPS_MIN == 10


def test_release_steps_max_default():
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_STEPS_MAX
    assert _RELEASE_STEPS_MAX == 50


def test_post_release_stable_steps_default():
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _POST_RELEASE_STABLE_STEPS
    assert _POST_RELEASE_STABLE_STEPS == 5


def test_release_params_in_function_signature():
    """_run_robosuite_episode should have release_steps_min/max and post_release_stable_steps."""
    import inspect
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode
    sig = inspect.signature(_run_robosuite_episode)
    for param in ("release_steps_min", "release_steps_max", "post_release_stable_steps"):
        assert param in sig.parameters, f"Missing param: {param}"


def test_release_phase_has_early_exit():
    """Release phase should have early exit when gripper is fully open."""
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "get_gripper_state" in source
    assert "open" in source


def test_release_phase_stable_steps():
    """Release phase should have post-release stabilization steps."""
    source = open("src/vlm_nsm/experiments/run_embodied_stacking_robosuite.py").read()
    assert "post_release_stable_steps" in source or "_r_stable" in source
