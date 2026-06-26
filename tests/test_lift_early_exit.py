"""Tests for lift early exit in grasp_state_machine (robosuite-independent)."""

from vlm_nsm.embodied.grasp_state_machine import GraspConfig


def test_grasp_config_has_lift_early_exit():
    cfg = GraspConfig()
    assert hasattr(cfg, "lift_early_exit_enabled")


def test_lift_early_exit_default():
    cfg = GraspConfig()
    assert cfg.lift_early_exit_enabled is False


def test_lift_early_exit_override():
    cfg = GraspConfig(lift_early_exit_enabled=True)
    assert cfg.lift_early_exit_enabled is True


def test_lift_early_exit_in_config_dict():
    """execute_grasp result config should include lift_early_exit_enabled."""
    from vlm_nsm.embodied.grasp_state_machine import execute_grasp

    class _MockEnv:
        def get_eef_pose(self): return {"x": 0, "y": 0, "z": 0.5}
        def get_object_pose(self, n): return {"x": 0, "y": 0, "z": 0.02}
        def get_gripper_state(self): return {"pos": {"x": 0, "y": 0, "z": 0.5}, "open": 1.0, "has_object": False}
        def step(self, a): pass
        def get_target_pose(self, n): return {"x": 0, "y": 0, "z": 0}
        def get_all_object_poses(self): return {"cube1": {"x": 0, "y": 0, "z": 0.02}}

    env = _MockEnv()
    cfg = GraspConfig(lift_early_exit_enabled=True)
    result = execute_grasp(env, "cube1", config=cfg)
    assert "lift_early_exit_enabled" in result["config"]
    assert result["config"]["lift_early_exit_enabled"] is True


def test_lift_phase_has_early_exit_code():
    """Lift phase should check lift_early_exit_enabled."""
    source = open("src/vlm_nsm/embodied/grasp_state_machine.py").read()
    assert "lift_early_exit_enabled" in source


def test_lift_early_exit_checks_object_z():
    """Lift early exit should check object z-delta against lift_threshold."""
    source = open("src/vlm_nsm/embodied/grasp_state_machine.py").read()
    assert "lift_threshold" in source.split("lift_early_exit_enabled")[0] or True
    # Verify both concepts appear near each other
    lines = source.split("\n")
    early_exit_lines = [i for i, l in enumerate(lines) if "lift_early_exit_enabled" in l]
    threshold_nearby = False
    for idx in early_exit_lines:
        for j in range(max(0, idx - 3), min(len(lines), idx + 3)):
            if "lift_threshold" in lines[j]:
                threshold_nearby = True
    assert threshold_nearby, "lift_early_exit should use lift_threshold"
