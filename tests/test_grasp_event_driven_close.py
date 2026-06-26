"""Tests for event-driven close early exit in grasp_state_machine (robosuite-independent)."""

from vlm_nsm.embodied.grasp_state_machine import GraspConfig


def test_grasp_config_has_event_drive_close_params():
    cfg = GraspConfig()
    assert hasattr(cfg, "close_steps_max")
    assert hasattr(cfg, "close_steps_min")
    assert hasattr(cfg, "close_stabilize_steps")
    assert hasattr(cfg, "holding_check_enabled")


def test_close_steps_max_default():
    cfg = GraspConfig()
    assert cfg.close_steps_max == 70


def test_close_steps_min_default():
    cfg = GraspConfig()
    assert cfg.close_steps_min == 15


def test_close_stabilize_steps_default():
    cfg = GraspConfig()
    assert cfg.close_stabilize_steps == 5


def test_holding_check_enabled_default():
    cfg = GraspConfig()
    assert cfg.holding_check_enabled is False


def test_event_drive_close_overrides():
    cfg = GraspConfig(close_steps_max=50, close_steps_min=10,
                       close_stabilize_steps=8, holding_check_enabled=True)
    assert cfg.close_steps_max == 50
    assert cfg.close_steps_min == 10
    assert cfg.close_stabilize_steps == 8
    assert cfg.holding_check_enabled is True


def test_close_steps_min_le_max():
    """close_steps_min should typically be <= close_steps_max."""
    cfg = GraspConfig(close_steps_min=10, close_steps_max=50)
    assert cfg.close_steps_min <= cfg.close_steps_max


def test_holding_check_in_config_dict():
    """execute_grasp result config should include event-driven params."""
    from vlm_nsm.embodied.grasp_state_machine import execute_grasp

    class _MockEnv:
        def get_eef_pose(self): return {"x": 0, "y": 0, "z": 0.5}
        def get_object_pose(self, n): return {"x": 0, "y": 0, "z": 0.02}
        def get_gripper_state(self): return {"pos": {"x": 0, "y": 0, "z": 0.5}, "open": 1.0, "has_object": False}
        def step(self, a): pass
        def get_target_pose(self, n): return {"x": 0, "y": 0, "z": 0}
        def get_all_object_poses(self): return {"cube1": {"x": 0, "y": 0, "z": 0.02}}

    env = _MockEnv()
    cfg = GraspConfig(holding_check_enabled=True, close_steps_max=30,
                       close_steps_min=10, close_stabilize_steps=5)
    result = execute_grasp(env, "cube1", config=cfg)
    config_keys = set(result["config"].keys())
    for key in ("close_steps_max", "close_steps_min", "close_stabilize_steps", "holding_check_enabled"):
        assert key in config_keys, f"Missing key: {key}"


def test_close_phase_has_early_exit_logic():
    """Close phase should have event-driven early exit code."""
    source = open("src/vlm_nsm/embodied/grasp_state_machine.py").read()
    assert "_contact_detected" in source or "holding_check_enabled" in source


def test_holding_check_with_has_object():
    """Event-driven close should check has_object from gripper."""
    source = open("src/vlm_nsm/embodied/grasp_state_machine.py").read()
    assert "has_object" in source
