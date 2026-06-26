"""Tests for grasp/place state machine config propagation (robosuite-independent)."""

from vlm_nsm.embodied.grasp_state_machine import GraspConfig, execute_grasp, grasp_attempt
from vlm_nsm.embodied.place_state_machine import PlaceConfig, _PLACE_MODE_PRESETS, _resolve_place_preset


class _MockEnv:
    """Minimal mock env for config propagation tests (no robosuite dependency)."""
    def __init__(self):
        self._eef = {"x": 0.0, "y": 0.0, "z": 0.5}
        self._poses = {"cube1": {"x": 0.1, "y": 0.1, "z": 0.02}}
        self._gripper = {"pos": {"x": 0.0, "y": 0.0, "z": 0.5}, "open": 1.0, "has_object": False}

    def get_eef_pose(self): return self._eef
    def get_object_pose(self, name): return self._poses.get(name, {"x": 0, "y": 0, "z": 0})
    def get_gripper_state(self): return self._gripper
    def step(self, action): pass
    def get_target_pose(self, name): return {"x": 0.0, "y": 0.0, "z": 0.0}
    def get_all_object_poses(self): return self._poses


def test_grasp_config_defaults():
    cfg = GraspConfig()
    assert cfg.kp == 0.05
    assert cfg.max_delta == 0.02
    assert cfg.close_steps == 70
    assert cfg.max_retries == 3
    assert cfg.reach_threshold == 0.015
    assert cfg.lift_height == 0.05
    assert cfg.lift_threshold == 0.005


def test_grasp_config_overrides():
    cfg = GraspConfig(kp=0.08, max_delta=0.015, close_steps=50, max_retries=5,
                       reach_threshold=0.02, lift_height=0.1, lift_threshold=0.01)
    assert cfg.kp == 0.08
    assert cfg.max_delta == 0.015
    assert cfg.close_steps == 50
    assert cfg.max_retries == 5
    assert cfg.reach_threshold == 0.02
    assert cfg.lift_height == 0.1
    assert cfg.lift_threshold == 0.01


def test_grasp_config_partial():
    cfg = GraspConfig(kp=0.1)
    assert cfg.kp == 0.1
    assert cfg.close_steps == 70  # default
    assert cfg.max_retries == 3  # default


def test_execute_grasp_accepts_config():
    """execute_grasp should accept a GraspConfig without error."""
    env = _MockEnv()
    cfg = GraspConfig(kp=0.08, close_steps=50, max_retries=2)
    # This will fail at runtime because the mock env is incomplete,
    # but the config acceptance and propagation should not error.
    result = execute_grasp(env, "cube1", config=cfg)
    assert "config" in result
    assert result["config"]["kp"] == 0.08
    assert result["config"]["close_steps"] == 50
    assert result["config"]["max_retries"] == 2
    assert "lift_threshold" in result["config"]
    assert result["config"]["lift_threshold"] == 0.005


def test_grasp_attempt_accepts_config():
    env = _MockEnv()
    cfg = GraspConfig(kp=0.08, descend_offset=0.01)
    success, diag = grasp_attempt(env, "cube1", config=cfg)
    # Will fail on approach but should not raise
    assert isinstance(success, bool)


def test_place_config_defaults():
    cfg = PlaceConfig()
    assert cfg.kp == 0.05
    assert cfg.max_delta == 0.02
    assert cfg.preplace_height == 0.15
    assert cfg.descend_offset == 0.02
    assert cfg.xy_micro_align_steps == 30
    assert cfg.max_retries == 2
    assert cfg.hold_steps == 10


def test_place_config_overrides():
    cfg = PlaceConfig(kp=0.08, max_delta=0.015, preplace_height=0.10,
                       descend_offset=0.01, max_retries=3)
    assert cfg.kp == 0.08
    assert cfg.max_delta == 0.015
    assert cfg.preplace_height == 0.10
    assert cfg.descend_offset == 0.01
    assert cfg.max_retries == 3


def test_place_mode_simple():
    cfg = PlaceConfig(mode="simple")
    assert cfg.preplace_height == 0.15
    assert cfg.xy_micro_align_steps == 30
    assert cfg.hold_steps == 10


def test_place_mode_fast():
    cfg = PlaceConfig(mode="fast")
    assert cfg.preplace_height == 0.08
    assert cfg.xy_micro_align_steps == 10
    assert cfg.hold_steps == 5


def test_place_mode_safe():
    cfg = PlaceConfig(mode="safe")
    assert cfg.preplace_height == 0.15
    assert cfg.descend_offset == 0.01
    assert cfg.hold_steps == 20
    assert cfg.xy_micro_align_steps == 50


def test_place_mode_with_override():
    """Mode preset should be overridable by explicit kwargs."""
    cfg = PlaceConfig(mode="fast", preplace_height=0.12)
    assert cfg.preplace_height == 0.12  # override
    assert cfg.xy_micro_align_steps == 10  # from fast preset


def test_place_mode_unknown_falls_back_to_simple():
    cfg = PlaceConfig(mode="unknown")
    assert cfg.preplace_height == 0.15  # simple default
    assert cfg.hold_steps == 10


def test_place_presets_have_expected_keys():
    for mode in ("simple", "fast", "safe"):
        preset = _PLACE_MODE_PRESETS[mode]
        for key in ("preplace_height", "descend_offset", "xy_align_threshold",
                     "hold_steps", "xy_micro_align_steps"):
            assert key in preset, f"Missing {key} in {mode}"


def test_resolve_place_preset():
    resolved = _resolve_place_preset("fast", hold_steps=8)
    assert resolved["preplace_height"] == 0.08
    assert resolved["hold_steps"] == 8  # override


def test_execute_grasp_config_dict_keys():
    """execute_grasp result config should include lift_threshold."""
    env = _MockEnv()
    cfg = GraspConfig()
    result = execute_grasp(env, "cube1", config=cfg)
    config_keys = set(result["config"].keys())
    expected = {"kp", "max_delta", "approach_height", "descend_offset",
                "close_steps", "max_retries", "lift_height", "lift_threshold"}
    assert expected.issubset(config_keys), f"Missing keys: {expected - config_keys}"
