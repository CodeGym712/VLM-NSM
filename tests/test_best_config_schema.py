"""Tests for best config schema validation (robosuite-independent)."""

import json
import re
from pathlib import Path


_CONFIG_PATH = Path("generated/configs/stage_7A-9/robosuite_best_execution_config_stage7a9.json")


def _load_best_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text())


def test_config_schema_numeric_fields():
    """All numeric fields should be numbers."""
    config = _load_best_config()
    numeric_keys = ["kp", "max_delta", "reach_threshold",
                    "grasp_close_steps", "lift_height", "release_open_steps"]
    for key in numeric_keys:
        val = config.get(key)
        assert isinstance(val, (int, float)), f"{key} should be numeric, got {type(val)}"


def test_config_schema_boolean_fields():
    """All boolean fields should be bools."""
    config = _load_best_config()
    bool_keys = ["early_stop_on_goal", "use_place_state_machine",
                 "horizon_aware", "place_retry"]
    for key in bool_keys:
        val = config.get(key)
        assert isinstance(val, bool), f"{key} should be bool, got {type(val)}"


def test_config_kp_positive():
    """kp must be positive."""
    assert _load_best_config()["kp"] > 0


def test_config_max_delta_positive():
    """max_delta must be positive."""
    assert _load_best_config()["max_delta"] > 0


def test_config_reach_threshold_positive():
    """reach_threshold must be positive."""
    assert _load_best_config()["reach_threshold"] > 0


def test_config_kp_max_delta_ratio():
    """kp * reach_distance should not exceed max_delta for stable control."""
    config = _load_best_config()
    kp = config["kp"]
    max_delta = config["max_delta"]
    # At 0.1m distance, kp * 0.1 should produce a delta <= max_delta
    # Otherwise the controller would saturate even at close range
    assert kp * 0.1 <= max_delta * 1.5, (
        f"kp={kp} * 0.1 = {kp*0.1} exceeds max_delta={max_delta} * 1.5 = {max_delta*1.5}"
    )


def test_config_grasp_close_steps_range():
    """grasp_close_steps should be in [10, 200]."""
    val = _load_best_config()["grasp_close_steps"]
    assert 10 <= val <= 200


def test_config_release_open_steps_range():
    """release_open_steps should be in [10, 200]."""
    val = _load_best_config()["release_open_steps"]
    assert 10 <= val <= 200


def test_config_lift_height_range():
    """lift_height should be in [0.01, 0.5]."""
    val = _load_best_config()["lift_height"]
    assert 0.01 <= val <= 0.5


def test_config_schema_no_extra_keys():
    """Config should not contain unexpected keys."""
    config = _load_best_config()
    allowed = {
        "kp", "max_delta", "reach_threshold",
        "grasp_close_steps", "lift_height", "release_open_steps",
        "early_stop_on_goal", "use_place_state_machine",
        "horizon_aware", "place_retry",
    }
    extra = set(config.keys()) - allowed
    assert not extra, f"Unexpected config keys: {extra}"


def test_config_schema_key_naming():
    """All keys should use snake_case."""
    config = _load_best_config()
    snake_case = re.compile(r'^[a-z]+(_[a-z]+)*$')
    for key in config:
        assert snake_case.match(key), f"Key '{key}' is not snake_case"
