"""Tests for best config JSON loading with defaults (robosuite-independent)."""

import json
import tempfile
from pathlib import Path

from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
    _REACH_THRESHOLD,
    _LIFT_HEIGHT,
    _GRASP_CLOSE_STEPS,
    _RELEASE_OPEN_STEPS,
)


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _apply_config_defaults(config: dict) -> dict:
    """Simulate the config loading logic from main()."""
    effective = {
        "kp": config.get("kp", 0.05),
        "max_delta": config.get("max_delta", 0.02),
        "reach_threshold": config.get("reach_threshold", _REACH_THRESHOLD),
        "grasp_close_steps": config.get("grasp_close_steps", _GRASP_CLOSE_STEPS),
        "lift_height": config.get("lift_height", _LIFT_HEIGHT),
        "release_open_steps": config.get("release_open_steps", _RELEASE_OPEN_STEPS),
        "early_stop_on_goal": config.get("early_stop_on_goal", True),
        "use_place_state_machine": config.get("use_place_state_machine", True),
        "horizon_aware": config.get("horizon_aware", True),
        "place_retry": config.get("place_retry", True),
    }
    return effective


def test_best_config_file_exists():
    """Best config JSON file should exist and be valid JSON."""
    path = Path("generated/configs/stage_7A-9/robosuite_best_execution_config_stage7a9.json")
    assert path.exists()
    config = json.loads(path.read_text())
    assert isinstance(config, dict)


def test_best_config_has_required_keys():
    """Best config should have all required fields."""
    path = Path("generated/configs/stage_7A-9/robosuite_best_execution_config_stage7a9.json")
    config = json.loads(path.read_text())
    required = [
        "kp", "max_delta", "reach_threshold",
        "grasp_close_steps", "lift_height", "release_open_steps",
        "early_stop_on_goal", "use_place_state_machine",
        "horizon_aware", "place_retry",
    ]
    for key in required:
        assert key in config, f"Missing required key: {key}"


def test_best_config_values_in_range():
    """Best config values should be in reasonable ranges."""
    path = Path("generated/configs/stage_7A-9/robosuite_best_execution_config_stage7a9.json")
    config = json.loads(path.read_text())
    assert 0.01 <= config["kp"] <= 0.5
    assert 0.001 <= config["max_delta"] <= 0.1
    assert 0.001 <= config["reach_threshold"] <= 0.1
    assert 10 <= config["grasp_close_steps"] <= 200
    assert 0.01 <= config["lift_height"] <= 0.5
    assert 10 <= config["release_open_steps"] <= 200


def test_defaults_applied_for_missing_keys():
    """Missing config keys should fall back to module defaults."""
    config = {}
    effective = _apply_config_defaults(config)
    assert effective["kp"] == 0.05
    assert effective["max_delta"] == 0.02
    assert effective["reach_threshold"] == _REACH_THRESHOLD
    assert effective["early_stop_on_goal"] is True


def test_config_overrides_defaults():
    """Config values should override defaults."""
    config = {"kp": 0.08, "max_delta": 0.015, "early_stop_on_goal": False}
    effective = _apply_config_defaults(config)
    assert effective["kp"] == 0.08
    assert effective["max_delta"] == 0.015
    assert effective["early_stop_on_goal"] is False


def test_partial_config():
    """Partial config should fill missing fields from defaults."""
    config = {"kp": 0.1}
    effective = _apply_config_defaults(config)
    assert effective["kp"] == 0.1
    assert effective["max_delta"] == 0.02  # default


def test_config_with_file():
    """Loading from a temp file should work."""
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "test_config.json"
        data = {"kp": 0.08, "max_delta": 0.015, "early_stop_on_goal": True}
        cfg_path.write_text(json.dumps(data))
        loaded = _load_config(cfg_path)
        assert loaded["kp"] == 0.08
        assert loaded["max_delta"] == 0.015
