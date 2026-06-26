"""Tests for Stage 7A-10 robosuite execution config (robosuite-independent)."""

import json
from pathlib import Path

from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
    _REACH_THRESHOLD, _LIFT_HEIGHT, _GRASP_CLOSE_STEPS, _RELEASE_OPEN_STEPS,
)


def _apply_stage7a10_defaults(config: dict) -> dict:
    """Simulate config loading logic from main() for Stage 7A-10 fields."""
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
        "skip_retreat_if_goal_satisfied": config.get("skip_retreat_if_goal_satisfied", False),
        "terminate_after_safe_release": config.get("terminate_after_safe_release", False),
        "grasp_max_retries": config.get("grasp_max_retries", 3),
        "place_max_retries": config.get("place_max_retries", 2),
        "place_mode": config.get("place_mode", "simple"),
    }
    return effective


def test_stage7a10_config_file_exists():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    assert path.exists()
    config = json.loads(path.read_text())
    assert isinstance(config, dict)


def test_stage7a10_config_has_stage7a9_keys():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    for key in ["kp", "max_delta", "reach_threshold", "grasp_close_steps",
                 "lift_height", "release_open_steps", "early_stop_on_goal",
                 "use_place_state_machine", "horizon_aware", "place_retry"]:
        assert key in config, f"Missing key: {key}"


def test_stage7a10_config_has_new_keys():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    for key in ["skip_retreat_if_goal_satisfied", "terminate_after_safe_release",
                 "grasp_max_retries", "place_max_retries", "place_mode"]:
        assert key in config, f"Missing Stage 7A-10 key: {key}"


def test_new_config_values_in_range():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    assert 0.01 <= config["kp"] <= 0.5
    assert 0.001 <= config["max_delta"] <= 0.1
    assert 10 <= config.get("grasp_close_steps", 50) <= 200
    assert config["place_mode"] in ("simple", "fast", "safe")
    assert isinstance(config["skip_retreat_if_goal_satisfied"], bool)
    assert isinstance(config["terminate_after_safe_release"], bool)
    assert 1 <= config.get("grasp_max_retries", 3) <= 10
    assert 1 <= config.get("place_max_retries", 2) <= 10


def test_defaults_for_new_fields():
    effective = _apply_stage7a10_defaults({})
    assert effective["skip_retreat_if_goal_satisfied"] is False
    assert effective["terminate_after_safe_release"] is False
    assert effective["grasp_max_retries"] == 3
    assert effective["place_max_retries"] == 2
    assert effective["place_mode"] == "simple"


def test_override_new_fields():
    cfg = {"skip_retreat_if_goal_satisfied": True, "place_mode": "fast",
           "grasp_max_retries": 5}
    effective = _apply_stage7a10_defaults(cfg)
    assert effective["skip_retreat_if_goal_satisfied"] is True
    assert effective["place_mode"] == "fast"
    assert effective["grasp_max_retries"] == 5
    assert effective["place_max_retries"] == 2  # default


def test_place_mode_choices():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    assert config["place_mode"] in ("simple", "fast", "safe")


def test_skip_retreat_requires_bool():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    assert isinstance(config.get("skip_retreat_if_goal_satisfied"), bool)


def test_terminate_release_requires_bool():
    path = Path("generated/configs/stage_7A-10/robosuite_best_execution_config_stage7a10.json")
    config = json.loads(path.read_text())
    assert isinstance(config.get("terminate_after_safe_release"), bool)
