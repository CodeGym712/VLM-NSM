"""Tests for Stage 7A-11 config schema (robosuite-independent)."""

import json
from pathlib import Path


def test_candidate_config_exists():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    assert path.exists()


def test_candidate_config_is_valid_json():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert isinstance(config, dict)


def test_candidate_has_stage7a11_params():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    for key in ["holding_check_enabled", "close_steps_min", "close_steps_max",
                 "close_stabilize_steps", "lift_early_exit_enabled",
                 "release_steps_min", "release_steps_max", "post_release_stable_steps"]:
        assert key in config, f"Missing key: {key}"


def test_candidate_close_steps_valid():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert 5 <= config.get("close_steps_min", 0) <= config.get("close_steps_max", 999)
    assert config.get("close_steps_max", 0) <= 200


def test_candidate_holding_check_is_bool():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert isinstance(config.get("holding_check_enabled"), bool)


def test_candidate_lift_early_exit_is_bool():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert isinstance(config.get("lift_early_exit_enabled"), bool)


def test_candidate_release_steps_valid():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert 0 <= config.get("release_steps_min", 0) <= config.get("release_steps_max", 999)
    assert config.get("release_steps_max", 0) <= 200


def test_candidate_post_release_stable_non_negative():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    assert config.get("post_release_stable_steps", -1) >= 0


def test_candidate_has_stage7a10_params():
    path = Path("generated/configs/stage_7A-11/robosuite_best_execution_config_stage7a11_candidate.json")
    config = json.loads(path.read_text())
    for key in ["kp", "max_delta", "reach_threshold", "early_stop_on_goal",
                 "skip_retreat_if_goal_satisfied", "place_mode"]:
        assert key in config, f"Missing key: {key}"
