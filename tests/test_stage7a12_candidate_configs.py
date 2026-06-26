"""Tests for Stage 7A-12 candidate config files (robosuite-independent).

Verifies that all three candidate configs exist, have valid JSON,
contain expected fields with correct types and ranges.
"""

import json
from pathlib import Path

_CONFIG_DIR = Path("generated/configs/stage_7A-12")

CANDIDATE_FILES = [
    "robosuite_stage7a12_candidate_A_fixed15.json",
    "robosuite_stage7a12_candidate_B_adaptive_10_15_25.json",
    "robosuite_stage7a12_candidate_C_adaptive_10_18_28.json",
]

REQUIRED_KEYS = [
    "kp", "max_delta", "reach_threshold",
    "grasp_close_steps", "close_mode",
    "early_stop_on_goal", "use_place_state_machine",
    "horizon_aware", "place_retry",
    "grasp_max_retries", "place_max_retries", "place_mode",
    "approach_height", "descend_offset", "lift_height",
    "lift_early_exit_enabled",
    "release_steps_min", "release_steps_max", "post_release_stable_steps",
]

BOOLEAN_KEYS = [
    "early_stop_on_goal", "use_place_state_machine",
    "horizon_aware", "place_retry",
    "lift_early_exit_enabled",
]


def _load_config(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    assert path.exists(), f"Config file not found: {path}"
    return json.loads(path.read_text())


class TestCandidateConfigsExist:
    def test_all_candidates_exist(self):
        for fname in CANDIDATE_FILES:
            assert (_CONFIG_DIR / fname).exists(), f"Missing: {fname}"

    def test_all_valid_json(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            for key in REQUIRED_KEYS:
                assert key in cfg, f"{fname}: missing key '{key}'"

    def test_boolean_keys(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            for key in BOOLEAN_KEYS:
                assert isinstance(cfg[key], bool), f"{fname}: {key} should be bool"

    def test_numeric_ranges(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            assert 0.01 <= cfg["kp"] <= 0.5, f"{fname}: kp out of range"
            assert 0.001 <= cfg["max_delta"] <= 0.1, f"{fname}: max_delta out of range"
            assert 0.001 <= cfg["reach_threshold"] <= 0.1, f"{fname}: reach_threshold out of range"
            assert 10 <= cfg["grasp_close_steps"] <= 200, f"{fname}: grasp_close_steps out of range"
            assert 1 <= cfg["grasp_max_retries"] <= 5, f"{fname}: grasp_max_retries out of range"

    def test_close_mode_valid(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            assert cfg["close_mode"] in ("fixed", "adaptive", "aggressive"), \
                f"{fname}: close_mode={cfg['close_mode']}"

    def test_place_mode_valid(self):
        for fname in CANDIDATE_FILES:
            cfg = _load_config(fname)
            assert cfg["place_mode"] in ("simple", "fast", "safe"), \
                f"{fname}: place_mode={cfg['place_mode']}"


class TestCandidateA:
    """Candidate A: fixed15 — fixed close mode, close_steps=15, no uncertainty marking."""

    def test_close_mode_fixed(self):
        cfg = _load_config(CANDIDATE_FILES[0])
        assert cfg["close_mode"] == "fixed"

    def test_grasp_close_steps_15(self):
        cfg = _load_config(CANDIDATE_FILES[0])
        assert cfg["grasp_close_steps"] == 15

    def test_mark_uncertain_false(self):
        cfg = _load_config(CANDIDATE_FILES[0])
        assert cfg.get("mark_grasp_uncertain", False) is False

    def test_holding_check_disabled(self):
        cfg = _load_config(CANDIDATE_FILES[0])
        assert cfg.get("holding_check_enabled", False) is False


class TestCandidateB:
    """Candidate B: adaptive_10_15_25 — adaptive close, close_steps_fast=15."""

    def test_close_mode_adaptive(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg["close_mode"] == "adaptive"

    def test_close_steps_min_10(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg["close_steps_min"] == 10

    def test_close_steps_fast_15(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg.get("close_steps_fast") == 15

    def test_close_steps_max_25(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg["close_steps_max"] == 25

    def test_fallback_extend_close_enabled(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg.get("fallback_extend_close_if_not_holding", False) is True

    def test_mark_uncertain_true(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg.get("mark_grasp_uncertain", False) is True

    def test_holding_check_enabled(self):
        cfg = _load_config(CANDIDATE_FILES[1])
        assert cfg.get("holding_check_enabled", False) is True


class TestCandidateC:
    """Candidate C: adaptive_10_18_28 — adaptive close, close_steps_fast=18."""

    def test_close_mode_adaptive(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg["close_mode"] == "adaptive"

    def test_close_steps_min_10(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg["close_steps_min"] == 10

    def test_close_steps_fast_18(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg.get("close_steps_fast") == 18

    def test_close_steps_max_28(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg["close_steps_max"] == 28

    def test_fallback_extend_close_enabled(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg.get("fallback_extend_close_if_not_holding", False) is True

    def test_mark_uncertain_true(self):
        cfg = _load_config(CANDIDATE_FILES[2])
        assert cfg.get("mark_grasp_uncertain", False) is True
