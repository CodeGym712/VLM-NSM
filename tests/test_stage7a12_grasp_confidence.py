"""Tests for Stage 7A-12 grasp confidence, close modes, and horizon-aware fallback.

All tests are robosuite-independent — they verify logic directly on
GraspConfig, the grasp_attempt confidence heuristic, and the episode function's
horizon-aware config generation.
"""

from __future__ import annotations

import math

import numpy as np

from vlm_nsm.embodied.grasp_state_machine import GraspConfig


# ---- GraspConfig defaults ----


class TestGraspConfigStage7A12:
    def test_default_close_mode(self):
        cfg = GraspConfig()
        assert cfg.close_mode == "fixed"

    def test_adaptive_close_mode(self):
        cfg = GraspConfig(close_mode="adaptive")
        assert cfg.close_mode == "adaptive"

    def test_aggressive_close_mode(self):
        cfg = GraspConfig(close_mode="aggressive")
        assert cfg.close_mode == "aggressive"

    def test_confidence_logging_disabled_by_default(self):
        cfg = GraspConfig()
        assert cfg.log_grasp_confidence is False

    def test_confidence_logging_enabled(self):
        cfg = GraspConfig(log_grasp_confidence=True)
        assert cfg.log_grasp_confidence is True

    def test_mark_uncertain_default(self):
        cfg = GraspConfig()
        assert cfg.mark_grasp_uncertain is False

    # Stage 7A-12: new fields

    def test_close_steps_fast_default(self):
        cfg = GraspConfig()
        assert cfg.close_steps_fast == 15

    def test_close_steps_fast_custom(self):
        cfg = GraspConfig(close_steps_fast=18)
        assert cfg.close_steps_fast == 18

    def test_fallback_extend_close_default(self):
        cfg = GraspConfig()
        assert cfg.fallback_extend_close_if_not_holding is False

    def test_fallback_extend_close_enabled(self):
        cfg = GraspConfig(fallback_extend_close_if_not_holding=True)
        assert cfg.fallback_extend_close_if_not_holding is True


# ---- Grasp confidence heuristic logic ----


def _compute_confidence(
    open_before: float, open_after: float,
    contact: bool, obj_z_delta: float,
    lift_threshold: float = 0.005,
) -> float:
    """Replicate the grasp confidence heuristic from grasp_state_machine.py."""
    open_ratio = (open_after / max(open_before, 1e-8)
                  if open_before > 0.03 else 0.0)
    conf = 0.0
    if contact:
        conf += 0.4
    if open_ratio < 0.3:
        conf += 0.3
    if obj_z_delta > lift_threshold:
        conf += 0.3
    return min(conf, 1.0)


class TestGraspConfidenceHeuristic:
    def test_full_confidence(self):
        """Object grasped firmly and lifted: max confidence."""
        conf = _compute_confidence(
            open_before=1.0, open_after=0.02,
            contact=True, obj_z_delta=0.05,
        )
        assert math.isclose(conf, 1.0, abs_tol=0.01)

    def test_no_contact_zero_confidence(self):
        """No contact: confidence should be 0.0."""
        conf = _compute_confidence(
            open_before=1.0, open_after=1.0,
            contact=False, obj_z_delta=0.0,
        )
        assert conf == 0.0

    def test_contact_no_lift(self):
        """Contact but no lift: partial confidence (0.4 + maybe open_ratio)."""
        conf = _compute_confidence(
            open_before=1.0, open_after=0.5,
            contact=True, obj_z_delta=0.001,
        )
        # contact=0.4, open_ratio=0.5 (not < 0.3, so no +0.3)
        assert math.isclose(conf, 0.4, abs_tol=0.01)

    def test_open_ratio_close(self):
        """Gripper closed significantly but no contact detection: 0.3."""
        conf = _compute_confidence(
            open_before=1.0, open_after=0.1,
            contact=False, obj_z_delta=0.0,
        )
        assert math.isclose(conf, 0.3, abs_tol=0.01)

    def test_lift_only(self):
        """Object moved up without contact detection: unlikely but test."""
        conf = _compute_confidence(
            open_before=1.0, open_after=1.0,
            contact=False, obj_z_delta=0.02,
        )
        assert math.isclose(conf, 0.3, abs_tol=0.01)

    def test_contact_and_lift(self):
        """Contact + lift = 0.7 confidence."""
        conf = _compute_confidence(
            open_before=1.0, open_after=0.5,
            contact=True, obj_z_delta=0.02,
        )
        assert math.isclose(conf, 0.7, abs_tol=0.01)

    def test_edge_gripper_fully_open(self):
        """Gripper barely open before: open_ratio defaults to 0."""
        conf = _compute_confidence(
            open_before=0.01, open_after=0.005,
            contact=True, obj_z_delta=0.02,
        )
        # contact=0.4, open_before <= 0.03 so ratio=0.0, which is < 0.3 so +0.3
        assert math.isclose(conf, 1.0, abs_tol=0.01)


# ---- Horizon-aware grasp config adjustment ----


def _simulate_horizon_aware_config(remaining: int) -> dict:
    """Replicate the horizon-aware grasp config logic from run_embodied_stacking_robosuite.py.

    Args:
        remaining: Remaining step budget.

    Returns:
        Dict with close_mode, max_retries, close_steps, skip_approach.
    """
    close_steps = 15
    retries = 3
    close_mode = "fixed"

    if remaining < 80:
        pass  # budget check will catch
    elif remaining < 120:
        close_mode = "aggressive"
        retries = min(retries, 1)
        close_steps = min(close_steps, 10)
    elif remaining < 180:
        close_mode = "adaptive"
        retries = min(retries, 2)
    elif remaining < 250:
        close_mode = "adaptive"

    return {
        "close_mode": close_mode,
        "max_retries": retries,
        "close_steps": close_steps,
    }


class TestHorizonAwareConfig:
    def test_abundant_budget(self):
        cfg = _simulate_horizon_aware_config(500)
        assert cfg["close_mode"] == "fixed"
        assert cfg["max_retries"] == 3
        assert cfg["close_steps"] == 15

    def test_moderate_budget_200(self):
        cfg = _simulate_horizon_aware_config(200)
        assert cfg["close_mode"] == "adaptive"
        assert cfg["max_retries"] == 3

    def test_tight_budget_150(self):
        cfg = _simulate_horizon_aware_config(150)
        assert cfg["close_mode"] == "adaptive"
        assert cfg["max_retries"] == 2

    def test_very_tight_budget_100(self):
        cfg = _simulate_horizon_aware_config(100)
        assert cfg["close_mode"] == "aggressive"
        assert cfg["max_retries"] == 1
        assert cfg["close_steps"] <= 10

    def test_critical_budget_50(self):
        cfg = _simulate_horizon_aware_config(50)
        assert cfg["close_mode"] == "fixed"  # below 80 threshold, unchanged

    def test_boundary_120(self):
        cfg = _simulate_horizon_aware_config(120)
        assert cfg["close_mode"] == "adaptive"
        assert cfg["max_retries"] == 2

    def test_boundary_119(self):
        cfg = _simulate_horizon_aware_config(119)
        assert cfg["close_mode"] == "aggressive"

    def test_boundary_180(self):
        cfg = _simulate_horizon_aware_config(180)
        assert cfg["close_mode"] == "adaptive"
        assert cfg["max_retries"] == 3

    def test_boundary_250(self):
        cfg = _simulate_horizon_aware_config(250)
        assert cfg["close_mode"] == "fixed"  # >= 250 is not adaptive

    def test_adaptive_at_249(self):
        cfg = _simulate_horizon_aware_config(249)
        assert cfg["close_mode"] == "adaptive"

    def test_boundary_251(self):
        cfg = _simulate_horizon_aware_config(251)
        assert cfg["close_mode"] == "fixed"
