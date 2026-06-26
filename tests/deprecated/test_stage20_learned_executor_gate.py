"""Stage 20-U6: Tests for LearnedExecutorGate."""
import numpy as np
import pytest

from vlm_nsm.embodied.stage20_learned_executor_gate import (
    LearnedExecutorGate, GateConfig, GateResult,
)


class TestLearnedExecutorGate:
    def test_gate_passes_normal_action(self):
        gate = LearnedExecutorGate()
        action = np.array([0.1, 0.0, 0.05, 0.0, 0.0, 0.0, -0.5], dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=True)
        assert result.passed is True
        assert result.reason == "ok"

    def test_gate_fails_non_finite(self):
        gate = LearnedExecutorGate()
        action = np.array([0.1, np.nan, 0.05, 0.0, 0.0, 0.0, -0.5], dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=True)
        assert result.passed is False
        assert "non_finite" in result.reason

    def test_gate_fails_not_warm(self):
        gate = LearnedExecutorGate()
        action = np.zeros(7, dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=False)
        assert result.passed is False
        assert "not_warm" in result.reason

    def test_gate_fails_norm_exceeded(self):
        gate = LearnedExecutorGate(GateConfig(action_norm_max=0.5))
        action = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=True)
        assert result.passed is False
        assert "norm" in result.reason

    def test_gate_fails_dim6_range(self):
        gate = LearnedExecutorGate(GateConfig(action_dim6_max=1.0))
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=True)
        assert result.passed is False
        assert "dim6" in result.reason

    def test_gate_fails_dim_abs_max(self):
        gate = LearnedExecutorGate(GateConfig(action_dim_abs_max=1.0))
        action = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = gate.check(action, phase_id=0, is_warm=True)
        assert result.passed is False

    def test_gate_phase_override(self):
        gate = LearnedExecutorGate(GateConfig(
            phase_action_norm_max={1: 0.3}))
        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Phase 0: norm 0.5 should pass (default max is 3.0)
        r0 = gate.check(action, phase_id=0, is_warm=True)
        assert r0.passed is True
        # Phase 1: norm 0.5 should fail (override max is 0.3)
        r1 = gate.check(action, phase_id=1, is_warm=True)
        assert r1.passed is False

    def test_trigger_counts(self):
        gate = LearnedExecutorGate()
        gate.check(np.zeros(7), phase_id=0, is_warm=True)
        gate.check(np.array([np.nan]*7), phase_id=0, is_warm=True)
        counts = gate.trigger_counts
        assert counts.get("passed", 0) == 1
        assert counts.get("non_finite_action", 0) == 1

    def test_high_risk_d1p1(self):
        # Use phase override to allow larger dim6, then high_risk catches it
        gate = LearnedExecutorGate(GateConfig(
            phase_dim6_max={1: 2.0},  # allow up to 2.0 for phase 1
            high_risk_dim6_abs_max=1.0,  # but d1p1 high-risk caps at 1.0
        ))
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5], dtype=np.float32)
        result = gate.check(action, phase_id=1, direction_id=1, is_warm=True)
        assert result.passed is False
        assert "high_risk" in result.reason
