"""Tests for NeuroSymbolicSafetyGate."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.neuro_symbolic.safety_gate import (
    GateConfig,
    GateResult,
    GateStatistics,
    NeuroSymbolicSafetyGate,
)


def _action(dx=0.0, dy=0.0, dz=0.0, gripper=0.0):
    """Build a 7-dim action."""
    return np.array([dx, dy, dz, 0.0, 0.0, 0.0, gripper], dtype=np.float32)


class TestGateConfig:
    """Tests for GateConfig."""

    def test_defaults(self):
        cfg = GateConfig()
        assert cfg.vla_confidence_threshold == 0.6
        assert cfg.max_xyz_deviation == 0.02
        assert cfg.enable_phase_constraints is True

    def test_serialization_roundtrip(self):
        cfg = GateConfig(
            vla_confidence_threshold=0.7,
            max_xyz_deviation=0.015,
            table_z=0.82,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate_config.json"
            cfg.to_json(path)
            loaded = GateConfig.from_json(path)
            assert loaded.vla_confidence_threshold == 0.7
            assert loaded.max_xyz_deviation == 0.015
            assert loaded.table_z == 0.82

    def test_from_dict(self):
        data = {"vla_confidence_threshold": 0.5, "max_xyz_deviation": 0.01}
        cfg = GateConfig.from_dict(data)
        assert cfg.vla_confidence_threshold == 0.5
        assert cfg.max_xyz_deviation == 0.01


class TestNeuroSymbolicSafetyGate:
    """Tests for NeuroSymbolicSafetyGate."""

    @pytest.fixture
    def gate(self):
        return NeuroSymbolicSafetyGate(GateConfig(
            vla_confidence_threshold=0.6,
            max_xyz_deviation=0.02,
            table_z=0.80,
        ))

    def test_approve_valid_action(self, gate):
        """A valid VLA action with high confidence should be approved."""
        vla = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        scripted = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert result.approved is True
        assert result.reason == "approved"

    def test_reject_low_confidence(self):
        """Low VLA confidence should trigger fallback when use_confidence=True."""
        gate = NeuroSymbolicSafetyGate(GateConfig(use_confidence=True))
        vla = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        scripted = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.3,  # below threshold
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert result.approved is False
        assert result.reason == "vla_low_confidence"
        assert np.allclose(result.action, scripted)

    def test_confidence_ignored_by_default(self):
        """Low VLA confidence should NOT trigger fallback when use_confidence=False (default)."""
        gate = NeuroSymbolicSafetyGate()  # use_confidence=False by default
        vla = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        scripted = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.3,  # low, but ignored
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert result.approved is True  # confidence check disabled

    def test_reject_large_deviation(self, gate):
        """Action too far from scripted should be rejected."""
        vla = _action(dx=0.05, dy=0.0, dz=0.0, gripper=-1.0)   # 5cm deviation
        scripted = _action(dx=0.0, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert result.approved is False
        assert "deviant" in result.reason

    def test_reject_workspace_violation(self, gate):
        """Action that would move EEF below table should be rejected."""
        vla = _action(dx=0.0, dy=0.0, dz=-0.15, gripper=-1.0)
        scripted = _action(dx=0.0, dy=0.0, dz=-0.15, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.85]),  # 0.85-0.15=0.70 < 0.75
            phase="reach_pick",
        )
        assert result.approved is False
        assert result.reason == "workspace_violation"

    def test_reject_gripper_open_during_close_phase(self, gate):
        """Opening gripper during grasp_close should be rejected."""
        vla = _action(dx=0.0, dy=0.0, dz=0.0, gripper=-1.0)   # OPEN
        scripted = _action(dx=0.0, dy=0.0, dz=0.0, gripper=1.0)  # CLOSE
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="grasp_close",
        )
        assert result.approved is False
        assert "gripper" in result.reason

    def test_reject_gripper_close_during_release_phase(self, gate):
        """Closing gripper during release_verify should be rejected."""
        vla = _action(dx=0.0, dy=0.0, dz=0.0, gripper=1.0)    # CLOSE
        scripted = _action(dx=0.0, dy=0.0, dz=0.0, gripper=-1.0)  # OPEN
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="release_verify",
        )
        assert result.approved is False

    def test_reject_nan_action(self, gate):
        """NaN in VLA action should always be rejected."""
        vla = _action(dx=float("nan"), dy=0.0, dz=0.0, gripper=-1.0)
        scripted = _action(dx=0.0, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert result.approved is False
        assert result.reason == "non_finite_action"

    def test_rotation_always_zeroed(self, gate):
        """Output action should have zero rotation regardless of input."""
        vla = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        vla[3:6] = [0.5, -0.3, 0.2]  # bogus rotation
        scripted = _action(dx=0.01, dy=0.0, dz=0.0, gripper=-1.0)
        result = gate.validate(
            vla_action=vla,
            scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="reach_pick",
        )
        assert np.all(result.action[3:6] == 0.0)

    def test_gate_with_phase_constraints_disabled(self):
        """Disabling phase constraints prevents phase-semantic checks.

        Note: deviation checks are separate from phase-semantic constraints.
        """
        gate = NeuroSymbolicSafetyGate(GateConfig(
            enable_phase_constraints=False,
            enable_deviation_check=False,
        ))
        vla = _action(dx=0.0, dy=0.0, dz=0.0, gripper=-1.0)  # opening
        scripted = _action(dx=0.0, dy=0.0, dz=0.0, gripper=1.0)  # closing
        result = gate.validate(
            vla_action=vla, scripted_action=scripted,
            vla_confidence=0.85,
            eef_pos=np.array([0.0, 0.0, 0.90]),
            phase="grasp_close",
        )
        # Phase constraint is disabled, so gripper opening during grasp_close is NOT rejected
        assert result.approved is True


class TestGateStatistics:
    """Tests for GateStatistics collector."""

    def test_empty_stats(self):
        stats = GateStatistics()
        assert stats.total_checks == 0
        assert stats.acceptance_ratio == 0.0

    def test_record_and_summary(self):
        stats = GateStatistics()
        for i in range(10):
            if i < 7:
                result = GateResult(approved=True, action=np.zeros(7),
                                    reason="approved", vla_confidence=0.9)
            else:
                result = GateResult(approved=False, action=np.zeros(7),
                                    reason="vla_low_confidence", vla_confidence=0.4)
            stats.record(result)

        assert stats.total_checks == 10
        assert stats.approved == 7
        assert stats.rejected == 3
        assert stats.acceptance_ratio == 0.7
        assert stats.rejection_reasons == {"vla_low_confidence": 3}

    def test_reset(self):
        stats = GateStatistics()
        stats.record(GateResult(approved=True, action=np.zeros(7),
                                reason="approved", vla_confidence=0.9))
        stats.reset()
        assert stats.total_checks == 0


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_to_dict(self):
        result = GateResult(
            approved=True, action=np.array([0.0]*7),
            reason="approved", vla_confidence=0.85,
            deviation_from_scripted=0.005,
        )
        d = result.to_dict()
        assert d["approved"] is True
        assert d["reason"] == "approved"
        assert d["vla_confidence"] == 0.85
        assert d["deviation_from_scripted"] == 0.005
