"""Tests for OperationalSpacePID controller."""

import numpy as np
import pytest

from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID,
    PIDGain,
    GainSchedule,
    _OSC_POSE_MAX_DELTA,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pid() -> OperationalSpacePID:
    c = OperationalSpacePID()
    c.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    return c


# ── Initialization ───────────────────────────────────────────────────────────

def test_default_initialization():
    pid = OperationalSpacePID()
    assert pid.controller_type == "operational_space_pid"
    assert pid.action_dim == 7
    assert pid._current_pos is None
    assert pid._target_pos is None


def test_custom_gain_schedule():
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.1, ki=0.01, kd=0.02, max_delta=0.03, i_max=0.05)
    pid = OperationalSpacePID(gain_schedule=gs)
    g = pid._gain_schedule.get("reach_pick")
    assert g.kp == 0.1
    assert g.ki == 0.01
    assert g.kd == 0.02


def test_custom_workspace():
    ws = {"x_min": -0.5, "x_max": 0.5, "y_min": -0.3, "y_max": 0.3, "z_min": 0.5, "z_max": 1.5}
    pid = OperationalSpacePID(workspace_bounds=ws)
    assert pid._workspace["x_min"] == -0.5


# ── Target Setting ───────────────────────────────────────────────────────────

def test_set_target_dict(pid):
    pid.set_target(pos={"x": 0.1, "y": 0.2, "z": 1.0})
    np.testing.assert_array_almost_equal(pid._target_pos, [0.1, 0.2, 1.0])


def test_set_target_array(pid):
    pid.set_target(pos=np.array([0.1, 0.2, 1.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(pid._target_pos, [0.1, 0.2, 1.0])


def test_set_target_resets_integral(pid):
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    pid._integral_error = np.array([0.01, 0.02, 0.03])
    # Same target → integral preserved
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    np.testing.assert_array_almost_equal(pid._integral_error, [0.01, 0.02, 0.03])
    # Different target → integral reset
    pid.set_target(pos={"x": 0.2, "y": 0.0, "z": 0.9})
    np.testing.assert_array_almost_equal(pid._integral_error, [0.0, 0.0, 0.0])


def test_set_target_vel(pid):
    pid.set_target(vel=np.array([0.01, -0.01, 0.0], dtype=np.float32))
    np.testing.assert_array_almost_equal(pid._target_vel, [0.01, -0.01, 0.0])


def test_set_gripper(pid):
    pid.set_target(close_gripper=True)
    assert pid._gripper_target == 1.0
    pid.set_target(close_gripper=False)
    assert pid._gripper_target == -1.0


# ── Current State Setting ────────────────────────────────────────────────────

def test_set_current_dict(pid):
    pid.set_current({"x": 0.05, "y": 0.1, "z": 0.95})
    np.testing.assert_array_almost_equal(pid._current_pos, [0.05, 0.1, 0.95])


def test_set_current_array(pid):
    pid.set_current(np.array([0.05, 0.1, 0.95], dtype=np.float32))
    np.testing.assert_array_almost_equal(pid._current_pos, [0.05, 0.1, 0.95])


def test_set_current_velocity_finite_diff(pid):
    """Velocity estimated from position difference."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_current({"x": 0.01, "y": 0.0, "z": 0.9})
    # dt=0.02, dx=0.01 → vx ≈ 0.5 m/s
    assert pid._current_vel is not None
    assert abs(pid._current_vel[0] - 0.5) < 0.1


def test_set_current_explicit_velocity(pid):
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9},
                    vel=np.array([0.1, -0.2, 0.05], dtype=np.float32))
    np.testing.assert_array_almost_equal(pid._current_vel, [0.1, -0.2, 0.05])


# ── Phase Switching ──────────────────────────────────────────────────────────

def test_set_phase_updates_gains(pid):
    pid.set_phase("grasp")
    assert pid._active_gain.kp == 0.07
    assert pid._active_gain.max_delta == 0.01

    pid.set_phase("reach_pick")
    assert pid._active_gain.kp == 0.10
    assert pid._active_gain.max_delta == 0.02


def test_set_unknown_phase_uses_default(pid):
    pid.set_phase("nonexistent_phase")
    assert pid._active_gain.kp == 0.05  # default PIDGain


# ── PID Computation ──────────────────────────────────────────────────────────

def test_pid_zero_error(pid):
    """When current == target, output should be zero."""
    pid.set_current({"x": 0.1, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    v_des = pid.compute_pid_velocity()
    np.testing.assert_array_almost_equal(v_des, [0.0, 0.0, 0.0])


def test_pid_pure_proportional(pid):
    """Without I and D, output = kp * error, clipped."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.10, ki=0.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")

    v_des = pid.compute_pid_velocity()
    # error_x = 0.1, kp=0.10 → vx = 0.01, clipped to 0.05
    assert abs(v_des[0] - 0.01) < 1e-6
    assert v_des[1] == 0.0
    assert v_des[2] == 0.0


def test_pid_proportional_clipped(pid):
    """Large error should be clipped to max_delta."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=1.0, ki=0.0, kd=0.0, max_delta=0.02, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 1.0, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")

    v_des = pid.compute_pid_velocity()
    # error_x = 1.0, kp=1.0 → vx = 1.0, clipped to 0.02
    assert abs(v_des[0]) == 0.02


def test_pid_integral_accumulation(pid):
    """Integral term accumulates error over time."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=1.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs, dt=0.02)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")

    # Step 1: error=0.1, integral += 0.1*0.02 = 0.002, output = 0.002
    v1 = pid.compute_pid_velocity()
    assert abs(v1[0] - 0.002) < 1e-4

    # Simulate movement toward target
    pid.set_current({"x": 0.002, "y": 0.0, "z": 0.9})

    # Step 2: error=0.098, integral += 0.098*0.02 = 0.00196
    v2 = pid.compute_pid_velocity()
    assert v2[0] > v1[0]  # integral grows


def test_pid_integral_anti_windup(pid):
    """Integral should not exceed i_max."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=10.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs, dt=0.02)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 1.0, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")

    # Many steps with large error
    for _ in range(100):
        v = pid.compute_pid_velocity()
        assert abs(v[0]) <= 0.05  # max_delta
        assert np.all(np.abs(pid._integral_error) <= 0.03)  # anti-windup


def test_pid_derivative_damping(pid):
    """D term should oppose motion."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=0.0, kd=0.5, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")

    # Moving right (positive vx) → D term negative (opposes motion)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9},
                    vel=np.array([0.05, 0.0, 0.0], dtype=np.float32))
    v_des = pid.compute_pid_velocity()
    assert v_des[0] < 0  # D term opposes positive velocity


def test_pid_feedforward(pid):
    """Feedforward velocity should pass through directly."""
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=0.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.0, "y": 0.0, "z": 0.9},
                   vel=np.array([0.01, -0.005, 0.0], dtype=np.float32))
    pid.set_phase("reach_pick")

    v_des = pid.compute_pid_velocity()
    np.testing.assert_array_almost_equal(v_des, [0.01, -0.005, 0.0])


def test_pid_no_target_returns_zero(pid):
    """Without target set, output should be zero."""
    pid._target_pos = None
    v_des = pid.compute_pid_velocity()
    np.testing.assert_array_almost_equal(v_des, [0.0, 0.0, 0.0])


def test_pid_no_current_returns_zero():
    """Without current position set, output should be zero."""
    pid = OperationalSpacePID()
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    v_des = pid.compute_pid_velocity()
    np.testing.assert_array_almost_equal(v_des, [0.0, 0.0, 0.0])


# ── Workspace Clipping ──────────────────────────────────────────────────────

def test_workspace_clip_z_low(pid):
    """Should not allow moving below z_min."""
    ws = {"x_min": -0.5, "x_max": 0.5, "y_min": -0.5, "y_max": 0.5,
          "z_min": 0.8, "z_max": 1.2}
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=0.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs, workspace_bounds=ws)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.79})  # at boundary
    # Target far below
    pid.set_target(pos={"x": 0.0, "y": 0.0, "z": 0.7},
                   vel=np.array([0.0, 0.0, -0.05], dtype=np.float32))
    pid.set_phase("reach_pick")

    v_des = pid.compute_pid_velocity()
    # z velocity should be >= 0 (can't go further below z_min)
    assert v_des[2] >= 0.0


def test_workspace_clip_z_high(pid):
    """Should not allow moving above z_max."""
    ws = {"x_min": -0.5, "x_max": 0.5, "y_min": -0.5, "y_max": 0.5,
          "z_min": 0.8, "z_max": 1.2}
    gs = GainSchedule()
    gs.gains["reach_pick"] = PIDGain(kp=0.0, ki=0.0, kd=0.0, max_delta=0.05, i_max=0.03)
    pid = OperationalSpacePID(gain_schedule=gs, workspace_bounds=ws)
    pid.set_current({"x": 0.0, "y": 0.0, "z": 1.21})  # at boundary
    pid.set_target(pos={"x": 0.0, "y": 0.0, "z": 1.3},
                   vel=np.array([0.0, 0.0, 0.05], dtype=np.float32))
    pid.set_phase("reach_pick")

    v_des = pid.compute_pid_velocity()
    assert v_des[2] <= 0.0  # can't go further above z_max


# ── NaN Protection ──────────────────────────────────────────────────────────

def test_pid_nan_protection(pid):
    """NaN in internal state → zero output."""
    pid.set_current({"x": float('nan'), "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    v_des = pid.compute_pid_velocity()
    # Should produce finite output (or safe zero)
    assert np.all(np.isfinite(v_des))


def test_pid_inf_protection(pid):
    """Inf in internal state → zero output."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": float('inf'), "y": 0.0, "z": 0.9})
    v_des = pid.compute_pid_velocity()
    assert np.all(np.isfinite(v_des))


# ── Action Output ────────────────────────────────────────────────────────────

def test_act_output_shape(pid):
    """act() should return 7-dim action."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.02, "y": 0.0, "z": 0.9})
    action = pid.act("reach_pick", np.zeros(10))
    assert len(action) == 7
    assert action.dtype == np.float32


def test_act_values_in_range(pid):
    """Normalized actions should be in [-1, 1]."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")
    for _ in range(50):
        action = pid.act("reach_pick", np.zeros(10))
        assert np.all(action[:3] >= -1.0)
        assert np.all(action[:3] <= 1.0)
        assert action[6] in (-1.0, 1.0)
        # Simulate movement
        if pid._current_pos is not None:
            pid.set_current(pid._current_pos + action[:3] * _OSC_POSE_MAX_DELTA)


def test_act_gripper_per_phase(pid):
    """Gripper should follow per-phase logic."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.0, "y": 0.0, "z": 0.9})

    a = pid.act("reach_pick", np.zeros(10))
    assert a[6] == -1.0  # open

    a = pid.act("grasp", np.zeros(10))
    assert a[6] == 1.0   # close

    a = pid.act("release", np.zeros(10))
    assert a[6] == -1.0  # open


# ── Reset ────────────────────────────────────────────────────────────────────

def test_reset_clears_state(pid):
    pid.set_current({"x": 0.1, "y": 0.2, "z": 0.95})
    pid.set_target(pos={"x": 0.2, "y": 0.3, "z": 1.0})
    pid._integral_error = np.array([0.01, 0.02, 0.03])
    pid._step_idx = 42

    pid.reset()

    assert pid._current_pos is None
    assert pid._target_pos is None
    assert pid._step_idx == 0
    np.testing.assert_array_almost_equal(pid._integral_error, [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(pid._target_vel, [0.0, 0.0, 0.0])


# ── Diagnostics ──────────────────────────────────────────────────────────────

def test_diagnostics_after_compute(pid):
    """diagnostics should reflect last PID computation."""
    pid.set_current({"x": 0.0, "y": 0.0, "z": 0.9})
    pid.set_target(pos={"x": 0.1, "y": 0.0, "z": 0.9})
    pid.set_phase("reach_pick")
    pid.compute_pid_velocity()

    diag = pid.diagnostics
    assert "p_term" in diag
    assert "i_term" in diag
    assert "d_term" in diag
    assert "v_des" in diag
    assert "active_gain" in diag
    assert diag["active_gain"]["kp"] == 0.10


# ── GainSchedule ─────────────────────────────────────────────────────────────

def test_gain_schedule_defaults():
    gs = GainSchedule()
    g = gs.get("reach_pick")
    assert g.kp == 0.10
    assert g.ki == 0.005

    g = gs.get("grasp")
    assert g.max_delta == 0.01
    assert g.kd == 0.0


def test_gain_schedule_unknown_phase():
    gs = GainSchedule()
    g = gs.get("unknown")
    assert g.kp == 0.05  # default PIDGain
    assert g.max_delta == 0.02
