"""Operational Space PID Controller with velocity feedforward.

Replaces the pure P-controller in robosuite OSC_POSE action space with
a full PID + feedforward controller. This is Layer 1 of the hierarchical
inverse dynamics architecture.

Formulation:
    error     = target_pos - current_pos
    p_term    = kp * error
    i_term    = ki * clip(integral_error, -i_max, i_max)
    d_term    = kd * (-current_vel + target_vel)
    ff_term   = target_vel (from Learned ID or zero)
    v_des     = p_term + i_term + d_term + ff_term
    v_des     = clip(v_des, -max_delta, max_delta)
    action[:3] = v_des / OSC_POSE_MAX_DELTA  (normalize to [-1, 1])

Key features:
    - Anti-windup: integral clamped to [-i_max, i_max], reset on target change
    - Workspace bounds: predicted next EEF checked against workspace limits
    - NaN protection: all intermediate values checked
    - Phase-specific gain scheduling via ``GainSchedule``
    - Velocity estimation: from env or finite-difference fallback
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vlm_nsm.embodied.controllers.base_controller import BaseController

# Robosuite OSC_POSE max position delta per step (meters)
_OSC_POSE_MAX_DELTA = 0.05

# Default workspace bounds (robosuite Stack env)
_DEFAULT_WORKSPACE = {
    "x_min": -0.25, "x_max": 0.15,
    "y_min": -0.25, "y_max": 0.15,
    "z_min": 0.78,  "z_max": 1.15,
}


# ── Gain Schedule ────────────────────────────────────────────────────────────

@dataclass
class PIDGain:
    """PID gain triplet for a single phase."""
    kp: float = 0.05      # proportional gain
    ki: float = 0.0       # integral gain
    kd: float = 0.0       # derivative gain
    max_delta: float = 0.02  # per-step delta clamp (meters)
    i_max: float = 0.03   # integral anti-windup clamp (meters)


@dataclass
class GainSchedule:
    """Per-phase PID gain schedule.

    Defaults calibrated for 2-block stacking with Panda arm in robosuite.
    """

    gains: dict[str, PIDGain] = field(default_factory=lambda: {
        "reach_pick":   PIDGain(kp=0.10, ki=0.005, kd=0.0,  max_delta=0.02,  i_max=0.03),
        "grasp":        PIDGain(kp=0.07, ki=0.002, kd=0.0,  max_delta=0.01,  i_max=0.02),
        "lift":         PIDGain(kp=0.10, ki=0.002, kd=0.0,  max_delta=0.02,  i_max=0.03),
        "reach_place":  PIDGain(kp=0.08, ki=0.005, kd=0.0,  max_delta=0.015, i_max=0.03),
        "release":      PIDGain(kp=0.04, ki=0.001, kd=0.0,  max_delta=0.005, i_max=0.01),
        "retreat":      PIDGain(kp=0.08, ki=0.001, kd=0.0,  max_delta=0.02,  i_max=0.03),
        # SkillLibrary aliases (same gains, different names)
        "reach":        PIDGain(kp=0.10, ki=0.005, kd=0.0,  max_delta=0.02,  i_max=0.03),
        "release_place": PIDGain(kp=0.04, ki=0.001, kd=0.0,  max_delta=0.005, i_max=0.01),
        "pull":         PIDGain(kp=0.10, ki=0.003, kd=0.0,  max_delta=0.03,  i_max=0.05),
    })

    def get(self, phase_name: str) -> PIDGain:
        return self.gains.get(phase_name, PIDGain())


# ── PID Controller ───────────────────────────────────────────────────────────

class OperationalSpacePID(BaseController):
    """Layer 1: PID + velocity feedforward in operational space.

    Produces 7-dim normalized actions for robosuite OSC_POSE.

    Args:
        action_dim: Action dimension (7 for BASIC composite controller).
        gain_schedule: Per-phase PID gains.
        workspace_bounds: Dict with x_min/x_max/y_min/y_max/z_min/z_max.
        grasp_threshold: Gripper value above which grasp is considered closed.
        dt: Control timestep in seconds (for integral accumulation).
    """

    def __init__(
        self,
        action_dim: int = 7,
        gain_schedule: GainSchedule | None = None,
        workspace_bounds: dict | None = None,
        grasp_threshold: float = -0.5,
        dt: float = 0.02,
    ):
        self._action_dim = action_dim
        self._gain_schedule = gain_schedule or GainSchedule()
        self._workspace = workspace_bounds or _DEFAULT_WORKSPACE
        self._grasp_threshold = grasp_threshold
        self._dt = dt

        # Internal state
        self._current_pos: np.ndarray | None = None
        self._current_vel: np.ndarray | None = None
        self._prev_pos: np.ndarray | None = None  # for finite-diff vel
        self._target_pos: np.ndarray | None = None
        self._target_vel: np.ndarray = np.zeros(3, dtype=np.float32)
        self._integral_error: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_error: np.ndarray = np.zeros(3, dtype=np.float32)
        self._step_idx: int = 0
        self._gripper_target: float = -1.0  # default open
        self._active_gain: PIDGain = PIDGain()

        # Diagnostics
        self._last_p_term: np.ndarray = np.zeros(3)
        self._last_i_term: np.ndarray = np.zeros(3)
        self._last_d_term: np.ndarray = np.zeros(3)
        self._last_output: np.ndarray = np.zeros(3)

    # ── Setters ──────────────────────────────────────────────────────────

    def set_target(self, pos: dict | np.ndarray | None = None,
                   vel: np.ndarray | None = None,
                   close_gripper: bool | None = None):
        """Set target position, velocity feedforward, and gripper state.

        Resets integral error on position target change.

        Args:
            pos: Target position dict {"x","y","z"} or array [x,y,z].
            vel: Target velocity (from Learned ID or zero).
            close_gripper: True=close, False=open, None=keep.
        """
        if pos is not None:
            new_target = np.asarray(
                [pos["x"], pos["y"], pos["z"]] if isinstance(pos, dict)
                else pos, dtype=np.float32
            )
            if self._target_pos is None or not np.allclose(new_target, self._target_pos):
                # Target changed → reset integral to prevent windup
                self._integral_error = np.zeros(3, dtype=np.float32)
            self._target_pos = new_target

        if vel is not None:
            self._target_vel = np.asarray(vel, dtype=np.float32)

        if close_gripper is not None:
            self._gripper_target = 1.0 if close_gripper else -1.0

    def set_current(self, pos: dict | np.ndarray | None = None,
                    vel: np.ndarray | None = None):
        """Set current EEF position and optionally velocity.

        Args:
            pos: Current position dict {"x","y","z"} or array [x,y,z].
            vel: Current EEF velocity. If None, estimated via finite difference.
        """
        if pos is not None:
            new_pos = np.asarray(
                [pos["x"], pos["y"], pos["z"]] if isinstance(pos, dict)
                else pos, dtype=np.float32
            )
            if self._current_pos is not None and vel is None:
                # Finite-difference velocity estimate
                self._current_vel = (new_pos - self._current_pos) / max(self._dt, 1e-4)
            self._prev_pos = self._current_pos
            self._current_pos = new_pos

        if vel is not None:
            self._current_vel = np.asarray(vel, dtype=np.float32)

    def set_phase(self, phase_name: str):
        """Switch to phase-specific PID gains. Resets integral to prevent cross-phase windup."""
        self._active_gain = self._gain_schedule.get(phase_name)
        self._integral_error = np.zeros(3, dtype=np.float32)

    # ── Core PID computation ─────────────────────────────────────────────

    def compute_pid_velocity(self) -> np.ndarray:
        """Compute PID velocity command.

        Returns:
            3-dim velocity [vx, vy, vz] in meters/step.
            Returns zeros if no target or current position set.
        """
        if self._target_pos is None or self._current_pos is None:
            return np.zeros(3, dtype=np.float32)

        g = self._active_gain
        error = self._target_pos - self._current_pos

        # NaN guard: if input is invalid, reset and return zero
        if not np.all(np.isfinite(error)):
            self._integral_error = np.zeros(3, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        # P term: proportional
        p_term = g.kp * error

        # I term: integral with anti-windup
        if g.ki > 0:
            self._integral_error += error * self._dt
            self._integral_error = np.clip(self._integral_error, -g.i_max, g.i_max)
            i_term = g.ki * self._integral_error
        else:
            i_term = np.zeros(3, dtype=np.float32)

        # D term: damp velocity
        if g.kd > 0:
            actual_vel = self._current_vel if self._current_vel is not None else np.zeros(3)
            # Damp toward target velocity, penalize excess actual velocity
            d_term = g.kd * (self._target_vel - actual_vel)
        else:
            d_term = np.zeros(3, dtype=np.float32)

        # FF term: velocity feedforward (from Learned ID or explicit target)
        ff_term = self._target_vel.copy()

        # Composite
        v_des = p_term + i_term + d_term + ff_term
        v_des = np.clip(v_des, -g.max_delta, g.max_delta)

        # Workspace check: predicted next position (v_des is per-step delta, no dt needed)
        predicted_pos = self._current_pos + v_des
        v_des = self._clip_to_workspace(v_des, predicted_pos)

        # NaN protection
        if not np.all(np.isfinite(v_des)):
            v_des = np.zeros(3, dtype=np.float32)

        # Store diagnostics
        self._last_p_term = p_term
        self._last_i_term = i_term
        self._last_d_term = d_term
        self._last_output = v_des
        self._prev_error = error.copy()

        return v_des

    def _clip_to_workspace(self, v_des: np.ndarray,
                           predicted_pos: np.ndarray) -> np.ndarray:
        """Clip velocity to keep predicted position within workspace."""
        w = self._workspace
        clipped = v_des.copy()
        for i, (axis, c_min, c_max) in enumerate([
            ("x", w["x_min"], w["x_max"]),
            ("y", w["y_min"], w["y_max"]),
            ("z", w["z_min"], w["z_max"]),
        ]):
            if predicted_pos[i] < c_min:
                clipped[i] = max(clipped[i], 0.0)  # don't go further negative
            elif predicted_pos[i] > c_max:
                clipped[i] = min(clipped[i], 0.0)  # don't go further positive
        return clipped

    # ── BaseController interface ─────────────────────────────────────────

    def act(self, step_name: str, observation: np.ndarray) -> np.ndarray:
        """Produce 7-dim normalized action for robosuite OSC_POSE.

        Args:
            step_name: Current skill step name (e.g. "reach_pick").
            observation: Observation vector (unused; target-based control).

        Returns:
            7-dim action: [dx, dy, dz, 0, 0, 0, gripper] in [-1, 1].
        """
        # Update gain schedule
        self.set_phase(step_name)

        # Compute velocity command
        v_des = self.compute_pid_velocity()

        # Per-phase gripper logic
        gripper_map = {
            "reach_pick": -1.0, "grasp": 1.0, "lift": 1.0,
            "reach_place": 1.0, "release": -1.0, "retreat": -1.0,
        }
        self._gripper_target = gripper_map.get(step_name, self._gripper_target)

        # Build normalized action
        action = np.zeros(self._action_dim, dtype=np.float32)
        action[:3] = np.clip(v_des / _OSC_POSE_MAX_DELTA, -1.0, 1.0)
        action[6] = self._gripper_target
        self._step_idx += 1
        return action

    def reset(self):
        """Reset all internal state for a new episode."""
        self._current_pos = None
        self._current_vel = None
        self._prev_pos = None
        self._target_pos = None
        self._target_vel = np.zeros(3, dtype=np.float32)
        self._integral_error = np.zeros(3, dtype=np.float32)
        self._prev_error = np.zeros(3, dtype=np.float32)
        self._step_idx = 0
        self._gripper_target = -1.0
        self._active_gain = PIDGain()
        self._last_p_term = np.zeros(3)
        self._last_i_term = np.zeros(3)
        self._last_d_term = np.zeros(3)
        self._last_output = np.zeros(3)

    @property
    def controller_type(self) -> str:
        return "operational_space_pid"

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def diagnostics(self) -> dict:
        """Return per-step PID diagnostics for logging."""
        return {
            "p_term": self._last_p_term.tolist(),
            "i_term": self._last_i_term.tolist(),
            "d_term": self._last_d_term.tolist(),
            "v_des": self._last_output.tolist(),
            "integral_error": self._integral_error.tolist(),
            "current_pos": self._current_pos.tolist() if self._current_pos is not None else None,
            "target_pos": self._target_pos.tolist() if self._target_pos is not None else None,
            "active_gain": {
                "kp": self._active_gain.kp,
                "ki": self._active_gain.ki,
                "kd": self._active_gain.kd,
                "max_delta": self._active_gain.max_delta,
            },
        }
