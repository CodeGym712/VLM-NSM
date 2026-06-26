"""RobosuiteScriptedController — proportional control for robosuite OSC_POSE.

Produces 7-dim action vectors normalized to [-1, 1] suitable for
robosuite 1.5.2 BASIC composite controller (OSC_POSE + gripper).

Normalized actions are internally scaled by robosuite to physical deltas
(output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]).

This controller uses proportional control:
  normalized_delta = kp * (target_pos - current_pos) / 0.05
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.embodied.controllers.base_controller import BaseController

# Max physical delta mapped by OSC_POSE per step
_OSC_POSE_MAX_DELTA = 0.05


class RobosuiteScriptedController(BaseController):
    """Proportional controller for robosuite normalized action space.

    Args:
        action_dim: Action dimension (7 for BASIC composite controller).
        kp: Proportional gain for position control.
        max_delta: Maximum per-step position delta in meters.
        grasp_threshold: Gripper close threshold (negative = close).
    """

    def __init__(
        self,
        action_dim: int = 7,
        kp: float = 0.05,
        max_delta: float = 0.02,
        grasp_threshold: float = -0.5,
    ):
        self._action_dim = action_dim
        self._kp = kp
        self._max_delta = max_delta
        self._grasp_threshold = grasp_threshold
        self._step_idx = 0

        # Target state set by the executor
        self._current_pos: np.ndarray | None = None
        self._target_pos: np.ndarray | None = None
        self._gripper_target: float = 0.0  # 0 = open, 1 = close

    def set_target(self, pos: dict | np.ndarray | None = None,
                   close_gripper: bool | None = None):
        """Set the target position and/or gripper state.

        Args:
            pos: Target position dict {"x", "y", "z"} or array [x, y, z].
            close_gripper: True to close, False to open, None to keep.
        """
        if pos is not None:
            if isinstance(pos, dict):
                self._target_pos = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
            else:
                self._target_pos = np.asarray(pos, dtype=np.float32)
        if close_gripper is not None:
            self._gripper_target = 1.0 if close_gripper else -1.0

    def set_current_pos(self, pos: dict | np.ndarray | None = None):
        """Set the current EEF position (for proportional control).

        Args:
            pos: Current position dict {"x", "y", "z"} or array [x, y, z].
        """
        if pos is not None:
            if isinstance(pos, dict):
                self._current_pos = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float32)
            else:
                self._current_pos = np.asarray(pos, dtype=np.float32)

    def act(self, step_name: str, observation: np.ndarray) -> np.ndarray:
        """Produce a 7-dim normalized action for the given skill step.

        Uses proportional control: delta = kp * (target - current), then
        normalizes to [-1, 1] by dividing by _OSC_POSE_MAX_DELTA.

        Args:
            step_name: Current skill step name.
            observation: Observation vector (unused for target-based control).

        Returns:
            7-dim action: [dx, dy, dz, 0, 0, 0, gripper] in [-1, 1].
        """
        action = self._proportional_action()

        if step_name == "reach_pick":
            self._gripper_target = -1.0
        elif step_name == "grasp":
            self._gripper_target = 1.0
        elif step_name == "lift":
            self._gripper_target = 1.0
        elif step_name == "reach_place":
            self._gripper_target = 1.0
        elif step_name == "release":
            self._gripper_target = -1.0
        elif step_name == "retreat":
            self._gripper_target = -1.0

        action[6] = self._gripper_target
        self._step_idx += 1
        return action

    def _proportional_action(self) -> np.ndarray:
        """Compute proportional control delta, normalized to [-1, 1]."""
        action = np.zeros(self._action_dim, dtype=np.float32)
        if self._target_pos is not None and self._current_pos is not None:
            # Error in meters
            error = self._target_pos - self._current_pos
            # Scaled delta
            delta = self._kp * error
            delta = np.clip(delta, -self._max_delta, self._max_delta)
            # Normalize to action space
            action[:3] = np.clip(delta / _OSC_POSE_MAX_DELTA, -1.0, 1.0)
        elif self._target_pos is not None:
            # Without current position, use unscaled target
            delta = self._target_pos * self._kp
            delta = np.clip(delta, -self._max_delta, self._max_delta)
            action[:3] = np.clip(delta / _OSC_POSE_MAX_DELTA, -1.0, 1.0)
        return action

    def reset(self):
        """Reset controller state."""
        self._step_idx = 0
        self._target_pos = None
        self._current_pos = None
        self._gripper_target = 0.0

    @property
    def controller_type(self) -> str:
        return "robosuite_scripted"

    @property
    def action_dim(self) -> int:
        return self._action_dim
