"""ScriptedController — hardcoded pose sequences for each skill step.

No learning involved. Used as a sanity check for the embodied execution pipeline.
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.embodied.controllers.base_controller import BaseController


class ScriptedController(BaseController):
    """Controller with hardcoded action sequences per skill step.

    Produces delta-position actions for a robot end-effector:
    [dx, dy, dz, dgripper] where dgripper is binary open/close.
    """

    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim
        self._step = 0

    def act(self, step_name: str, observation: np.ndarray) -> np.ndarray:
        """Return a hardcoded action for the given skill step.

        Args:
            step_name: Skill step name.
            observation: Observation vector (unused by scripted controller).

        Returns:
            Action vector of shape (action_dim,).
        """
        action_map = {
            "reach_pick": np.array([0.0, 0.0, -0.05, 0.0], dtype=np.float32),
            "grasp": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "lift": np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32),
            "reach_place": np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            "release": np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            "retreat": np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float32),
        }
        action = action_map.get(step_name, np.zeros(self.action_dim, dtype=np.float32))

        # Ensure correct dimension
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)), mode="constant")
        elif len(action) > self.action_dim:
            action = action[: self.action_dim]

        self._step += 1
        return action

    def reset(self):
        self._step = 0

    @property
    def controller_type(self) -> str:
        return "scripted"
