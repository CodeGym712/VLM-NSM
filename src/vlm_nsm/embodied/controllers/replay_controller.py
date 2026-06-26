"""ReplayController — replays recorded demonstrations.

ReplayController success rate serves as data quality sanity check before
training Behavior Cloning models. If replay_success_rate is low,
do not proceed to BC training.
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.embodied.controllers.base_controller import BaseController


class ReplayController(BaseController):
    """Replays recorded trajectories from a DemonstrationRecorder npz file.

    The replay success rate measures data quality. If demonstrations are
    inconsistent or corrupted, replay will fail — indicating the data
    should not be used for BC training.
    """

    def __init__(self, actions: np.ndarray | None = None):
        self._actions = actions
        self._idx = 0

    def load(self, actions: np.ndarray):
        """Load action sequence to replay.

        Args:
            actions: Action array of shape (T, action_dim).
        """
        self._actions = np.asarray(actions, dtype=np.float32)
        self._idx = 0

    def act(self, step_name: str, observation: np.ndarray) -> np.ndarray:
        """Return the next action from the recorded sequence.

        Args:
            step_name: Ignored in replay mode (the recorded action is used).
            observation: Ignored in replay mode.

        Returns:
            The next recorded action vector.

        Raises:
            RuntimeError: If no actions are loaded or sequence is exhausted.
        """
        if self._actions is None:
            raise RuntimeError("No actions loaded. Call load() first.")
        if self._idx >= len(self._actions):
            raise RuntimeError("Replay sequence exhausted.")
        action = self._actions[self._idx]
        self._idx += 1
        return action

    def reset(self):
        self._idx = 0

    @property
    def controller_type(self) -> str:
        return "replay"

    @property
    def remaining(self) -> int:
        """Number of remaining steps in the loaded sequence."""
        if self._actions is None:
            return 0
        return max(0, len(self._actions) - self._idx)

    @property
    def replay_success(self) -> bool:
        """Whether the replay completed without exhaustion."""
        return self._actions is not None and self._idx >= len(self._actions)
