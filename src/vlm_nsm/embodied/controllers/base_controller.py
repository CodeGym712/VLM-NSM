"""BaseController — abstract interface for all controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseController(ABC):
    """Abstract base class for all controllers.

    A controller receives a skill step name and produces an action vector
    to be executed in the environment.
    """

    @abstractmethod
    def act(self, step_name: str, observation: np.ndarray) -> np.ndarray:
        """Produce an action given the current step and observation.

        Args:
            step_name: Current skill step name (e.g. "reach_pick", "grasp").
            observation: Current observation vector.

        Returns:
            Action vector of shape (action_dim,).
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset the controller state for a new episode."""
        ...

    @property
    @abstractmethod
    def controller_type(self) -> str:
        """Return a string identifier for the controller type."""
        ...
