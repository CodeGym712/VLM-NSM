"""SymbolConditionedObservationFilter.

Uses symbolic action context available from the planner to select task-relevant
observations. Does NOT use privileged ground-truth labels unavailable at test time.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SymbolConditionedObservationFilter:
    """Filters raw environment observations using symbolic action context.

    The filter uses only information available at test time:
    - The symbolic action produced by the planner (action name + args).
    - The raw observation from the environment.

    It does NOT use privileged ground-truth labels such as object positions
    from a simulator state that would be unavailable in a real deployment.
    """

    def __init__(self, obs_dim: int = 16):
        self.obs_dim = obs_dim

    def filter(
        self,
        action_name: str,
        action_args: list[str],
        raw_observation: dict[str, Any],
    ) -> np.ndarray:
        """Produce a filtered observation vector from raw observation + action context.

        Args:
            action_name: Symbolic action name (e.g. "move", "pick", "place").
            action_args: Symbolic action arguments (e.g. ["cube1", "table", "cube2"]).
            raw_observation: Full environment observation dict. Expected keys include
                object pose entries like ``{name}_pos``, ``{name}_quat``.

        Returns:
            Filtered observation array of shape (obs_dim,).
        """
        # Extract target object and location from action args
        obj = action_args[0] if len(action_args) > 0 else ""
        source = action_args[1] if len(action_args) > 1 else ""
        target = action_args[2] if len(action_args) > 2 else ""

        features: list[float] = []

        # Action context (one-hot style encoding)
        action_names = ["move", "pick", "place", "reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
        for name in action_names:
            features.append(1.0 if action_name == name else 0.0)

        # Object-relevant pose features
        for key in (f"{obj}_pos", f"{obj}_quat", f"{target}_pos", f"{target}_quat"):
            val = raw_observation.get(key, [0.0] * 3 if "pos" in key else [0.0] * 4)
            if isinstance(val, (list, tuple, np.ndarray)):
                features.extend(float(v) for v in val)
            else:
                features.append(float(val))

        # Pad or truncate to obs_dim
        arr = np.array(features, dtype=np.float32)
        if len(arr) < self.obs_dim:
            arr = np.pad(arr, (0, self.obs_dim - len(arr)), mode="constant")
        elif len(arr) > self.obs_dim:
            arr = arr[: self.obs_dim]

        return arr
