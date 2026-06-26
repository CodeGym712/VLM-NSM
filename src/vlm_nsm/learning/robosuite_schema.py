"""Lightweight robosuite observation/action schema helpers.

This module intentionally avoids heavy ML dependencies so runtime
collection code can import observation/action metadata without pulling
in ``torch``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


OBS_KEYS = [
    "eef_x", "eef_y", "eef_z",
    "gripper_open", "gripper_has_object",
    "cube1_x", "cube1_y", "cube1_z",
    "cube2_x", "cube2_y", "cube2_z",
]
"""Names of the 11-dim observation vector components."""

OBS_DIM = 11
"""Observation dimension (eef_xyz=3 + gripper_state=2 + 2 cubes x xyz=6)."""

ACTION_DIM = 7
"""Action dimension (dx, dy, dz, 0, 0, 0, gripper)."""


def build_robosuite_obs_vector(
    eef: dict[str, float],
    gripper: dict[str, Any],
    object_poses: dict[str, dict[str, float]],
    cubes: list[str] | None = None,
) -> np.ndarray:
    """Build a flat observation vector from robosuite env state."""
    if cubes is None:
        cubes = ["cube1", "cube2"]
    arr = [
        eef["x"], eef["y"], eef["z"],
        float(gripper.get("open", -1.0)),
        float(gripper.get("has_object", False)),
    ]
    for cube in cubes:
        pose = object_poses.get(cube, {"x": 0.0, "y": 0.0, "z": 0.0})
        arr += [pose["x"], pose["y"], pose["z"]]
    return np.array(arr, dtype=np.float32)
