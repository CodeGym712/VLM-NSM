"""Pure goal feature utilities for Stage 20 skill learning.

These helpers are intentionally independent of Robosuite. They operate on
small pose/state dictionaries so dataset builders and tests can use the same
goal-delta definitions as future learned skill executors.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


PHASES = {
    "reach_pick",
    "grasp",
    "lift",
    "reach_place",
    "place",
    "release",
    "retreat",
}


def _xyz(pose: Any) -> np.ndarray:
    if isinstance(pose, dict):
        if "xyz" in pose:
            arr = pose["xyz"]
        elif "position" in pose:
            arr = pose["position"]
        else:
            arr = [pose.get("x", 0.0), pose.get("y", 0.0), pose.get("z", 0.0)]
    else:
        arr = pose
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size < 3:
        raise ValueError(f"pose must contain at least xyz, got shape {out.shape}")
    return out[:3]


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("scalar value is empty")
    return float(arr[0])


def compute_relative_pose(a_pose: Any, b_pose: Any) -> np.ndarray:
    """Return xyz relative pose ``a - b``.

    Orientation fields may exist in dicts or longer arrays; they are ignored
    for the Stage 20 foundation features.
    """
    return _xyz(a_pose) - _xyz(b_pose)


def compute_placement_bin(top_pose: Any, bottom_pose: Any, center_threshold: float = 0.025) -> str:
    """Discretize top-vs-bottom XY displacement into center/cardinal/quadrant."""
    rel = compute_relative_pose(top_pose, bottom_pose)
    dx, dy = float(rel[0]), float(rel[1])
    parts: list[str] = []
    if abs(dx) <= center_threshold and abs(dy) <= center_threshold:
        return "center"
    if dy > center_threshold:
        parts.append("back")
    elif dy < -center_threshold:
        parts.append("front")
    if dx > center_threshold:
        parts.append("right")
    elif dx < -center_threshold:
        parts.append("left")
    return "-".join(parts) if parts else "center"


def compute_goal_delta(phase: str, current_state: dict[str, Any], target_spec: dict[str, Any]) -> np.ndarray:
    """Compute phase-specific desired state change from current state to target."""
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}")

    if phase in {"reach_pick", "retreat"}:
        return compute_relative_pose(target_spec["target_gripper_pose"], current_state["gripper_pose"])
    if phase == "grasp":
        return np.array([
            _scalar(target_spec["target_gripper_open"]) - _scalar(current_state["gripper_open"])
        ], dtype=float)
    if phase == "lift":
        return np.array([
            _scalar(target_spec["target_top_z"]) - float(_xyz(current_state["top_pose"])[2])
        ], dtype=float)
    if phase in {"reach_place", "place"}:
        return compute_relative_pose(target_spec["target_top_pose"], current_state["top_pose"])
    if phase == "release":
        values = [
            _scalar(target_spec["target_gripper_open"]) - _scalar(current_state["gripper_open"])
        ]
        if "object_stability" in current_state or "target_object_stability" in target_spec:
            values.append(
                _scalar(target_spec.get("target_object_stability", 1.0))
                - _scalar(current_state.get("object_stability", 0.0))
            )
        return np.asarray(values, dtype=float)
    raise ValueError(f"Unsupported phase: {phase}")


def compute_reached_goal_delta(
    phase_start_state: dict[str, Any],
    current_state: dict[str, Any],
    fields: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return state deltas already reached since the phase started."""
    if fields is None:
        fields = sorted(set(phase_start_state) & set(current_state))
    reached: dict[str, np.ndarray] = {}
    for field in fields:
        if field not in phase_start_state or field not in current_state:
            continue
        try:
            if field.endswith("_pose") or field in {"top_pose", "bottom_pose", "gripper_pose"}:
                reached[field] = _xyz(current_state[field]) - _xyz(phase_start_state[field])
            else:
                reached[field] = np.asarray(current_state[field], dtype=float) - np.asarray(
                    phase_start_state[field], dtype=float
                )
        except (TypeError, ValueError):
            continue
    return reached


def compute_goal_error(goal_delta: Any) -> float:
    """Return finite L2 norm of a goal delta or nested reached-goal dict."""
    if isinstance(goal_delta, dict):
        parts = [np.asarray(v, dtype=float).reshape(-1) for v in goal_delta.values()]
        if not parts:
            return 0.0
        arr = np.concatenate(parts)
    else:
        arr = np.asarray(goal_delta, dtype=float).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    return float(np.linalg.norm(arr))


def compute_progress_scalar(
    initial_error: float,
    current_error: float,
    clip: bool = True,
    eps: float = 1e-8,
) -> float:
    """Return progress from 0 to 1 as error decreases."""
    denom = max(abs(float(initial_error)), eps)
    progress = 1.0 - float(current_error) / denom
    if clip:
        progress = min(1.0, max(0.0, progress))
    return float(progress)


def compute_stalled_steps(error_history: Iterable[float], tolerance: float = 1e-3) -> int:
    """Count the low-change suffix sample length in an error history."""
    values = [float(v) for v in error_history]
    if len(values) < 2:
        return 0
    count = 1
    for prev, cur in zip(reversed(values[:-1]), reversed(values[1:])):
        if abs(cur - prev) <= tolerance:
            count += 1
        else:
            break
    return count if count > 1 else 0
