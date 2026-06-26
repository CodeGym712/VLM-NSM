"""Build Stage 20 role-conditioned skill learning windows.

This builder converts existing Stage 19 per-step trajectories into fixed-size
windows for goal-oriented role-conditioned low-level policy learning. It does
not run Robosuite, Qwen, DashScope, or any training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.data.stage20_goal_features import (
    compute_goal_delta,
    compute_goal_error,
    compute_placement_bin,
    compute_progress_scalar,
    compute_reached_goal_delta,
    compute_relative_pose,
)


PHASE_ID_TO_NAME = {
    0: "reach_pick",
    1: "grasp",
    2: "lift",
    3: "reach_place",
    4: "release",
    5: "retreat",
}

PHASE_TO_ID = {v: k for k, v in PHASE_ID_TO_NAME.items()}
DEFAULT_DIRECTION_TO_ID = {"cube1_to_cube2": 0, "cube2_to_cube1": 1, "unknown": 2}
DEFAULT_PLACEMENT_BIN_TO_ID = {
    "center": 0,
    "front": 1,
    "back": 2,
    "left": 3,
    "right": 4,
    "front-left": 5,
    "front-right": 6,
    "back-left": 7,
    "back-right": 8,
    "unknown": 9,
}
DEFAULT_TEACHER_PROFILE_TO_ID = {"cube2_robust": 0, "default": 1, "unknown": 2}

ROLE_STATE_DIM = 20
GOAL_DELTA_DIM = 16
REACHED_GOAL_DELTA_DIM = 20

ROLE_STATE_LAYOUT = [
    "gripper_xyz[0:3]",
    "gripper_open[3]",
    "top_xyz[4:7]",
    "bottom_xyz[7:10]",
    "relative_gripper_to_top_xyz[10:13]",
    "relative_top_to_bottom_xyz[13:16]",
    "top_size_xyz[16:19]",
    "bottom_size_scalar[19]",
]

GOAL_DELTA_LAYOUT = [
    "gripper_delta_xyz[0:3]",
    "top_delta_xyz[3:6]",
    "gripper_open_delta[6]",
    "target_top_z_delta[7]",
    "relative_top_bottom_delta_xyz[8:11]",
    "reserved[11:16]",
]

REACHED_GOAL_DELTA_LAYOUT = [
    "gripper_xyz_delta[0:3]",
    "gripper_open_delta[3]",
    "top_xyz_delta[4:7]",
    "bottom_xyz_delta[7:10]",
    "relative_gripper_to_top_delta_xyz[10:13]",
    "relative_top_to_bottom_delta_xyz[13:16]",
    "reserved[16:20]",
]

CUBE_SIZES = {
    "cube1": np.array([0.04, 0.04, 0.04], dtype=np.float32),
    "cube2": np.array([0.05, 0.05, 0.05], dtype=np.float32),
}


@dataclass
class TrajectoryRecord:
    dataset_dir: Path
    manifest_record: dict[str, Any]
    trajectory_path: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_trajectory_path(dataset_dir: Path, record: dict[str, Any]) -> Path | None:
    raw = record.get("trajectory_path")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else dataset_dir / path


def _pose_xyz(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "xyz" in value:
            raw = value["xyz"]
        elif "position" in value:
            raw = value["position"]
        elif all(k in value for k in ("x", "y", "z")):
            raw = [value["x"], value["y"], value["z"]]
        else:
            return None
    else:
        raw = value
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _obs_key(data: np.lib.npyio.NpzFile) -> str | None:
    if "filtered_obs" in data:
        return "filtered_obs"
    if "obs" in data:
        return "obs"
    return None


def _state_from_obs(obs_row: np.ndarray, top_object: str, bottom_object: str) -> dict[str, Any]:
    cube1 = obs_row[5:8]
    cube2 = obs_row[8:11]
    poses = {"cube1": cube1, "cube2": cube2}
    return {
        "gripper_pose": obs_row[0:3],
        "gripper_open": float(obs_row[3]),
        "top_pose": poses[top_object],
        "bottom_pose": poses[bottom_object],
    }


def _infer_direction(record: dict[str, Any], top_object: str, bottom_object: str) -> str:
    direction = record.get("direction")
    if direction:
        if direction == "cube1_on_cube2":
            return "cube1_to_cube2"
        if direction == "cube2_on_cube1":
            return "cube2_to_cube1"
        return str(direction)
    if top_object and bottom_object:
        return f"{top_object}_to_{bottom_object}"
    return "unknown"


def _infer_teacher_profile(record: dict[str, Any]) -> str:
    return str(record.get("teacher_profile") or record.get("scripted_profile") or record.get("teacher_mode") or "unknown")


def _initial_pose_from_record(record: dict[str, Any], object_id: str) -> np.ndarray | None:
    scene = record.get("initial_scene") or {}
    poses = scene.get("object_poses") or {}
    return _pose_xyz(poses.get(object_id))


def _target_spec_for_phase(phase: str, state: dict[str, Any], top_size: np.ndarray) -> dict[str, Any]:
    top_pose = np.asarray(state["top_pose"], dtype=np.float32)
    bottom_pose = np.asarray(state["bottom_pose"], dtype=np.float32)
    stack_pose = bottom_pose + np.array([0.0, 0.0, max(float(top_size[2]), 0.04)], dtype=np.float32)
    grasp_pose = top_pose + np.array([0.0, 0.0, 0.03], dtype=np.float32)
    lift_z = max(float(top_pose[2]) + 0.08, float(bottom_pose[2]) + 0.10)
    retreat_pose = stack_pose + np.array([0.0, 0.0, 0.10], dtype=np.float32)

    if phase == "reach_pick":
        return {"target_gripper_pose": grasp_pose}
    if phase == "grasp":
        return {"target_gripper_open": 0.0}
    if phase == "lift":
        return {"target_top_z": lift_z}
    if phase in {"reach_place", "place"}:
        return {"target_top_pose": stack_pose}
    if phase == "release":
        return {"target_gripper_open": 1.0, "target_object_stability": 1.0}
    if phase == "retreat":
        return {"target_gripper_pose": retreat_pose}
    raise ValueError(f"Unsupported phase: {phase}")


def _role_state_vector(state: dict[str, Any], top_size: np.ndarray, bottom_size: np.ndarray) -> np.ndarray:
    gripper = np.asarray(state["gripper_pose"], dtype=np.float32)
    top = np.asarray(state["top_pose"], dtype=np.float32)
    bottom = np.asarray(state["bottom_pose"], dtype=np.float32)
    out = np.zeros((ROLE_STATE_DIM,), dtype=np.float32)
    out[0:3] = gripper
    out[3] = float(state["gripper_open"])
    out[4:7] = top
    out[7:10] = bottom
    out[10:13] = compute_relative_pose(gripper, top)
    out[13:16] = compute_relative_pose(top, bottom)
    out[16:19] = top_size
    out[19] = float(np.mean(bottom_size))
    return out


def _goal_vector(phase: str, raw_delta: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vec = np.zeros((GOAL_DELTA_DIM,), dtype=np.float32)
    mask = np.zeros((GOAL_DELTA_DIM,), dtype=np.float32)
    delta = np.asarray(raw_delta, dtype=np.float32).reshape(-1)

    if phase in {"reach_pick", "retreat"}:
        vec[0:3] = delta[:3]
        mask[0:3] = 1.0
    elif phase in {"reach_place", "place"}:
        vec[3:6] = delta[:3]
        mask[3:6] = 1.0
        vec[8:11] = delta[:3]
        mask[8:11] = 1.0
    elif phase == "grasp":
        vec[6] = float(delta[0])
        mask[6] = 1.0
    elif phase == "lift":
        vec[7] = float(delta[0])
        mask[7] = 1.0
    elif phase == "release":
        vec[6] = float(delta[0])
        mask[6] = 1.0
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return vec, mask


def _reached_vector(start_state: dict[str, Any], state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    reached = compute_reached_goal_delta(
        {
            "gripper_pose": start_state["gripper_pose"],
            "gripper_open": start_state["gripper_open"],
            "top_pose": start_state["top_pose"],
            "bottom_pose": start_state["bottom_pose"],
            "relative_gripper_to_top": compute_relative_pose(start_state["gripper_pose"], start_state["top_pose"]),
            "relative_top_to_bottom": compute_relative_pose(start_state["top_pose"], start_state["bottom_pose"]),
        },
        {
            "gripper_pose": state["gripper_pose"],
            "gripper_open": state["gripper_open"],
            "top_pose": state["top_pose"],
            "bottom_pose": state["bottom_pose"],
            "relative_gripper_to_top": compute_relative_pose(state["gripper_pose"], state["top_pose"]),
            "relative_top_to_bottom": compute_relative_pose(state["top_pose"], state["bottom_pose"]),
        },
    )
    vec = np.zeros((REACHED_GOAL_DELTA_DIM,), dtype=np.float32)
    mask = np.zeros((REACHED_GOAL_DELTA_DIM,), dtype=np.float32)
    slots = [
        ("gripper_pose", 0, 3),
        ("gripper_open", 3, 4),
        ("top_pose", 4, 7),
        ("bottom_pose", 7, 10),
        ("relative_gripper_to_top", 10, 13),
        ("relative_top_to_bottom", 13, 16),
    ]
    for key, start, end in slots:
        if key in reached:
            arr = np.asarray(reached[key], dtype=np.float32).reshape(-1)
            width = end - start
            vec[start:end] = arr[:width]
            mask[start:end] = 1.0
    return vec, mask


def _normalization(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.size == 0:
        return {"mean": [], "std": []}
    flat = arr.reshape((-1, arr.shape[-1])).astype(np.float64)
    return {
        "mean": flat.mean(axis=0).astype(float).tolist(),
        "std": np.maximum(flat.std(axis=0), 1e-6).astype(float).tolist(),
    }


def _mapping_id(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _split_indices(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        groups[(str(record.get("direction", "unknown")), int(record.get("phase_id", -1)))].append(idx)

    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    for indices in groups.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            train.extend(shuffled)
            continue
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    if records and val_ratio > 0.0 and not val and len(records) > 1:
        all_indices = list(range(len(records)))
        rng.shuffle(all_indices)
        n_val = max(1, int(round(len(records) * val_ratio)))
        n_val = min(n_val, len(records) - 1)
        val_set = set(all_indices[:n_val])
        train = [i for i in range(len(records)) if i not in val_set]
        val = sorted(val_set)
    return sorted(train), sorted(val)


def discover_trajectories(input_datasets: list[str | Path], missing: Counter) -> list[TrajectoryRecord]:
    out: list[TrajectoryRecord] = []
    for raw_dir in input_datasets:
        dataset_dir = Path(raw_dir)
        if not dataset_dir.exists():
            missing["input_dataset_missing"] += 1
            continue
        manifest = dataset_dir / "manifest.jsonl"
        if not manifest.exists():
            missing["manifest_jsonl"] += 1
            continue
        for record in _load_jsonl(manifest):
            path = _resolve_trajectory_path(dataset_dir, record)
            if path is None:
                missing["trajectory_path"] += 1
                continue
            if not path.exists():
                missing["trajectory_file_missing"] += 1
                continue
            out.append(TrajectoryRecord(dataset_dir, record, path))
    return out


def build_role_conditioned_dataset(
    *,
    input_datasets: list[str | Path],
    output_dir: str | Path,
    history_len: int = 4,
    val_ratio: float = 0.2,
    seed: int = 0,
    strict: bool = False,
) -> dict[str, Any]:
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    missing = Counter()
    skipped = Counter()
    records = discover_trajectories(input_datasets, missing)
    windows: dict[str, list[np.ndarray]] = defaultdict(list)
    manifest_records: list[dict[str, Any]] = []
    direction_counts = Counter()
    phase_counts = Counter()
    placement_bin_counts = Counter()
    teacher_profile_counts = Counter()

    # Stage 20-R: contact/holding signal tracking
    contact_label_count = 0
    contact_positive_count = 0
    contact_positive_by_phase: dict[str, int] = {}
    contact_total_by_phase: dict[str, int] = {}
    contact_positive_by_direction: dict[str, int] = {}
    contact_total_by_direction: dict[str, int] = {}
    contact_missing_count = 0

    holding_label_count = 0
    holding_positive_count = 0
    holding_positive_by_phase: dict[str, int] = {}
    holding_total_by_phase: dict[str, int] = {}
    holding_positive_by_direction: dict[str, int] = {}
    holding_total_by_direction: dict[str, int] = {}
    holding_missing_count = 0

    # Stage 20-S: track signal source distribution
    contact_source_counts: dict[str, int] = {}
    holding_source_counts: dict[str, int] = {}
    direction_to_id = dict(DEFAULT_DIRECTION_TO_ID)
    placement_bin_to_id = dict(DEFAULT_PLACEMENT_BIN_TO_ID)
    teacher_profile_to_id = dict(DEFAULT_TEACHER_PROFILE_TO_ID)
    trajectories_used = 0
    trajectories_skipped = 0

    for traj_index, item in enumerate(records):
        record = item.manifest_record
        top_object = str(record.get("object_id") or record.get("top_object") or "")
        bottom_object = str(record.get("target_object_id") or record.get("bottom_object") or "")
        if top_object not in {"cube1", "cube2"} or bottom_object not in {"cube1", "cube2"}:
            skipped["unsupported_objects"] += 1
            trajectories_skipped += 1
            if strict:
                raise ValueError(f"Unsupported objects in {item.trajectory_path}: {top_object}, {bottom_object}")
            continue

        try:
            data = np.load(item.trajectory_path, allow_pickle=True)
            obs_name = _obs_key(data)
            if obs_name is None:
                raise KeyError("obs")
            if "action" not in data:
                raise KeyError("action")
            obs = np.asarray(data[obs_name], dtype=np.float32)
            actions = np.asarray(data["action"], dtype=np.float32)
            phase_ids = np.asarray(data["phase_id"], dtype=np.int64) if "phase_id" in data else np.zeros(len(obs), dtype=np.int64)

            # Stage 20-R: extract contact/holding signals from trajectory NPZ
            # Stage 20-S: add obs[4] fallback when explicit fields missing
            contact_source = "unavailable"
            holding_source = "unavailable"

            if "contact_signal" in data:
                traj_contact = np.asarray(data["contact_signal"], dtype=np.float32).ravel()
                contact_source = "explicit_contact_signal"
            elif "gripper_has_object" in data:
                traj_contact = np.asarray(data["gripper_has_object"], dtype=np.float32).ravel()
                contact_source = "explicit_gripper_has_object"
            elif obs.shape[1] > 4:
                traj_contact = obs[:, 4].astype(np.float32).ravel()
                contact_source = "obs4_fallback_holding_like"
            else:
                traj_contact = None

            if "holding_signal" in data:
                traj_holding = np.asarray(data["holding_signal"], dtype=np.float32).ravel()
                holding_source = "explicit_holding_signal"
            elif "gripper_has_object" in data:
                traj_holding = np.asarray(data["gripper_has_object"], dtype=np.float32).ravel()
                holding_source = "explicit_gripper_has_object"
            elif obs.shape[1] > 4:
                traj_holding = obs[:, 4].astype(np.float32).ravel()
                holding_source = "obs4_fallback_holding_like"
            else:
                traj_holding = None

            traj_has_contact = traj_contact is not None
            traj_has_holding = traj_holding is not None
        except Exception as exc:
            if isinstance(exc, KeyError):
                missing[str(exc).strip("'\"")] += 1
            else:
                skipped["load_error"] += 1
            trajectories_skipped += 1
            if strict:
                raise ValueError(f"Failed to load {item.trajectory_path}: {exc}") from exc
            continue

        required_ok = obs.ndim == 2 and obs.shape[1] >= 11 and actions.ndim == 2 and len(obs) == len(actions)
        if not required_ok:
            missing["obs_action_schema"] += 1
            trajectories_skipped += 1
            if strict:
                raise ValueError(f"Invalid obs/action schema in {item.trajectory_path}")
            continue
        if len(obs) < history_len:
            skipped["too_short"] += 1
            trajectories_skipped += 1
            continue

        direction = _infer_direction(record, top_object, bottom_object)
        teacher_profile = _infer_teacher_profile(record)
        top_size = CUBE_SIZES.get(top_object, np.array([0.04, 0.04, 0.04], dtype=np.float32))
        bottom_size = CUBE_SIZES.get(bottom_object, np.array([0.04, 0.04, 0.04], dtype=np.float32))
        initial_top = _initial_pose_from_record(record, top_object)
        initial_bottom = _initial_pose_from_record(record, bottom_object)
        if initial_top is None:
            initial_top = obs[0, 5:8] if top_object == "cube1" else obs[0, 8:11]
        if initial_bottom is None:
            initial_bottom = obs[0, 5:8] if bottom_object == "cube1" else obs[0, 8:11]
        placement_bin = compute_placement_bin(initial_top, initial_bottom)

        phase_start_states: dict[int, dict[str, Any]] = {}
        trajectories_used += 1
        for t in range(history_len - 1, len(obs)):
            hist_slice = slice(t - history_len + 1, t + 1)
            obs_hist = obs[hist_slice]
            action = actions[t]

            role_hist = np.zeros((history_len, ROLE_STATE_DIM), dtype=np.float32)
            goal_hist = np.zeros((history_len, GOAL_DELTA_DIM), dtype=np.float32)
            goal_mask_hist = np.zeros((history_len, GOAL_DELTA_DIM), dtype=np.float32)
            reached_hist = np.zeros((history_len, REACHED_GOAL_DELTA_DIM), dtype=np.float32)
            reached_mask_hist = np.zeros((history_len, REACHED_GOAL_DELTA_DIM), dtype=np.float32)
            progress_hist = np.zeros((history_len,), dtype=np.float32)
            phase_hist = phase_ids[hist_slice].astype(np.int64)

            skip_window = False
            for j, step in enumerate(range(t - history_len + 1, t + 1)):
                phase_id = int(phase_ids[step])
                phase = PHASE_ID_TO_NAME.get(phase_id)
                if phase is None:
                    missing["unsupported_phase"] += 1
                    skip_window = True
                    break
                state = _state_from_obs(obs[step], top_object, bottom_object)
                phase_start_states.setdefault(phase_id, state)
                target_spec = _target_spec_for_phase(phase, state, top_size)
                raw_goal = compute_goal_delta(phase, state, target_spec)
                role_hist[j] = _role_state_vector(state, top_size, bottom_size)
                goal_hist[j], goal_mask_hist[j] = _goal_vector(phase, raw_goal, state)
                reached_hist[j], reached_mask_hist[j] = _reached_vector(phase_start_states[phase_id], state)
                initial_goal = compute_goal_delta(phase, phase_start_states[phase_id], target_spec)
                initial_error = compute_goal_error(initial_goal)
                current_error = compute_goal_error(raw_goal)
                progress_hist[j] = compute_progress_scalar(initial_error, current_error)

            if skip_window:
                skipped["window_missing_fields"] += 1
                if strict:
                    raise ValueError(f"Missing required fields in {item.trajectory_path} at step {t}")
                continue

            phase_id_current = int(phase_ids[t])
            direction_counts[direction] += 1
            phase_counts[str(phase_id_current)] += 1
            placement_bin_counts[placement_bin] += 1
            teacher_profile_counts[teacher_profile] += 1

            # Stage 20-R: track contact/holding stats
            phase_key = str(phase_id_current)
            if traj_has_contact:
                contact_label_count += 1
                contact_val = traj_contact[t]
                if contact_val > 0.5:
                    contact_positive_count += 1
                    contact_positive_by_phase[phase_key] = contact_positive_by_phase.get(phase_key, 0) + 1
                    contact_positive_by_direction[direction] = contact_positive_by_direction.get(direction, 0) + 1
                contact_total_by_phase[phase_key] = contact_total_by_phase.get(phase_key, 0) + 1
                contact_total_by_direction[direction] = contact_total_by_direction.get(direction, 0) + 1
            else:
                contact_missing_count += 1

            if traj_has_holding:
                holding_label_count += 1
                holding_val = traj_holding[t]
                if holding_val > 0.5:
                    holding_positive_count += 1
                    holding_positive_by_phase[phase_key] = holding_positive_by_phase.get(phase_key, 0) + 1
                    holding_positive_by_direction[direction] = holding_positive_by_direction.get(direction, 0) + 1
                holding_total_by_phase[phase_key] = holding_total_by_phase.get(phase_key, 0) + 1
                holding_total_by_direction[direction] = holding_total_by_direction.get(direction, 0) + 1
            else:
                holding_missing_count += 1

            windows["obs_history"].append(obs_hist.astype(np.float32))
            windows["role_state_history"].append(role_hist)
            windows["goal_delta_history"].append(goal_hist)
            windows["goal_delta_mask"].append(goal_mask_hist)
            windows["reached_goal_delta_history"].append(reached_hist)
            windows["reached_goal_delta_mask"].append(reached_mask_hist)
            windows["progress_scalar_history"].append(progress_hist)
            windows["phase_id_history"].append(phase_hist)
            windows["actions"].append(action.astype(np.float32))

            # Stage 20-R: extract contact/holding labels per-window
            if traj_has_contact:
                contact_hist = traj_contact[hist_slice]
                windows["contact_label_history"].append(contact_hist.astype(np.float32))
                windows["contact_label"].append(contact_hist[-1:].astype(np.float32))
                contact_source_counts[contact_source] = contact_source_counts.get(contact_source, 0) + 1
            if traj_has_holding:
                holding_hist = traj_holding[hist_slice]
                windows["holding_label_history"].append(holding_hist.astype(np.float32))
                windows["holding_label"].append(holding_hist[-1:].astype(np.float32))
                holding_source_counts[holding_source] = holding_source_counts.get(holding_source, 0) + 1

            manifest_records.append({
                "window_id": f"w_{len(manifest_records):08d}",
                "episode_id": record.get("episode_id", item.trajectory_path.stem),
                "trajectory_path": str(item.trajectory_path),
                "trajectory_index": traj_index,
                "step_index": t,
                "direction": direction,
                "direction_id": _mapping_id(direction_to_id, direction),
                "phase_id": phase_id_current,
                "phase_name": PHASE_ID_TO_NAME.get(phase_id_current, "unknown"),
                "placement_bin": placement_bin,
                "placement_bin_id": _mapping_id(placement_bin_to_id, placement_bin),
                "teacher_profile": teacher_profile,
                "teacher_profile_id": _mapping_id(teacher_profile_to_id, teacher_profile),
                "top_object": top_object,
                "bottom_object": bottom_object,
                "top_role": "object_to_pick",
                "bottom_role": "support_object",
                "success": bool(record.get("success", False)),
            })

    arrays: dict[str, np.ndarray] = {}
    for key, values in windows.items():
        arrays[key] = np.stack(values).astype(np.float32) if values else np.empty((0,), dtype=np.float32)
    arrays.setdefault("obs_history", np.empty((0, history_len, 0), dtype=np.float32))
    arrays.setdefault("role_state_history", np.empty((0, history_len, ROLE_STATE_DIM), dtype=np.float32))
    arrays.setdefault("goal_delta_history", np.empty((0, history_len, GOAL_DELTA_DIM), dtype=np.float32))
    arrays.setdefault("goal_delta_mask", np.empty((0, history_len, GOAL_DELTA_DIM), dtype=np.float32))
    arrays.setdefault(
        "reached_goal_delta_history",
        np.empty((0, history_len, REACHED_GOAL_DELTA_DIM), dtype=np.float32),
    )
    arrays.setdefault(
        "reached_goal_delta_mask",
        np.empty((0, history_len, REACHED_GOAL_DELTA_DIM), dtype=np.float32),
    )
    arrays.setdefault("progress_scalar_history", np.empty((0, history_len), dtype=np.float32))
    arrays.setdefault("actions", np.empty((0, 0), dtype=np.float32))
    # Stage 20-R: contact/holding arrays
    arrays.setdefault("contact_label_history", np.empty((0, history_len), dtype=np.float32))
    arrays.setdefault("contact_label", np.empty((0,), dtype=np.float32))
    arrays.setdefault("contact_label_mask", np.empty((0,), dtype=np.float32))
    arrays.setdefault("holding_label_history", np.empty((0, history_len), dtype=np.float32))
    arrays.setdefault("holding_label", np.empty((0,), dtype=np.float32))
    arrays.setdefault("holding_label_mask", np.empty((0,), dtype=np.float32))

    # Stage 20-R: build contact/holding masks from actual arrays
    if len(arrays.get("contact_label", [])) > 0:
        arrays["contact_label_mask"] = np.ones(len(arrays["contact_label"]), dtype=np.float32)
    if len(arrays.get("holding_label", [])) > 0:
        arrays["holding_label_mask"] = np.ones(len(arrays["holding_label"]), dtype=np.float32)

    # Stage 20-R: per-phase and per-direction positive rates
    def _safe_rate(positive: int, total: int) -> float:
        return round(positive / total, 6) if total > 0 else 0.0

    contact_positive_by_phase_rate = {
        k: _safe_rate(contact_positive_by_phase.get(k, 0), contact_total_by_phase.get(k, 1))
        for k in set(list(contact_positive_by_phase) + list(contact_total_by_phase))
    }
    contact_positive_by_direction_rate = {
        k: _safe_rate(contact_positive_by_direction.get(k, 0), contact_total_by_direction.get(k, 1))
        for k in set(list(contact_positive_by_direction) + list(contact_total_by_direction))
    }
    holding_positive_by_phase_rate = {
        k: _safe_rate(holding_positive_by_phase.get(k, 0), holding_total_by_phase.get(k, 1))
        for k in set(list(holding_positive_by_phase) + list(holding_total_by_phase))
    }
    holding_positive_by_direction_rate = {
        k: _safe_rate(holding_positive_by_direction.get(k, 0), holding_total_by_direction.get(k, 1))
        for k in set(list(holding_positive_by_direction) + list(holding_total_by_direction))
    }
    total_windows = len(manifest_records)

    if manifest_records:
        arrays["phase_id_history"] = np.stack(windows["phase_id_history"]).astype(np.int64)
    elif "phase_id_history" in arrays:
        arrays["phase_id_history"] = np.empty((0, history_len), dtype=np.int64)

    np.savez_compressed(output / "windows.npz", **arrays)
    train_idx, val_idx = _split_indices(manifest_records, val_ratio, seed)
    for idx, split in [(train_idx, "train"), (val_idx, "val")]:
        for i in idx:
            manifest_records[i]["split"] = split
    _write_jsonl(output / "manifest.jsonl", manifest_records)
    _write_jsonl(output / "train_manifest.jsonl", [manifest_records[i] for i in train_idx])
    _write_jsonl(output / "val_manifest.jsonl", [manifest_records[i] for i in val_idx])

    obs_dim = int(arrays["obs_history"].shape[-1]) if arrays.get("obs_history", np.empty((0,))).ndim >= 3 else 0
    action_dim = int(arrays["actions"].shape[-1]) if arrays.get("actions", np.empty((0,))).ndim >= 2 else 0
    can_train = bool(len(manifest_records) > 0 and obs_dim > 0 and action_dim > 0)
    if can_train:
        recommendation = "ready_for_training"
    elif missing or skipped:
        recommendation = "need_recorder_patch"
    else:
        recommendation = "need_teacher_collection"

    summary = {
        "num_input_datasets": len(input_datasets),
        "num_trajectories_loaded": len(records),
        "num_trajectories_used": trajectories_used,
        "num_trajectories_skipped": trajectories_skipped,
        "num_windows": len(manifest_records),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "role_state_dim": ROLE_STATE_DIM,
        "goal_delta_dim": GOAL_DELTA_DIM,
        "reached_goal_delta_dim": REACHED_GOAL_DELTA_DIM,
        "history_len": history_len,
        "direction_counts": dict(direction_counts),
        "phase_counts": dict(phase_counts),
        "placement_bin_counts": dict(placement_bin_counts),
        "teacher_profile_counts": dict(teacher_profile_counts),
        "missing_field_counts": dict(missing),
        "skipped_reason_counts": dict(skipped),
        "can_train_mlp_baseline": can_train,
        "recommendation": recommendation,
        "phase_to_id": PHASE_TO_ID,
        "direction_to_id": direction_to_id,
        "placement_bin_to_id": placement_bin_to_id,
        "teacher_profile_to_id": teacher_profile_to_id,
        "role_state_layout": ROLE_STATE_LAYOUT,
        "goal_delta_layout": GOAL_DELTA_LAYOUT,
        "reached_goal_delta_layout": REACHED_GOAL_DELTA_LAYOUT,
        # Stage 20-R: contact/holding audit fields
        "contact_label_available_rate": round(contact_label_count / total_windows, 6) if total_windows > 0 else 0.0,
        "holding_label_available_rate": round(holding_label_count / total_windows, 6) if total_windows > 0 else 0.0,
        "contact_positive_rate": round(contact_positive_count / contact_label_count, 6) if contact_label_count > 0 else 0.0,
        "holding_positive_rate": round(holding_positive_count / holding_label_count, 6) if holding_label_count > 0 else 0.0,
        "contact_positive_rate_by_phase": contact_positive_by_phase_rate,
        "holding_positive_rate_by_phase": holding_positive_by_phase_rate,
        "contact_positive_rate_by_direction": contact_positive_by_direction_rate,
        "holding_positive_rate_by_direction": holding_positive_by_direction_rate,
        "contact_missing_count": contact_missing_count,
        "holding_missing_count": holding_missing_count,
        "contact_signal_source": "obs_index_4_gripper_has_object_heuristic",
        "holding_signal_source": "obs_index_4_gripper_has_object_heuristic",
        # Stage 20-S: signal source distribution
        "contact_signal_source_counts": dict(contact_source_counts),
        "holding_signal_source_counts": dict(holding_source_counts),
        "no_per_object_contact": True,
        "no_direct_mujoco_contact": True,
        "true_mujoco_contact_available": False,
        "per_object_contact_available": False,
        "no_phase_proxy_labels": True,
        "holding_label_caveat": "obs[4] gripper_has_object heuristic — NOT true MuJoCo contact. Holding-like signal useful for lift/transport verification but not per-object contact detection.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = {
        "obs_history": _normalization(arrays["obs_history"]),
        "role_state_history": _normalization(arrays["role_state_history"]),
        "goal_delta_history": _normalization(arrays["goal_delta_history"]),
        "reached_goal_delta_history": _normalization(arrays["reached_goal_delta_history"]),
        "actions": _normalization(arrays["actions"]),
    }
    (output / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_readme(output, summary)
    return summary


def _write_readme(output: Path, summary: dict[str, Any]) -> None:
    text = f"""# Stage 20 Role-Conditioned Stack Skill Dataset

This dataset was built for:

```text
obs_history + role_state + goal_delta + reached_goal_delta + progress_scalar + phase_id -> action
```

## Files

- `windows.npz`: dense arrays for training.
- `manifest.jsonl`: one record per window.
- `train_manifest.jsonl`: training split.
- `val_manifest.jsonl`: validation split.
- `summary.json`: counts, mappings, and audit fields.
- `normalization_stats.json`: mean/std for core tensors.

## Feature Layout

Role state dim: `{ROLE_STATE_DIM}`

```text
{chr(10).join(ROLE_STATE_LAYOUT)}
```

Goal delta dim: `{GOAL_DELTA_DIM}`

```text
{chr(10).join(GOAL_DELTA_LAYOUT)}
```

Reached-goal delta dim: `{REACHED_GOAL_DELTA_DIM}`

```text
{chr(10).join(REACHED_GOAL_DELTA_LAYOUT)}
```

Goal and reached-goal masks indicate which slots are active for the current phase.

## Summary

```json
{json.dumps(summary, indent=2, ensure_ascii=False)}
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 20 role-conditioned skill dataset")
    parser.add_argument("--input_datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", default="false")
    args = parser.parse_args()

    strict = str(args.strict).lower() in {"1", "true", "yes", "y"}
    summary = build_role_conditioned_dataset(
        input_datasets=args.input_datasets,
        output_dir=args.output_dir,
        history_len=args.history_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
        strict=strict,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
