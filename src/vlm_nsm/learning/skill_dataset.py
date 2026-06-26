"""Dataset utilities for skill-labeled demonstrations.

Loads Stage 19 skill demonstration datasets and produces training samples:

``obs_history + goal_condition -> action_chunk``

The implementation avoids crossing episode boundaries by indexing windows
within a single trajectory file at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


DEFAULT_SKILL_TO_ID = {
    "pick_place": 0,
    "pick": 1,
    "place": 2,
    "place_on": 3,
    "stack": 4,
}

DEFAULT_OBJECT_TO_ID = {
    "": 0,
    "table": 1,
    "cube1": 2,
    "cube2": 3,
    "cube3": 4,
    "cube4": 5,
    "cube5": 6,
}


def load_manifest(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load ``manifest.jsonl`` records from a skill demo dataset."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_normalization_stats(dataset_dir: str | Path, only_success: bool = True) -> dict[str, list[float]]:
    """Compute observation/action mean and std over selected trajectories."""
    records = load_manifest(dataset_dir)
    root = Path(dataset_dir)
    obs_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []

    for record in records:
        if only_success and not record.get("success", False):
            continue
        traj_path = record.get("trajectory_path")
        if not traj_path:
            continue
        data = np.load(root / traj_path, allow_pickle=True)
        obs_arrays.append(np.asarray(data["obs"], dtype=np.float32))
        action_arrays.append(np.asarray(data["action"], dtype=np.float32))

    if not obs_arrays or not action_arrays:
        raise ValueError("No trajectories available for normalization stats")

    obs = np.concatenate(obs_arrays, axis=0)
    action = np.concatenate(action_arrays, axis=0)
    return {
        "obs_mean": obs.mean(axis=0).astype(float).tolist(),
        "obs_std": np.maximum(obs.std(axis=0), 1e-6).astype(float).tolist(),
        "action_mean": action.mean(axis=0).astype(float).tolist(),
        "action_std": np.maximum(action.std(axis=0), 1e-6).astype(float).tolist(),
    }


class SkillDataset(Dataset):
    """Windowed dataset for skill policy learning."""

    def __init__(
        self,
        dataset_dir: str | Path,
        obs_history_len: int = 10,
        action_chunk_len: int = 16,
        *,
        normalize: bool = False,
        normalization_stats: dict[str, Any] | None = None,
        phase_filter: list[int] | set[int] | None = None,
        skill_filter: list[str] | set[str] | None = None,
        include_failures: bool = False,
        stride: int = 1,
    ):
        if obs_history_len <= 0:
            raise ValueError("obs_history_len must be positive")
        if action_chunk_len <= 0:
            raise ValueError("action_chunk_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.dataset_dir = Path(dataset_dir)
        self.obs_history_len = obs_history_len
        self.action_chunk_len = action_chunk_len
        self.normalize = normalize
        self.phase_filter = set(phase_filter) if phase_filter is not None else None
        self.skill_filter = set(skill_filter) if skill_filter is not None else None
        self.include_failures = include_failures
        self.stride = stride

        self.records = load_manifest(self.dataset_dir)
        if normalize:
            normalization_stats = normalization_stats or compute_normalization_stats(
                self.dataset_dir,
                only_success=not include_failures,
            )
        self.normalization_stats = normalization_stats

        self._trajectories: list[dict[str, Any]] = []
        self._index: list[tuple[int, int]] = []
        self._load_trajectories()

    def _load_trajectories(self) -> None:
        for record in self.records:
            if not self.include_failures and not record.get("success", False):
                continue
            if self.skill_filter is not None and record.get("skill_name") not in self.skill_filter:
                continue
            traj_path = record.get("trajectory_path")
            if not traj_path:
                continue

            data = np.load(self.dataset_dir / traj_path, allow_pickle=True)
            obs = np.asarray(data["obs"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            phase_id = np.asarray(data["phase_id"], dtype=np.int64) if "phase_id" in data else None

            if len(obs) != len(action):
                raise ValueError(f"obs/action length mismatch in {traj_path}")
            if len(obs) < self.obs_history_len + self.action_chunk_len:
                continue

            trajectory = {
                "record": record,
                "path": traj_path,
                "obs": obs,
                "action": action,
                "phase_id": phase_id,
                "skill_name": record.get("skill_name", ""),
                "object_id": record.get("object_id", ""),
                "target_object_id": record.get("target_object_id", ""),
                "success": bool(record.get("success", False)),
            }
            traj_idx = len(self._trajectories)
            self._trajectories.append(trajectory)

            max_start = len(obs) - self.obs_history_len - self.action_chunk_len + 1
            for start in range(0, max_start, self.stride):
                chunk_start = start + self.obs_history_len
                chunk_end = chunk_start + self.action_chunk_len
                if self.phase_filter is not None and phase_id is not None:
                    chunk_phases = set(int(p) for p in phase_id[chunk_start:chunk_end])
                    if not chunk_phases.issubset(self.phase_filter):
                        continue
                self._index.append((traj_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        traj_idx, start = self._index[index]
        trajectory = self._trajectories[traj_idx]
        obs_start = start
        obs_end = start + self.obs_history_len
        action_start = obs_end
        action_end = action_start + self.action_chunk_len

        obs_history = trajectory["obs"][obs_start:obs_end].copy()
        action_chunk = trajectory["action"][action_start:action_end].copy()

        if self.normalize:
            obs_history = self._normalize(obs_history, "obs")
            action_chunk = self._normalize(action_chunk, "action")

        goal_condition = np.array(
            [
                DEFAULT_SKILL_TO_ID.get(trajectory["skill_name"], -1),
                DEFAULT_OBJECT_TO_ID.get(trajectory["object_id"], -1),
                DEFAULT_OBJECT_TO_ID.get(trajectory["target_object_id"] or "", -1),
            ],
            dtype=np.int64,
        )

        phase_id = trajectory["phase_id"]
        phase_chunk = (
            phase_id[action_start:action_end].copy()
            if phase_id is not None
            else np.full((self.action_chunk_len,), -1, dtype=np.int64)
        )

        return {
            "obs_history": obs_history.astype(np.float32),
            "action_chunk": action_chunk.astype(np.float32),
            "goal_condition": goal_condition,
            "skill_id": int(goal_condition[0]),
            "object_id": trajectory["object_id"],
            "target_object_id": trajectory["target_object_id"],
            "phase_id": phase_chunk,
            "success": trajectory["success"],
            "trajectory_path": trajectory["path"],
            "start_index": start,
        }

    def _normalize(self, value: np.ndarray, prefix: str) -> np.ndarray:
        if not self.normalization_stats:
            raise ValueError("normalization_stats are required when normalize=True")
        mean = np.asarray(self.normalization_stats[f"{prefix}_mean"], dtype=np.float32)
        std = np.asarray(self.normalization_stats[f"{prefix}_std"], dtype=np.float32)
        return (value - mean) / std

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)
