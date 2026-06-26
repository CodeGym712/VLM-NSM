"""Dataset loader for Stage 20 role-conditioned skill windows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset


FeatureMode = Literal["obs_only", "goal_conditioned"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _window_index(record: dict[str, Any]) -> int:
    raw = str(record.get("window_id", ""))
    match = re.search(r"(\d+)$", raw)
    if not match:
        raise ValueError(f"Cannot parse window_id: {raw}")
    return int(match.group(1))


def _stats_array(stats: dict[str, Any], section: str, key: str, dim: int) -> np.ndarray:
    values = stats.get(section, {}).get(key, [])
    if not values:
        return np.zeros((dim,), dtype=np.float32) if key == "mean" else np.ones((dim,), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"Normalization stat {section}.{key} dim mismatch: {arr.shape[0]} != {dim}")
    if key == "std":
        arr = np.maximum(arr, 1e-6)
    return arr


class Stage20RoleConditionedDataset(Dataset):
    """Loads role-conditioned training windows from a Stage 20 dataset dir."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: Literal["train", "val", "all"] = "train",
        feature_mode: FeatureMode = "goal_conditioned",
        normalize: bool = True,
        num_phase_ids: int = 6,
        return_sequence: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.feature_mode = feature_mode
        self.normalize = normalize
        self.num_phase_ids = num_phase_ids
        self.return_sequence = return_sequence

        if feature_mode not in {"obs_only", "goal_conditioned"}:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unsupported split: {split}")

        windows_path = self.dataset_dir / "windows.npz"
        if not windows_path.exists():
            raise FileNotFoundError(windows_path)
        with np.load(windows_path, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.summary = _load_json(self.dataset_dir / "summary.json") if (self.dataset_dir / "summary.json").exists() else {}
        self.stats = (
            _load_json(self.dataset_dir / "normalization_stats.json")
            if (self.dataset_dir / "normalization_stats.json").exists()
            else {}
        )

        manifest_path = self.dataset_dir / (
            "train_manifest.jsonl" if split == "train" else "val_manifest.jsonl" if split == "val" else "manifest.jsonl"
        )
        self.records = _load_jsonl(manifest_path)
        self.indices = np.asarray([_window_index(r) for r in self.records], dtype=np.int64)
        self._validate_arrays()
        self._prepare_normalization()

    def _validate_arrays(self) -> None:
        required = [
            "obs_history",
            "role_state_history",
            "goal_delta_history",
            "reached_goal_delta_history",
            "progress_scalar_history",
            "phase_id_history",
            "actions",
        ]
        for key in required:
            if key not in self.arrays:
                raise ValueError(f"windows.npz missing {key}")
        n = int(self.arrays["actions"].shape[0])
        for idx in self.indices:
            if idx < 0 or idx >= n:
                raise ValueError(f"Window index out of range: {idx} >= {n}")
        for key in required:
            arr = np.asarray(self.arrays[key])
            if not np.isfinite(arr.astype(np.float32)).all():
                raise ValueError(f"Non-finite values in {key}")

    def _prepare_normalization(self) -> None:
        obs_dim = int(self.arrays["obs_history"].shape[-1])
        role_dim = int(self.arrays["role_state_history"].shape[-1])
        goal_dim = int(self.arrays["goal_delta_history"].shape[-1])
        reached_dim = int(self.arrays["reached_goal_delta_history"].shape[-1])
        action_dim = int(self.arrays["actions"].shape[-1])
        self.norm = {
            "obs_history": (
                _stats_array(self.stats, "obs_history", "mean", obs_dim),
                _stats_array(self.stats, "obs_history", "std", obs_dim),
            ),
            "role_state_history": (
                _stats_array(self.stats, "role_state_history", "mean", role_dim),
                _stats_array(self.stats, "role_state_history", "std", role_dim),
            ),
            "goal_delta_history": (
                _stats_array(self.stats, "goal_delta_history", "mean", goal_dim),
                _stats_array(self.stats, "goal_delta_history", "std", goal_dim),
            ),
            "reached_goal_delta_history": (
                _stats_array(self.stats, "reached_goal_delta_history", "mean", reached_dim),
                _stats_array(self.stats, "reached_goal_delta_history", "std", reached_dim),
            ),
            "actions": (
                _stats_array(self.stats, "actions", "mean", action_dim),
                _stats_array(self.stats, "actions", "std", action_dim),
            ),
        }

    @property
    def input_dim(self) -> int:
        if not len(self):
            return 0
        features = self[0]["features"]
        return int(features.shape[-1]) if self.return_sequence else int(features.numel())

    @property
    def action_dim(self) -> int:
        return int(self.arrays["actions"].shape[-1])

    def __len__(self) -> int:
        return len(self.indices)

    def _norm(self, key: str, value: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return value.astype(np.float32)
        mean, std = self.norm[key]
        return ((value.astype(np.float32) - mean) / std).astype(np.float32)

    def _phase_one_hot(self, phase_ids: np.ndarray) -> np.ndarray:
        out = np.zeros((*phase_ids.shape, self.num_phase_ids), dtype=np.float32)
        valid = (phase_ids >= 0) & (phase_ids < self.num_phase_ids)
        if valid.any():
            coords = np.where(valid)
            out[(*coords, phase_ids[valid].astype(np.int64))] = 1.0
        return out

    def __getitem__(self, item: int) -> dict[str, Any]:
        idx = int(self.indices[item])
        record = self.records[item]
        obs = self._norm("obs_history", np.asarray(self.arrays["obs_history"][idx], dtype=np.float32))
        action = np.asarray(self.arrays["actions"][idx], dtype=np.float32)
        phase_ids = np.asarray(self.arrays["phase_id_history"][idx], dtype=np.int64)

        if self.feature_mode == "obs_only":
            sequence = obs
        else:
            role = self._norm("role_state_history", np.asarray(self.arrays["role_state_history"][idx], dtype=np.float32))
            goal = self._norm("goal_delta_history", np.asarray(self.arrays["goal_delta_history"][idx], dtype=np.float32))
            reached = self._norm(
                "reached_goal_delta_history",
                np.asarray(self.arrays["reached_goal_delta_history"][idx], dtype=np.float32),
            )
            progress = np.asarray(self.arrays["progress_scalar_history"][idx], dtype=np.float32).reshape(-1, 1)
            goal_mask = np.asarray(self.arrays["goal_delta_mask"][idx], dtype=np.float32) if "goal_delta_mask" in self.arrays else np.ones_like(goal)
            reached_mask = (
                np.asarray(self.arrays["reached_goal_delta_mask"][idx], dtype=np.float32)
                if "reached_goal_delta_mask" in self.arrays
                else np.ones_like(reached)
            )
            phase_oh = self._phase_one_hot(phase_ids)
            sequence = np.concatenate(
                [
                    obs,
                    role,
                    goal,
                    goal_mask,
                    reached,
                    reached_mask,
                    progress,
                    phase_oh,
                ],
                axis=1,
            )
        features = sequence if self.return_sequence else sequence.reshape(-1)

        if not np.isfinite(features).all() or not np.isfinite(action).all():
            raise ValueError(f"Non-finite sample at window index {idx}")
        result = {
            "features": torch.from_numpy(features.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "obs_history": torch.from_numpy(obs.astype(np.float32)),
            "phase_id_history": torch.from_numpy(phase_ids),
            "direction_label": int(record.get("direction_id", -1)),
            "phase_label": int(record.get("phase_id", -1)),
            "window_id": record.get("window_id", f"w_{idx:08d}"),
            "index": idx,
        }
        # Stage 20-S: contact/holding labels if available
        for key in ("holding_label", "holding_label_mask",
                     "contact_label", "contact_label_mask"):
            if key in self.arrays:
                arr = np.asarray(self.arrays[key][idx], dtype=np.float32)
                result[key] = torch.from_numpy(arr)
        return result
