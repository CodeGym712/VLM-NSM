"""Robosuite demonstration dataset for behavior cloning training.

Loads full-trajectory npz files produced by DemonstrationRecorder
(obs, action, done, episode_id, success arrays) and provides
episode-level train/val split with action normalization.

Supports single-npz loading (with episode_id split) and
directory-based loading (one npz file per episode).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vlm_nsm.learning.robosuite_schema import ACTION_DIM, OBS_DIM, OBS_KEYS, build_robosuite_obs_vector


class RobosuiteDemoDataset(Dataset):
    """Load demonstration npz (full trajectories) for BC training.

    The npz file is expected to contain at minimum:
      - ``obs`` or specified ``obs_key``: (T, obs_dim) float32 — observation vectors
      - ``action``: (T, action_dim) float32 — action vectors
      - ``episode_id``: (T,) int32 — episode index per timestep (optional for directory mode)
      - ``success``: (T,) bool — per-timestep success flag (optional)

    Two loading modes:
      1. Single npz (``npz_path``): one file with ``episode_id`` for episode boundaries.
      2. Directory (``from_directory``): each npz file = one episode.
    """

    def __init__(
        self,
        npz_path: str | Path | None = None,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
        obs_key: str = "obs",
        episode_files: list[Path] | None = None,
        skip_split: bool = False,
        # Stage 7B-2: phase conditioning
        phase_key: str = "phase_id",
        condition_on_phase: bool = False,
        # Stage 7B-2: action norm diagnostics
        near_zero_std_dims: list[int] | None = None,
        action_std_raw: np.ndarray | None = None,
        fixed_zero_action_dims: list[int] | None = None,
    ):
        """Initialize dataset.

        Args:
            npz_path: Path to a single combined npz (optional if episode_files given).
            split: 'train' or 'val'.
            val_ratio: Fraction of episodes for validation.
            seed: Random seed for split.
            action_mean: Precomputed action mean (from train set).
            action_std: Precomputed action std (from train set).
            obs_key: Key to use for observations in the npz (default: 'obs').
            episode_files: List of per-episode npz files (directory mode).
            skip_split: If True, use all episode_files without re-splitting
                (used by from_directory which already splits the file list).
            phase_key: Key for phase_id in npz (default: 'phase_id').
            condition_on_phase: If True, append phase one-hot to obs.
            near_zero_std_dims: List of action dims with near-zero std.
            action_std_raw: Raw action std before flooring.
            fixed_zero_action_dims: List of action dims fixed at zero.
        """
        super().__init__()
        self.obs_key = obs_key
        self.action_mean = action_mean
        self.action_std = action_std
        self.phase_key = phase_key
        self.condition_on_phase = condition_on_phase
        self.near_zero_std_dims = near_zero_std_dims or []
        self.action_std_raw = action_std_raw
        self.fixed_zero_action_dims = fixed_zero_action_dims or []
        self.phase_id: np.ndarray | None = None
        self._n_phases: int = 6  # reach_pick, grasp, lift, reach_place, release, retreat

        if episode_files is not None:
            self._load_from_directory(episode_files, split, val_ratio, seed, skip_split)
        elif npz_path is not None:
            self._load_from_npz(npz_path, split, val_ratio, seed)
        else:
            raise ValueError("Either npz_path or episode_files must be provided")

    def _load_from_npz(
        self,
        npz_path: str | Path,
        split: str,
        val_ratio: float,
        seed: int,
    ):
        """Load from a single combined npz with episode_id for boundaries."""
        data = np.load(npz_path)
        self.obs: np.ndarray = data[self.obs_key]  # (T, obs_dim)
        self.action: np.ndarray = data["action"]  # (T, action_dim)
        self.episode_id: np.ndarray = data["episode_id"]  # (T,)
        self.success: np.ndarray = data.get("success", np.ones(len(self.obs), dtype=bool))  # (T,)

        # Stage 7B-2: load phase_id if available
        if self.condition_on_phase:
            if self.phase_key in data:
                self.phase_id = data[self.phase_key].astype(np.int64)
            elif "skill_id" in data:
                # Fallback to skill_id (legacy demos)
                self.phase_id = data["skill_id"].astype(np.int64)
            else:
                raise KeyError(
                    f"condition_on_phase=True but '{self.phase_key}' not found in npz. "
                    f"Available keys: {list(data.keys())}"
                )

        self.obs_dim = self.obs.shape[-1]
        self.action_dim = self.action.shape[-1]

        # Compute episode boundaries
        self._episodes = _compute_episode_boundaries(self.episode_id)
        _n_eps = len(self._episodes)
        rng = np.random.RandomState(seed)
        _n_val = max(1, int(_n_eps * val_ratio))
        _val_indices = set(rng.choice(_n_eps, _n_val, replace=False).tolist())

        if split == "val":
            self._episode_indices = sorted(_val_indices)
        else:
            self._episode_indices = [i for i in range(_n_eps) if i not in _val_indices]

        # Build flat step index
        self._flat_indices: list[int] = []
        for ep_idx in self._episode_indices:
            start, end = self._episodes[ep_idx]
            self._flat_indices.extend(range(start, end))

    def _load_from_directory(
        self,
        episode_files: list[Path],
        split: str,
        val_ratio: float,
        seed: int,
        skip_split: bool = False,
    ):
        """Load from a directory of per-episode npz files."""
        _n_eps = len(episode_files)
        if skip_split:
            # from_directory already split the file list; use all files
            selected_indices = list(range(_n_eps))
        else:
            rng = np.random.RandomState(seed)
            _n_val = max(1, int(_n_eps * val_ratio))
            perm = rng.permutation(_n_eps)
            _val_indices = set(perm[:_n_val].tolist())

            if split == "val":
                selected_indices = sorted(_val_indices)
            else:
                selected_indices = sorted([i for i in range(_n_eps) if i not in _val_indices])

        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        phase_id_list: list[np.ndarray] = []
        self._episodes: list[tuple[int, int]] = []
        current_offset = 0

        for i in range(_n_eps):
            if i not in selected_indices:
                continue
            ep_data = np.load(episode_files[i])
            ep_obs = ep_data[self.obs_key]  # (T, obs_dim)
            ep_action = ep_data["action"]  # (T, action_dim)
            T = len(ep_obs)
            obs_list.append(ep_obs)
            action_list.append(ep_action)
            if self.condition_on_phase:
                if self.phase_key in ep_data:
                    phase_id_list.append(ep_data[self.phase_key].astype(np.int64))
                elif "skill_id" in ep_data:
                    phase_id_list.append(ep_data["skill_id"].astype(np.int64))
                else:
                    raise KeyError(
                        f"condition_on_phase=True but '{self.phase_key}' not found in {episode_files[i]}. "
                        f"Available keys: {list(ep_data.keys())}"
                    )
            self._episodes.append((current_offset, current_offset + T))
            current_offset += T

        if not obs_list:
            self.obs = np.empty((0, 0), dtype=np.float32)
            self.action = np.empty((0, 0), dtype=np.float32)
            self.obs_dim = 0
            self.action_dim = 0
            self._flat_indices = []
            self._episodes = []
            self._episode_indices = []
            if self.condition_on_phase:
                self.phase_id = np.empty((0,), dtype=np.int64)
            return

        self.obs = np.concatenate(obs_list, axis=0).astype(np.float32)
        self.action = np.concatenate(action_list, axis=0).astype(np.float32)
        self.obs_dim = self.obs.shape[-1]
        self.action_dim = self.action.shape[-1]

        if self.condition_on_phase and phase_id_list:
            self.phase_id = np.concatenate(phase_id_list, axis=0).astype(np.int64)

        self._flat_indices: list[int] = list(range(len(self.obs)))
        self._episode_indices = list(range(len(self._episodes)))

    def __len__(self) -> int:
        return len(self._flat_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_step = self._flat_indices[idx]
        obs = torch.from_numpy(self.obs[global_step]).float()
        action = torch.from_numpy(self.action[global_step]).float()

        # Stage 7B-2: phase conditioning — append one-hot to obs
        if self.condition_on_phase and self.phase_id is not None:
            phase = int(self.phase_id[global_step])
            phase_one_hot = torch.zeros(self._n_phases, dtype=torch.float32)
            if 0 <= phase < self._n_phases:
                phase_one_hot[phase] = 1.0
            obs = torch.cat([obs, phase_one_hot])

        # Apply normalization if available
        if self.action_mean is not None and self.action_std is not None:
            action_mean_t = torch.from_numpy(self.action_mean).float()
            action_std_t = torch.from_numpy(self.action_std).float()
            action = (action - action_mean_t) / action_std_t

        return {"obs": obs, "action": action}

    def _add_phase_to_obs(self, obs: torch.Tensor, global_step: int) -> torch.Tensor:
        """Append phase one-hot encoding to observation tensor."""
        if self.phase_id is not None:
            phase = int(self.phase_id[global_step])
            phase_one_hot = torch.zeros(self._n_phases, dtype=torch.float32)
            if 0 <= phase < self._n_phases:
                phase_one_hot[phase] = 1.0
            obs = torch.cat([obs, phase_one_hot])
        return obs

    @property
    def num_episodes(self) -> int:
        return len(self._episode_indices)

    @property
    def obs_dim_with_phase(self) -> int:
        """Return effective obs dim (including phase one-hot if conditioned)."""
        if self.condition_on_phase:
            return self.obs_dim + self._n_phases
        return self.obs_dim

    def get_episode(self, ep_local_idx: int) -> dict[str, torch.Tensor]:
        """Return full observation and action tensors for a single episode.

        Args:
            ep_local_idx: Index within this dataset's episode list (0..num_episodes-1).

        Returns:
            Dict with ``obs`` (T, obs_dim) and ``action`` (T, action_dim).
        """
        ep_idx = self._episode_indices[ep_local_idx]
        start, end = self._episodes[ep_idx]
        obs = torch.from_numpy(self.obs[start:end]).float()
        action = torch.from_numpy(self.action[start:end]).float()
        if self.action_mean is not None and self.action_std is not None:
            mean_t = torch.from_numpy(self.action_mean).float()
            std_t = torch.from_numpy(self.action_std).float()
            action = (action - mean_t) / std_t
        # Phase conditioning for full episode
        if self.condition_on_phase and self.phase_id is not None:
            result = {"action": action}
            obs_list = [obs]
            phase_one_hots = []
            for t in range(start, end):
                phase = int(self.phase_id[t])
                oh = torch.zeros(self._n_phases, dtype=torch.float32)
                if 0 <= phase < self._n_phases:
                    oh[phase] = 1.0
                phase_one_hots.append(oh)
            obs_with_phase = torch.cat([obs, torch.stack(phase_one_hots)], dim=1)
            result["obs"] = obs_with_phase
            return result
        return {"obs": obs, "action": action}

    @classmethod
    def train_val(
        cls,
        npz_path: str | Path,
        val_ratio: float = 0.2,
        seed: int = 42,
        obs_key: str = "obs",
        episode_split: bool = True,
        # Stage 7B-2: phase conditioning
        condition_on_phase: bool = False,
        phase_key: str = "phase_id",
        # Stage 7B-2: action norm floor
        min_action_std: float = 1e-3,
    ) -> tuple[RobosuiteDemoDataset, RobosuiteDemoDataset]:
        """Create train/val datasets from a single npz with action normalization.

        Normalization statistics (mean, std) are computed exclusively on
        the **training** episodes to avoid data leakage.

        Args:
            npz_path: Path to the demonstration npz file.
            val_ratio: Fraction of episodes reserved for validation.
            seed: Random seed for deterministic split.
            obs_key: Key to use for observations in the npz.
            episode_split: If True, split by episode_id. If False, each row
                is treated as independent (no episode boundary detection).
            condition_on_phase: If True, append phase one-hot to obs.
            phase_key: Key for phase_id in npz.
            min_action_std: Minimum action std for numerical stability.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if not episode_split:
            # Simple row-level split (no episode boundaries)
            data = np.load(npz_path)
            action = data["action"]
            obs = data[obs_key]
            T = len(obs)
            rng = np.random.RandomState(seed)
            n_val = max(1, int(T * val_ratio))
            val_indices = set(rng.choice(T, n_val, replace=False).tolist())

            all_actions = action
            action_mean = np.mean(all_actions, axis=0).astype(np.float32)
            action_std_raw = np.std(all_actions, axis=0).astype(np.float32) + 1e-8
            # Apply min_action_std floor
            action_std = np.maximum(action_std_raw, min_action_std)
            near_zero_dims = np.where(action_std_raw < min_action_std)[0].tolist()

            train_ds = cls.__new__(cls)
            val_ds = cls.__new__(cls)
            # Manual init for row-level split
            for ds, split_name in [(train_ds, "train"), (val_ds, "val")]:
                ds.obs_key = obs_key
                ds.action_mean = action_mean
                ds.action_std = action_std
                ds.action_std_raw = action_std_raw
                ds.near_zero_std_dims = near_zero_dims
                ds.obs = obs
                ds.action = action
                ds.obs_dim = obs.shape[-1]
                ds.action_dim = action.shape[-1]
                ds.condition_on_phase = condition_on_phase
                ds.phase_key = phase_key
                ds.phase_id = None
                ds._n_phases = 6
                ds._flat_indices = sorted(set(range(T)) - val_indices) if split_name == "train" else sorted(val_indices)
                ds._episode_indices = list(range(len(ds._flat_indices)))
                ds._episodes = [(i, i + 1) for i in ds._flat_indices]
            return train_ds, val_ds

        # Episode-level split (original behavior)
        data = np.load(npz_path)
        action = data["action"]
        episode_id = data["episode_id"]
        episodes = _compute_episode_boundaries(episode_id)

        rng = np.random.RandomState(seed)
        n_eps = len(episodes)
        n_val = max(1, int(n_eps * val_ratio))
        val_indices = set(rng.choice(n_eps, n_val, replace=False).tolist())

        # Collect train-set actions only
        train_actions: list[np.ndarray] = []
        for i in range(n_eps):
            if i not in val_indices:
                start, end = episodes[i]
                train_actions.append(action[start:end])
        all_train_actions = np.concatenate(train_actions, axis=0)

        action_mean = np.mean(all_train_actions, axis=0).astype(np.float32)
        action_std_raw = np.std(all_train_actions, axis=0).astype(np.float32) + 1e-8
        # Apply min_action_std floor
        action_std = np.maximum(action_std_raw, min_action_std)
        near_zero_dims = np.where(action_std_raw < min_action_std)[0].tolist()

        train_ds = cls(
            npz_path=npz_path, split="train", val_ratio=val_ratio, seed=seed,
            action_mean=action_mean, action_std=action_std, obs_key=obs_key,
            condition_on_phase=condition_on_phase, phase_key=phase_key,
            near_zero_std_dims=near_zero_dims, action_std_raw=action_std_raw,
        )
        val_ds = cls(
            npz_path=npz_path, split="val", val_ratio=val_ratio, seed=seed,
            action_mean=action_mean, action_std=action_std, obs_key=obs_key,
            condition_on_phase=condition_on_phase, phase_key=phase_key,
            near_zero_std_dims=near_zero_dims, action_std_raw=action_std_raw,
        )
        return train_ds, val_ds

    @classmethod
    def from_directory(
        cls,
        demo_dir: str | Path,
        val_ratio: float = 0.2,
        seed: int = 42,
        obs_key: str = "obs",
        # Stage 7B-2: phase conditioning
        condition_on_phase: bool = False,
        phase_key: str = "phase_id",
        # Stage 7B-2: action norm floor
        min_action_std: float = 1e-3,
    ) -> tuple[RobosuiteDemoDataset, RobosuiteDemoDataset]:
        """Create train/val datasets from a directory of per-episode npz files.

        Each npz file in the directory must contain ``obs_key`` and ``action`` arrays.
        Each file represents one episode. Train/val split is at the episode level.

        Args:
            demo_dir: Directory containing per-episode npz files.
            val_ratio: Fraction of episodes reserved for validation.
            seed: Random seed for deterministic split.
            obs_key: Key to use for observations in the npz files.
            condition_on_phase: If True, append phase one-hot to obs.
            phase_key: Key for phase_id in npz.
            min_action_std: Minimum action std for numerical stability.

        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        demo_dir = Path(demo_dir)
        npz_files = sorted(demo_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {demo_dir}")

        # Compute action normalization stats from all episodes, then split
        all_actions: list[np.ndarray] = []
        for f in npz_files:
            ep_data = np.load(f)
            all_actions.append(ep_data["action"])
        all_actions_arr = np.concatenate(all_actions, axis=0)
        action_mean = np.mean(all_actions_arr, axis=0).astype(np.float32)
        action_std_raw = np.std(all_actions_arr, axis=0).astype(np.float32) + 1e-8
        # Apply min_action_std floor
        action_std = np.maximum(action_std_raw, min_action_std)
        near_zero_dims = np.where(action_std_raw < min_action_std)[0].tolist()

        train_files, val_files = [], []
        rng = np.random.RandomState(seed)
        n_eps = len(npz_files)
        n_val = max(1, int(n_eps * val_ratio))
        val_indices = set(rng.choice(n_eps, n_val, replace=False).tolist())
        for i, f in enumerate(npz_files):
            if i in val_indices:
                val_files.append(f)
            else:
                train_files.append(f)

        train_ds = cls(
            split="train", val_ratio=val_ratio, seed=seed,
            action_mean=action_mean, action_std=action_std,
            obs_key=obs_key, episode_files=train_files, skip_split=True,
            condition_on_phase=condition_on_phase, phase_key=phase_key,
            near_zero_std_dims=near_zero_dims, action_std_raw=action_std_raw,
        )
        val_ds = cls(
            split="val", val_ratio=val_ratio, seed=seed,
            action_mean=action_mean, action_std=action_std,
            obs_key=obs_key, episode_files=val_files, skip_split=True,
            condition_on_phase=condition_on_phase, phase_key=phase_key,
            near_zero_std_dims=near_zero_dims, action_std_raw=action_std_raw,
        )
        return train_ds, val_ds


def _compute_episode_boundaries(episode_id: np.ndarray) -> list[tuple[int, int]]:
    """Find (start, end) indices for each episode from the episode_id array."""
    episodes: list[tuple[int, int]] = []
    if len(episode_id) == 0:
        return episodes
    current = 0
    for i in range(1, len(episode_id)):
        if episode_id[i] != episode_id[i - 1]:
            episodes.append((current, i))
            current = i
    episodes.append((current, len(episode_id)))
    return episodes
