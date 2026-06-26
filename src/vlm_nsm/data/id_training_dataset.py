"""Inverse Dynamics Training Dataset.

Extracts (state_t, state_{t+1}) → action_t pairs from HDF5 episode data
for training a lightweight inverse dynamics model.

Design:
  - Input:  [eef_pos(3), gripper(1), object_poses(N*3), phase_onehot(10)]
  - Target: action(4) = [dx, dy, dz, gripper]
  - Only phases 0-3 (reach_pick, grasp, lift, reach_place) are available
    in existing HDF5 data. Phases 4-5 (release, retreat) are absent.
  - ~950 usable episodes, ~950K training samples.

Usage:
    from vlm_nsm.data.id_training_dataset import build_id_dataset

    X, Y = build_id_dataset("generated/datasets/vla_demos/d1_small_200ep")
    # X: (N, 26), Y: (N, 4) as numpy arrays
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# Phase ID mapping (6-phase vocabulary used in HDF5)
_PHASE_MAP = {
    0: "reach_pick",
    1: "grasp",
    2: "lift",
    3: "reach_place",
    4: "release",
    5: "retreat",
}
_N_PHASES_ID = 10  # padded to 10 for encoder compatibility
_N_OBJECTS = 3     # cube1, cube2, cube3 (pad to 3 for generality)

# Input dimension: eef(3) + gripper(1) + obj_poses(9) + phase_onehot(10) + desired_next_eef(3) = 26
_INPUT_DIM = 28  # eef(3)+gripper(1)+vel_hist(9)+act_hist(12)+desired(3), NO phase
_OUTPUT_DIM = 4
_HISTORY_LEN = 3


def _get_object_poses(h5: h5py.File, step: int, cubes: list[str]) -> np.ndarray:
    """Extract object positions at a given step.

    Since HDF5 does not directly store object poses, we estimate them
    from EEF position and action. For a more accurate dataset, the
    object poses should be recorded at collection time.

    Returns:
        (9,) array [obj1_x, obj1_y, obj1_z, obj2_x, ..., obj3_z],
        zero-padded for missing objects.
    """
    # HDF5 doesn't have per-step object poses directly.
    # We use the EEF position as a proxy for the held object position
    # and estimate target object position from action context.
    # For ID training, the key signal is EEF state → EEF action,
    # which is fully captured in the HDF5 proprio/actions fields.
    eef = h5["proprio/eef_pos"][step]  # (3,)
    poses = np.zeros(_N_OBJECTS * 3, dtype=np.float32)

    # Fill available object poses from eef (proxy for held object)
    # and pad remaining
    for i in range(min(len(cubes), _N_OBJECTS)):
        # Use EEF as coarse proxy; real object tracking would need env.get_object_pose
        poses[i * 3:(i + 1) * 3] = eef if i == 0 else np.zeros(3, dtype=np.float32)

    return poses


def build_id_dataset(
    dataset_dir: str | Path,
    *,
    success_only: bool = True,
    max_episodes: int | None = None,
    phases: tuple[int, ...] = (0, 1, 2, 3),
    stride: int = 1,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Build inverse dynamics training dataset from HDF5 episodes.

    Args:
        dataset_dir: Path to dataset root (contains episodes/ + manifest.jsonl).
        success_only: If True, use only successful episodes.
        max_episodes: Maximum number of episodes to use (for quick testing).
        phases: Tuple of phase IDs to include (default: 0-3).
        stride: Sampling stride (1 = every consecutive pair).
        val_split: Fraction of data for validation.
        seed: Random seed for train/val split.

    Returns:
        Dict with keys:
          - train_X, train_Y: training arrays
          - val_X, val_Y: validation arrays
          - input_dim, output_dim: dimensions
          - n_train, n_val: sample counts
          - stats: dict with action_mean, action_std
    """
    root = Path(dataset_dir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f]

    if success_only:
        episodes = [ep for ep in episodes if ep.get("success", True)]

    if max_episodes:
        episodes = episodes[:max_episodes]

    rng = np.random.RandomState(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * val_split))
    val_episodes = episodes[:n_val]
    train_episodes = episodes[n_val:]

    print(f"[ID data] {len(train_episodes)} train + {len(val_episodes)} val episodes")

    train_X, train_Y = _extract_samples(train_episodes, phases, stride, rng)
    val_X, val_Y = _extract_samples(val_episodes, phases, stride, rng)

    # Compute normalization stats from training data
    action_mean = train_Y.mean(axis=0).astype(np.float32)
    action_std = train_Y.std(axis=0).astype(np.float32)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)

    # Normalize targets
    train_Y_norm = (train_Y - action_mean) / action_std
    val_Y_norm = (val_Y - action_mean) / action_std

    print(f"[ID data] train: {len(train_X)} samples, val: {len(val_X)} samples")
    print(f"[ID data] action_mean={action_mean}, action_std={action_std}")

    return {
        "train_X": train_X,
        "train_Y": train_Y_norm,
        "val_X": val_X,
        "val_Y": val_Y_norm,
        "input_dim": _INPUT_DIM,
        "output_dim": _OUTPUT_DIM,
        "n_train": len(train_X),
        "n_val": len(val_X),
        "stats": {
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
        },
    }


def _extract_samples(
    episodes: list[dict],
    phases: tuple[int, ...],
    stride: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, Y) pairs from episode list with velocity/action history."""
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []

    phase_set = set(phases)

    for ep in episodes:
        path = ep.get("episode_path", "")
        if not path or not Path(path).exists():
            continue

        try:
            with h5py.File(path, "r") as h5:
                n_steps = h5["actions_4d"].shape[0]
                if n_steps < _HISTORY_LEN + 1:
                    continue

                actions = h5["actions_4d"][:]  # (T, 4)
                eef_pos = h5["proprio/eef_pos"][:]  # (T, 3)
                gripper = h5["proprio/gripper_open"][:]  # (T, 1)
                phase_ids = h5["symbolic/phase_id"][:]  # (T,)

                # Compute velocities via finite difference
                velocities = np.zeros_like(eef_pos, dtype=np.float32)
                velocities[1:] = (eef_pos[1:] - eef_pos[:-1]) / 0.02

                for t in range(_HISTORY_LEN, n_steps - 1, stride):
                    pid = int(phase_ids[t])
                    if pid not in phase_set:
                        continue

                    # Velocity history: last 3 frames
                    vel_hist = velocities[t - _HISTORY_LEN + 1:t + 1].flatten()
                    # Action history: last 3 frames (FIX: exclude current step)
                    act_hist = actions[t - _HISTORY_LEN:t].flatten()

                    x = _build_input_vector(
                        eef_pos[t], gripper[t], pid,
                        eef_pos[t + 1], vel_hist, act_hist,
                    )
                    y = actions[t]

                    X_list.append(x)
                    Y_list.append(y)

        except (OSError, KeyError, ValueError) as e:
            print(f"[ID data] WARNING: skipping {path}: {e}")
            continue

    if not X_list:
        return (np.zeros((0, _INPUT_DIM), dtype=np.float32),
                np.zeros((0, _OUTPUT_DIM), dtype=np.float32))

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    return X, Y


def _build_input_vector(
    eef: np.ndarray,
    gripper_val: np.ndarray,
    phase_id: int,  # kept for API compat, NOT used in vector
    desired_next_eef: np.ndarray,
    vel_history: np.ndarray | None = None,
    act_history: np.ndarray | None = None,
) -> np.ndarray:
    """Build 28-dim input vector for ID model (no phase_onehot).

    Layout:
      [0:3]   eef_pos (3)
      [3]     gripper_open (1)
      [4:13]  velocity_history (9) — 3 frames
      [13:25] action_history (12) — 3 frames
      [25:28] desired_next_eef (3)
    """
    vec = np.zeros(_INPUT_DIM, dtype=np.float32)
    vec[0:3] = eef
    vec[3] = float(gripper_val) if hasattr(gripper_val, '__len__') else gripper_val
    if vel_history is not None:
        n_v = min(len(vel_history), 9)
        vec[4:4 + n_v] = vel_history[:n_v]
    if act_history is not None:
        n_a = min(len(act_history), 12)
        vec[13:13 + n_a] = act_history[:n_a]
    vec[25:28] = desired_next_eef
    return vec


def save_id_dataset(dataset: dict, output_dir: str | Path):
    """Save ID dataset to disk as .npz files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out / "train.npz", X=dataset["train_X"], Y=dataset["train_Y"])
    np.savez_compressed(out / "val.npz", X=dataset["val_X"], Y=dataset["val_Y"])

    meta = {
        "input_dim": dataset["input_dim"],
        "output_dim": dataset["output_dim"],
        "n_train": dataset["n_train"],
        "n_val": dataset["n_val"],
        "stats": dataset["stats"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ID data] Saved to {out}")
