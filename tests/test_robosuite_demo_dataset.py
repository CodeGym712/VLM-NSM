"""Tests for RobosuiteDemoDataset (robosuite-independent).

Verifies:
  - Loading from npz with correct shapes
  - Episode boundary detection
  - Train/val split without data leakage
  - Action normalization from training set only
  - get_episode() returns full episode data
  - build_robosuite_obs_vector utility
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vlm_nsm.learning.dataset import (
    RobosuiteDemoDataset,
    build_robosuite_obs_vector,
    OBS_DIM,
    ACTION_DIM,
    _compute_episode_boundaries,
)


# ---- Fixtures ----


@pytest.fixture
def demo_npz():
    """Create a synthetic demonstration npz with 5 episodes, 100 steps each."""
    n_eps = 5
    steps_per_ep = 100
    T = n_eps * steps_per_ep

    obs = np.random.randn(T, OBS_DIM).astype(np.float32)
    action = np.random.randn(T, ACTION_DIM).astype(np.float32)
    episode_id = np.repeat(np.arange(n_eps), steps_per_ep).astype(np.int32)
    success = np.ones(T, dtype=bool)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez_compressed(f, obs=obs, action=action,
                            episode_id=episode_id, success=success)
        tmp_path = Path(f.name)

    yield tmp_path
    tmp_path.unlink(missing_ok=True)


# ---- _compute_episode_boundaries ----


class TestComputeEpisodeBoundaries:
    def test_basic(self):
        eps = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
        boundaries = _compute_episode_boundaries(eps)
        assert boundaries == [(0, 3), (3, 5), (5, 8)]

    def test_single_episode(self):
        eps = np.zeros(50, dtype=np.int32)
        boundaries = _compute_episode_boundaries(eps)
        assert boundaries == [(0, 50)]

    def test_empty(self):
        eps = np.array([], dtype=np.int32)
        boundaries = _compute_episode_boundaries(eps)
        assert boundaries == []


# ---- RobosuiteDemoDataset ----


class TestRobosuiteDemoDataset:
    def test_load_shapes(self, demo_npz):
        ds = RobosuiteDemoDataset(str(demo_npz), split="train")
        assert ds.obs_dim == OBS_DIM
        assert ds.action_dim == ACTION_DIM

    def test_train_val_split(self, demo_npz):
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            str(demo_npz), val_ratio=0.2, seed=42,
        )
        # 5 episodes * 0.2 = 1 val, 4 train
        assert train_ds.num_episodes == 4
        assert val_ds.num_episodes == 1
        assert len(train_ds) + len(val_ds) == 500  # total steps

    def test_train_val_no_overlap(self, demo_npz):
        """Verify train and val indices are disjoint."""
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            str(demo_npz), val_ratio=0.2, seed=42,
        )
        train_eps = set(train_ds._episode_indices)
        val_eps = set(val_ds._episode_indices)
        assert train_eps.isdisjoint(val_eps)

    def test_getitem_shapes(self, demo_npz):
        ds = RobosuiteDemoDataset(str(demo_npz), split="train")
        item = ds[0]
        assert isinstance(item, dict)
        assert item["obs"].shape == (OBS_DIM,)
        assert item["action"].shape == (ACTION_DIM,)
        assert isinstance(item["obs"], torch.Tensor)
        assert item["obs"].dtype == torch.float32

    def test_get_episode(self, demo_npz):
        ds = RobosuiteDemoDataset(str(demo_npz), split="train")
        ep = ds.get_episode(0)
        assert ep["obs"].dim() == 2  # (T, obs_dim)
        assert ep["obs"].shape[-1] == OBS_DIM
        assert ep["action"].shape[-1] == ACTION_DIM

    def test_normalization_computed_from_train_only(self, demo_npz):
        """Action normalization stats should come from train set only."""
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            str(demo_npz), val_ratio=0.2, seed=42,
        )
        # Both should have the same normalization (computed from train set)
        assert train_ds.action_mean is not None
        assert train_ds.action_std is not None
        assert np.allclose(train_ds.action_mean, val_ds.action_mean)
        assert np.allclose(train_ds.action_std, val_ds.action_std)

    def test_normalization_applied(self, demo_npz):
        train_ds, _ = RobosuiteDemoDataset.train_val(
            str(demo_npz), val_ratio=0.2, seed=42,
        )
        # After normalization, action should have approx 0 mean, 1 std
        normalized_actions = []
        for i in range(len(train_ds)):
            normalized_actions.append(train_ds[i]["action"].numpy())
        all_actions = np.stack(normalized_actions)
        assert np.allclose(np.mean(all_actions, axis=0), 0, atol=0.5)
        assert np.allclose(np.std(all_actions, axis=0), 1, atol=0.5)

    def test_dataloader_compatible(self, demo_npz):
        ds = RobosuiteDemoDataset(str(demo_npz), split="train")
        loader = DataLoader(ds, batch_size=16, shuffle=True)
        batch = next(iter(loader))
        assert batch["obs"].shape == (16, OBS_DIM)
        assert batch["action"].shape == (16, ACTION_DIM)


# ---- build_robosuite_obs_vector ----


class TestBuildObsVector:
    def test_basic(self):
        eef = {"x": 0.5, "y": 0.1, "z": 0.05}
        gripper = {"open": -0.5, "has_object": True}
        poses = {
            "cube1": {"x": 0.4, "y": 0.2, "z": 0.03},
            "cube2": {"x": 0.3, "y": 0.3, "z": 0.8},
            "table": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        obs = build_robosuite_obs_vector(eef, gripper, poses)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        # Check specific values (use np.isclose due to float32 precision)
        assert np.isclose(obs[0], 0.5), f"eef_x: {obs[0]}"
        assert np.isclose(obs[3], -0.5), f"gripper_open: {obs[3]}"
        assert obs[4] == 1.0, f"has_object: {obs[4]}"
        assert np.isclose(obs[5], 0.4), f"cube1_x: {obs[5]}"

    def test_missing_cube_defaults_to_zero(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        gripper = {"open": -1.0, "has_object": False}
        poses = {"table": {"x": 0.0, "y": 0.0, "z": 0.0}}
        obs = build_robosuite_obs_vector(eef, gripper, poses,
                                          cubes=["cube1", "cube2"])
        assert obs.shape == (OBS_DIM,)
        assert obs[5] == 0.0  # cube1_x defaults to 0
        assert obs[8] == 0.0  # cube2_x defaults to 0

    def test_custom_cubes(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        gripper = {"open": -1.0, "has_object": False}
        poses = {"a": {"x": 1.0, "y": 2.0, "z": 3.0}}
        obs = build_robosuite_obs_vector(eef, gripper, poses, cubes=["a"])
        assert obs.shape == (5 + 3,)  # 5 base + 3 for 1 cube
        assert obs[5] == 1.0


# ---- Module constants ----


def test_constants():
    assert OBS_DIM == 11
    assert ACTION_DIM == 7
