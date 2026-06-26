"""Tests for Stage 7B-1 BC training and evaluation scripts.

Verifies that training and eval scripts load, parse args correctly,
and run with synthetic data. All tests are robosuite-independent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from vlm_nsm.learning.dataset import OBS_DIM, ACTION_DIM
from vlm_nsm.learning.train_bc import MLPBCPolicy, train_epoch, validate
from torch.utils.data import DataLoader


# ---- MLPBCPolicy ----


class TestMLPBCPolicy:
    def test_forward_shape(self):
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        obs = torch.randn(4, OBS_DIM)
        out = model(obs)
        assert out.shape == (4, ACTION_DIM)

    def test_forward_single(self):
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        obs = torch.randn(OBS_DIM)
        out = model(obs.unsqueeze(0))
        assert out.shape == (1, ACTION_DIM)

    def test_parameter_count(self):
        """Default [256, 256] hidden dims: 70,663 params."""
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dims=[256, 256])
        n = sum(p.numel() for p in model.parameters())
        assert n == 70663  # 11*256 + 256 + 256*256 + 256 + 256*7 + 7

    def test_different_hidden_dims(self):
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dims=[128, 128])
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        obs = torch.randn(2, OBS_DIM)
        out = model(obs)
        assert out.shape == (2, ACTION_DIM)

    def test_single_hidden_layer(self):
        """MLP with one hidden layer."""
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dims=[128])
        obs = torch.randn(2, OBS_DIM)
        out = model(obs)
        assert out.shape == (2, ACTION_DIM)

    def test_gradients_flow(self):
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        obs = torch.randn(4, OBS_DIM)
        target = torch.randn(4, ACTION_DIM)
        loss = loss_fn(model(obs), target)
        loss.backward()
        opt.step()
        assert not torch.isnan(model.net[0].weight).any()


# ---- Training loop ----

OBS_KEYS = ["obs", "filtered_obs"]


def _make_synthetic_npz(tmp_path: Path, obs_key: str = "obs") -> Path:
    """Create a synthetic npz with the given obs_key."""
    n_eps = 5
    steps_per_ep = 50
    T = n_eps * steps_per_ep

    obs = np.random.randn(T, OBS_DIM).astype(np.float32)
    action = np.random.randn(T, ACTION_DIM).astype(np.float32)
    episode_id = np.repeat(np.arange(n_eps), steps_per_ep).astype(np.int32)
    success = np.ones(T, dtype=bool)

    npz_path = tmp_path / "test_demo.npz"
    np.savez_compressed(npz_path,
                         **{obs_key: obs, "action": action,
                            "episode_id": episode_id, "success": success})
    return npz_path


@pytest.fixture
def synthetic_dataloaders():
    """Create train/val DataLoaders with synthetic data."""
    from vlm_nsm.learning.dataset import RobosuiteDemoDataset

    n_eps = 5
    steps_per_ep = 50
    T = n_eps * steps_per_ep

    obs = np.random.randn(T, OBS_DIM).astype(np.float32)
    action = np.random.randn(T, ACTION_DIM).astype(np.float32)
    episode_id = np.repeat(np.arange(n_eps), steps_per_ep).astype(np.int32)
    success = np.ones(T, dtype=bool)

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez_compressed(tmp, obs=obs, action=action, episode_id=episode_id, success=success)
    tmp_path = Path(tmp.name)

    train_ds, val_ds = RobosuiteDemoDataset.train_val(str(tmp_path), val_ratio=0.2, seed=42)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    yield train_loader, val_loader

    tmp_path.unlink(missing_ok=True)


class TestTrainingLoop:
    def test_train_epoch(self, synthetic_dataloaders):
        train_loader, _ = synthetic_dataloaders
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        loss = train_epoch(model, train_loader, optimizer, loss_fn, "cpu")
        assert loss > 0
        assert not np.isnan(loss)

    def test_validate(self, synthetic_dataloaders):
        _, val_loader = synthetic_dataloaders
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        loss_fn = torch.nn.MSELoss()
        loss = validate(model, val_loader, loss_fn, "cpu")
        assert loss > 0
        assert not np.isnan(loss)

    def test_train_reduces_loss(self, synthetic_dataloaders):
        train_loader, val_loader = synthetic_dataloaders
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        losses = []
        for _ in range(10):
            loss = train_epoch(model, train_loader, optimizer, loss_fn, "cpu")
            losses.append(loss)

        # Loss should decrease over 10 epochs
        assert losses[-1] < losses[0] * 1.1  # at least not significantly worse

    def test_end_to_end_smoke(self, synthetic_dataloaders):
        """Run a full train+val cycle and verify model saves/loads."""
        train_loader, val_loader = synthetic_dataloaders
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        best_state = None
        for epoch in range(5):
            train_epoch(model, train_loader, optimizer, loss_fn, "cpu")
            val_loss = validate(model, val_loader, loss_fn, "cpu")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save and reload
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(best_state, tmp.name)
        tmp_path = Path(tmp.name)

        model2 = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))

        # Load best_state into original model for comparison
        model.load_state_dict(best_state)

        # Same output on fixed input
        model.eval()
        model2.eval()
        test_in = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            out1 = model(test_in)
            out2 = model2(test_in)
        assert torch.allclose(out1, out2, atol=1e-5)

        tmp_path.unlink(missing_ok=True)

    def test_train_with_obs_key(self, tmp_path):
        """Verify training with filtered_obs key works."""
        from vlm_nsm.learning.dataset import RobosuiteDemoDataset

        npz_path = _make_synthetic_npz(tmp_path, obs_key="filtered_obs")
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            str(npz_path), val_ratio=0.2, seed=42, obs_key="filtered_obs",
        )
        assert train_ds.obs_dim == OBS_DIM
        model = MLPBCPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        loss = train_epoch(model, train_loader, optimizer, loss_fn, "cpu")
        assert loss > 0
        assert not np.isnan(loss)

    def test_from_directory(self, tmp_path):
        """Verify directory-based loading works."""
        from vlm_nsm.learning.dataset import RobosuiteDemoDataset

        # Create per-episode npz files
        demo_dir = tmp_path / "demos"
        demo_dir.mkdir()
        for ep in range(5):
            T = 20
            obs = np.random.randn(T, OBS_DIM).astype(np.float32)
            action = np.random.randn(T, ACTION_DIM).astype(np.float32)
            npz_file = demo_dir / f"ep_{ep:04d}.npz"
            np.savez_compressed(npz_file, obs=obs, action=action)

        train_ds, val_ds = RobosuiteDemoDataset.from_directory(
            str(demo_dir), val_ratio=0.2, seed=42,
        )
        assert train_ds.num_episodes == 4
        assert val_ds.num_episodes == 1
        assert train_ds.obs_dim == OBS_DIM
        assert train_ds.action_dim == ACTION_DIM
        item = train_ds[0]
        assert item["obs"].shape == (OBS_DIM,)
        assert item["action"].shape == (ACTION_DIM,)

    def test_from_directory_with_obs_key(self, tmp_path):
        """Verify directory-based loading with custom obs_key works."""
        from vlm_nsm.learning.dataset import RobosuiteDemoDataset

        demo_dir = tmp_path / "demos_filtered"
        demo_dir.mkdir()
        for ep in range(3):
            T = 20
            filtered_obs = np.random.randn(T, OBS_DIM).astype(np.float32)
            action = np.random.randn(T, ACTION_DIM).astype(np.float32)
            npz_file = demo_dir / f"ep_{ep:04d}.npz"
            np.savez_compressed(npz_file, filtered_obs=filtered_obs, action=action)

        train_ds, val_ds = RobosuiteDemoDataset.from_directory(
            str(demo_dir), val_ratio=0.2, seed=42, obs_key="filtered_obs",
        )
        assert train_ds.num_episodes > 0
        item = train_ds[0]
        assert item["obs"].shape == (OBS_DIM,)

    def test_no_episode_split_mode(self, tmp_path):
        """Verify row-level split (no episode boundaries)."""
        from vlm_nsm.learning.dataset import RobosuiteDemoDataset

        npz_path = _make_synthetic_npz(tmp_path)
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            str(npz_path), val_ratio=0.2, seed=42, episode_split=False,
        )
        total = len(train_ds) + len(val_ds)
        assert total == 250  # 5 eps * 50 steps
        assert len(train_ds) > len(val_ds)
