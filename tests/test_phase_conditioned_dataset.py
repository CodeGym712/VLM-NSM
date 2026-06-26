"""Tests for phase-conditioned dataset.

Verifies:
  - phase_id loading from npz with condition_on_phase=True
  - one-hot encoding is appended to obs
  - obs_dim_with_phase is correct
  - error when phase_key missing
  - directory mode with phase conditioning
"""

import numpy as np
import pytest
import torch


def _make_npz(path: str, n_steps: int = 100, n_eps: int = 4, include_phase: bool = True):
    obs = np.random.randn(n_steps, 4).astype(np.float32)
    action = np.random.randn(n_steps, 3).astype(np.float32)
    ep_ids = np.repeat(np.arange(n_eps), n_steps // n_eps)
    if len(ep_ids) < n_steps:
        ep_ids = np.concatenate([ep_ids, np.full(n_steps - len(ep_ids), n_eps - 1)])
    kwargs = {"filtered_obs": obs, "action": action, "episode_id": ep_ids}
    if include_phase:
        kwargs["phase_id"] = np.random.randint(0, 6, size=n_steps).astype(np.int64)
    np.savez(path, **kwargs)


def test_phase_conditioning_basic():
    """condition_on_phase=True should load phase_id and append one-hot."""
    path = "/tmp/test_phase_basic.npz"
    _make_npz(path, n_steps=80, n_eps=4)

    train_ds, val_ds = RobosuiteDemoDataset.train_val(
        path, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", condition_on_phase=True,
    )
    assert train_ds.condition_on_phase
    assert train_ds.phase_id is not None
    assert train_ds.obs_dim_with_phase == 4 + 6  # 4 obs + 6 phase one-hot
    sample = train_ds[0]
    assert sample["obs"].shape[0] == 4 + 6, f"expected 10, got {sample['obs'].shape}"
    import os
    os.remove(path)


def test_phase_conditioning_obs_dim():
    """obs_dim_with_phase should be obs_dim + n_phases."""
    path = "/tmp/test_phase_obs_dim.npz"
    _make_npz(path, n_steps=80, n_eps=4)

    train_ds, _ = RobosuiteDemoDataset.train_val(
        path, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", condition_on_phase=True,
    )
    assert train_ds.obs_dim == 4
    assert train_ds.obs_dim_with_phase == 10  # 4 + 6 phases

    train_ds2, _ = RobosuiteDemoDataset.train_val(
        path, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", condition_on_phase=False,
    )
    assert train_ds2.obs_dim_with_phase == 4
    import os
    os.remove(path)


def test_phase_one_hot_valid():
    """Phase one-hot should have exactly one non-zero entry."""
    path = "/tmp/test_phase_onehot.npz"
    _make_npz(path, n_steps=40, n_eps=4)

    train_ds, _ = RobosuiteDemoDataset.train_val(
        path, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", condition_on_phase=True,
    )
    sample = train_ds[0]
    obs = sample["obs"]
    phase_part = obs[4:]
    assert torch.sum(phase_part > 0) == 1, "one-hot should have exactly 1 non-zero"
    assert torch.sum(phase_part) == 1.0, "one-hot should sum to 1"
    import os
    os.remove(path)


def test_phase_key_missing_error():
    """condition_on_phase=True should raise error when phase_key is missing."""
    path = "/tmp/test_phase_missing.npz"
    _make_npz(path, include_phase=False)

    with pytest.raises(KeyError, match="condition_on_phase"):
        RobosuiteDemoDataset.train_val(
            path, val_ratio=0.25, seed=42,
            obs_key="filtered_obs", condition_on_phase=True,
        )
    import os
    os.remove(path)


def test_phase_in_directory_mode():
    """Directory mode should support condition_on_phase."""
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    for i in range(3):
        n_steps = 20 + i * 10
        obs = np.random.randn(n_steps, 4).astype(np.float32)
        action = np.random.randn(n_steps, 3).astype(np.float32)
        phase_id = np.zeros(n_steps, dtype=np.int64)
        phase_id[:n_steps // 2] = 0
        phase_id[n_steps // 2:] = 1
        np.savez(f"{tmpdir}/ep_{i}.npz", filtered_obs=obs, action=action, phase_id=phase_id)

    train_ds, val_ds = RobosuiteDemoDataset.from_directory(
        tmpdir, val_ratio=0.5, seed=42,
        obs_key="filtered_obs", condition_on_phase=True,
    )
    assert train_ds.condition_on_phase
    assert train_ds.phase_id is not None
    assert train_ds.obs_dim_with_phase == 10
    sample = train_ds[0]
    assert sample["obs"].shape[0] == 10
    shutil.rmtree(tmpdir)


def test_phase_get_episode():
    """get_episode should include phase one-hot when conditioned."""
    path = "/tmp/test_phase_get_ep.npz"
    _make_npz(path, n_steps=80, n_eps=4)

    train_ds, _ = RobosuiteDemoDataset.train_val(
        path, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", condition_on_phase=True,
    )
    ep = train_ds.get_episode(0)
    assert "obs" in ep
    assert ep["obs"].shape[1] == 10  # 4 + 6
    import os
    os.remove(path)


# Need this import for dataset
from vlm_nsm.learning.dataset import RobosuiteDemoDataset  # noqa: E402
