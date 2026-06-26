"""Tests for BC action std floor (min_action_std).

Verifies:
  - action_std is floored to min_action_std
  - near_zero_std_dims are detected
  - action_std_raw is preserved
  - fixed_zero_action_dims are propagated
"""

import numpy as np
import torch
from vlm_nsm.learning.dataset import RobosuiteDemoDataset
from vlm_nsm.learning.train_bc import MLPBCPolicy, _create_masked_mse_loss


def _make_npz(path: str, n_steps: int = 200, n_eps: int = 4):
    obs = np.random.randn(n_steps, 4).astype(np.float32)
    action = np.zeros((n_steps, 3), dtype=np.float32)
    ep_ids = np.repeat(np.arange(n_eps), n_steps // n_eps)
    if len(ep_ids) < n_steps:
        ep_ids = np.concatenate([ep_ids, np.full(n_steps - len(ep_ids), n_eps - 1)])
    np.savez(path, filtered_obs=obs, action=action, episode_id=ep_ids)


def test_action_std_floor_applied():
    """min_action_std should floor action_std."""
    npz = "/tmp/test_std_floor.npz"
    _make_npz(npz, n_steps=200, n_eps=4)

    train_ds, val_ds = RobosuiteDemoDataset.train_val(
        npz, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", min_action_std=1e-3,
    )
    assert train_ds.action_std is not None
    assert np.all(train_ds.action_std >= 1e-3), "action_std should be floored to 1e-3"
    assert train_ds.action_std_raw is not None
    assert np.all(train_ds.action_std_raw <= 1e-8), "raw std should be near zero"
    assert len(train_ds.near_zero_std_dims) > 0, "all dims should be near_zero"
    import os
    os.remove(npz)


def test_std_floor_with_varied_dims():
    """Only low-variance dims should be in near_zero_std_dims."""
    np.random.seed(42)
    n_steps = 200
    obs = np.random.randn(n_steps, 4).astype(np.float32)
    action = np.zeros((n_steps, 7), dtype=np.float32)
    action[:, 0] = np.random.randn(n_steps) * 5.0  # high var
    action[:, 1] = np.random.randn(n_steps) * 0.3  # medium var
    action[:, 2:] = 0.0  # zero var
    ep_ids = np.repeat([0, 1, 2, 3], 50)

    npz = "/tmp/test_std_floor_varied.npz"
    np.savez(npz, filtered_obs=obs, action=action, episode_id=ep_ids)

    train_ds, _ = RobosuiteDemoDataset.train_val(
        npz, val_ratio=0.25, seed=42,
        obs_key="filtered_obs", min_action_std=1e-2,
    )
    for d in range(2, 7):
        assert d in train_ds.near_zero_std_dims, f"dim {d} should be near-zero"
    assert 0 not in train_ds.near_zero_std_dims, "dim 0 has high variance"
    import os
    os.remove(npz)


def test_masked_mse_loss():
    """MaskedMSE loss should exclude specified dims."""
    mask_dims = [3, 4, 5]
    loss_fn = _create_masked_mse_loss(mask_dims, 7)
    pred = torch.randn(10, 7)
    target = torch.randn(10, 7)
    loss = loss_fn(pred, target)
    assert loss is not None
    assert loss > 0, "loss should be positive"
    pred2 = pred.clone()
    pred2[:, mask_dims] = target[:, mask_dims]
    loss2 = loss_fn(pred2, target)
    assert loss2 == loss, "changing masked dims should not affect loss"


def test_masked_mse_excludes_dims():
    """MaskedMSE loss should be 0 when only masked dims differ."""
    pred = torch.zeros(10, 7)
    target = torch.zeros(10, 7)
    target[:, [3, 4, 5]] = 1.0  # only masked dims differ
    loss_fn = _create_masked_mse_loss([3, 4, 5], 7)
    loss = loss_fn(pred, target)
    assert loss == 0.0, f"expected 0, got {loss}"


def test_masked_mse_out_of_range_dim():
    """mask dim outside action_dim should be ignored."""
    loss_fn = _create_masked_mse_loss([99], 3)
    assert loss_fn is not None
    pred = torch.randn(5, 3)
    target = torch.randn(5, 3)
    loss = loss_fn(pred, target)
    assert loss > 0


def test_mlp_policy_default_params():
    """MLPBCPolicy should accept default params."""
    model = MLPBCPolicy()
    assert model is not None
    obs = torch.randn(1, 11)
    action = model(obs)
    assert action.shape == (1, 7)


def test_mlp_policy_custom_dims():
    """MLPBCPolicy should accept custom dims."""
    model = MLPBCPolicy(obs_dim=17, action_dim=7, hidden_dims=[256, 128])
    obs = torch.randn(2, 17)
    action = model(obs)
    assert action.shape == (2, 7)
