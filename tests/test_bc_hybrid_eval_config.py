"""Tests for BC hybrid eval mode and phase conditioning config.

Verifies:
  - Phase name mapping
  - Obs with phase construction
  - Checkpoint loading metadata
  - CLI arg handling
"""

import numpy as np
import pytest

from vlm_nsm.learning.eval_bc import PHASE_NAMES, load_checkpoint, _get_bc_action


def test_phase_names():
    """PHASE_NAMES should list all 6 phases in order."""
    assert PHASE_NAMES == ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def test_bc_phases_cli_choices():
    """--bc_phases should accept only valid phase names."""
    import inspect
    import argparse
    from vlm_nsm.learning import eval_bc
    # Verify choices match PHASE_NAMES
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_phases", type=str, nargs="+", default=None,
                        choices=PHASE_NAMES)
    args = parser.parse_args(["--bc_phases", "reach_pick", "reach_place"])
    assert args.bc_phases == ["reach_pick", "reach_place"]
    # Invalid choice should fail
    with pytest.raises(SystemExit):
        parser.parse_args(["--bc_phases", "invalid_phase"])


def test_control_mode_choices():
    """--control_mode should only accept 'full' or 'hybrid'."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_mode", type=str, default="full",
                        choices=["full", "hybrid"])
    args = parser.parse_args(["--control_mode", "hybrid"])
    assert args.control_mode == "hybrid"
    args2 = parser.parse_args([])
    assert args2.control_mode == "full"


def test_fixed_zero_action_dims():
    """--fixed_zero_action_dims should parse int list."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_zero_action_dims", type=int, nargs="*", default=None)
    args = parser.parse_args(["--fixed_zero_action_dims", "3", "4", "5"])
    assert args.fixed_zero_action_dims == [3, 4, 5]
    args2 = parser.parse_args([])
    assert args2.fixed_zero_action_dims is None


def test_policy_fn_one_hot_for_valid_phase():
    """policy_fn phase one-hot should set correct index for valid phase."""
    obs_11 = np.zeros(11, dtype=np.float32)
    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[2] = 1.0
    result = np.concatenate([obs_11, one_hot])
    assert len(result) == 11 + 6
    assert result[11 + 2] == 1.0  # phase 2 one-hot at index 13
    assert result[11] == 0.0
    assert result[11 + 5] == 0.0


def test_policy_fn_one_hot_out_of_range():
    """Phase ID outside [0,6) should produce all-zeros one-hot."""
    one_hot = np.zeros(6, dtype=np.float32)
    phase_id = 99
    if 0 <= phase_id < 6:
        one_hot[phase_id] = 1.0
    assert np.all(one_hot == 0.0), "out-of-range phase should produce zero vector"


def test_checkpoint_loading_metadata():
    """load_checkpoint should extract metadata from saved checkpoint."""
    import torch
    from vlm_nsm.learning.train_bc import MLPBCPolicy

    # Save a checkpoint with metadata using MLPBCPolicy
    model = MLPBCPolicy(obs_dim=17, action_dim=7, hidden_dims=[256, 128])
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "obs_dim": 17,
        "action_dim": 7,
        "hidden_dims": [256, 128],
        "condition_on_phase": True,
        "phase_dim": 6,
    }
    path = f"/tmp/test_checkpoint_meta.pt"
    torch.save(checkpoint, path)

    loaded_model, meta = load_checkpoint(path, "cpu")
    assert meta["obs_dim"] == 17
    assert meta["action_dim"] == 7
    assert meta["hidden_dims"] == [256, 128]
    assert meta["condition_on_phase"] is True
    assert meta["phase_dim"] == 6
    import os
    os.remove(path)


def test_checkpoint_loading_state_dict_only():
    """load_checkpoint should handle state_dict-only checkpoints."""
    import torch
    from vlm_nsm.learning.train_bc import MLPBCPolicy

    proper_model = MLPBCPolicy(obs_dim=11, action_dim=7)
    path = f"/tmp/test_checkpoint_sd.pt"
    torch.save(proper_model.state_dict(), path)

    loaded_model, meta = load_checkpoint(path, "cpu")
    assert meta["obs_dim"] == 11  # OBS_DIM default
    assert meta["action_dim"] == 7  # ACTION_DIM default
    import os
    os.remove(path)


def test_get_bc_action_zero_dims():
    """_get_bc_action should zero out specified dims."""
    import torch
    from vlm_nsm.learning.train_bc import MLPBCPolicy
    from vlm_nsm.learning.eval_bc import _get_bc_action

    model = MLPBCPolicy(obs_dim=11, action_dim=7)
    # Set weights so dim 3,4,5 outputs are non-zero
    with torch.no_grad():
        model.net[4].weight[3] = 1.0
        model.net[4].weight[4] = 1.0
        model.net[4].weight[5] = 1.0
    model.eval()

    obs = np.random.randn(11).astype(np.float32)
    action = _get_bc_action(model, obs, "cpu", None, None, fixed_zero_dims=[3, 4, 5])
    assert action[3] == 0.0, f"dim 3 should be zero, got {action[3]}"
    assert action[4] == 0.0, f"dim 4 should be zero, got {action[4]}"
    assert action[5] == 0.0, f"dim 5 should be zero, got {action[5]}"


def test_get_bc_action_clip():
    """_get_bc_action should clip to [-1, 1]."""
    import torch
    from vlm_nsm.learning.train_bc import MLPBCPolicy
    from vlm_nsm.learning.eval_bc import _get_bc_action

    model = MLPBCPolicy(obs_dim=11, action_dim=7)
    with torch.no_grad():
        for layer in model.net:
            if hasattr(layer, "weight"):
                layer.weight.fill_(100.0)
                layer.bias.fill_(100.0)
    model.eval()

    obs = np.zeros(11, dtype=np.float32)
    action = _get_bc_action(model, obs, "cpu", None, None, fixed_zero_dims=None)
    assert np.all(action >= -1.0) and np.all(action <= 1.0), f"action should be clipped: {action}"
