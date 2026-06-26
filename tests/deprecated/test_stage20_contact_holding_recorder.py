"""Stage 20-R: Tests for contact/holding signal recording in DemonstrationRecorder.

Tests:
- Fake trajectory with gripper_has_object is saved/loaded via pop_last_episode_data.
- Old trajectory without contact fields still loads (backward compatibility).
- contact_signal_available flag is set correctly.
- No phase proxy labels are fabricated.
- save() method includes contact/holding fields.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder


def _make_fake_steps(
    recorder: DemonstrationRecorder,
    n_steps: int = 20,
    include_gripper_has_object: bool = True,
):
    """Record fake steps with optional gripper_has_object in obs[4]."""
    for i in range(n_steps):
        obs = np.zeros(11, dtype=np.float32)
        obs[0] = 0.5 + 0.01 * i  # eef_x
        obs[1] = 0.0
        obs[2] = 0.8
        obs[3] = 0.05  # gripper_open
        if include_gripper_has_object:
            # Simulate grasping: partially closed = object between fingers
            if 10 <= i < 18:
                obs[3] = 0.015  # gripper partially closed
                obs[4] = 1.0  # has_object = True
            else:
                obs[4] = 0.0  # has_object = False
        else:
            obs[4] = 0.0

        obs[5] = 0.06  # cube1_x
        obs[6] = 0.03  # cube1_y
        obs[7] = 0.83  # cube1_z
        obs[8] = -0.04  # cube2_x
        obs[9] = 0.00  # cube2_y
        obs[10] = 0.835  # cube2_z

        action = np.zeros(7, dtype=np.float32)
        action[0] = 0.01
        action[6] = -1.0  # gripper close

        recorder.record_step(
            obs=obs,
            filtered_obs=obs,
            action=action,
            done=(i == n_steps - 1),
            skill_id=0,
            step_id=i,
            success=True,
            phase_id=0,
        )
    recorder.end_episode()


class TestContactHoldingRecorder:

    def test_pop_last_episode_includes_gripper_has_object(self):
        """pop_last_episode_data includes gripper_has_object when obs has index 4."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=20, include_gripper_has_object=True)
        data = rec.pop_last_episode_data()
        assert data is not None

        # Core fields present
        assert "gripper_has_object" in data
        assert "contact_signal" in data
        assert "holding_signal" in data

        # Type check
        gh = data["gripper_has_object"]
        assert isinstance(gh, np.ndarray)
        assert gh.dtype == np.float32
        assert len(gh) == 20

        # Values: should be 1.0 during grasp steps (10-17), 0.0 otherwise
        assert gh[0] == 0.0
        assert gh[10] == 1.0
        assert gh[17] == 1.0
        assert gh[18] == 0.0

        # Contact and holding are copies of gripper_has_object
        np.testing.assert_array_equal(data["contact_signal"], gh)
        np.testing.assert_array_equal(data["holding_signal"], gh)

    def test_contact_signal_available_flag(self):
        """contact_signal_available_flag is True when obs has index 4."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=20, include_gripper_has_object=True)
        data = rec.pop_last_episode_data()
        assert data is not None
        assert data["contact_signal_available_flag"] == True
        assert data["holding_signal_available_flag"] == True

    def test_no_fabricated_per_object_fields(self):
        """No fabricated top_in_gripper / bottom_in_gripper per-step arrays."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=20, include_gripper_has_object=True)
        data = rec.pop_last_episode_data()
        assert data is not None
        # These should NOT be present as per-step arrays (unavailable)
        assert "top_in_gripper" not in data
        assert "bottom_in_gripper" not in data
        assert "holding_top_object" not in data
        assert "holding_bottom_object" not in data

    def test_save_includes_contact_fields(self):
        """save() method includes contact/holding fields in NPZ."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=5, include_gripper_has_object=True)
        rec.end_episode()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            npz_path, meta_path = rec.save(
                filename="test.npz",
                task="stacking",
                n_blocks=2,
                env_type="robosuite",
            )
            # Actually save to tmpdir
            import shutil
            npz_dest = out / "test.npz"
            shutil.copy2(npz_path, npz_dest)

            loaded = np.load(npz_dest, allow_pickle=True)
            assert "gripper_has_object" in loaded
            assert "contact_signal" in loaded
            assert "holding_signal" in loaded
            assert loaded["contact_signal_available_flag"] == True

    def test_short_obs_no_contact(self):
        """obs with < 5 dims does not produce contact fields but still works."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=3,
            action_dim=7,
        )
        # Record with 3-dim obs (no gripper_has_object)
        for i in range(5):
            obs = np.array([0.5, 0.0, 0.8], dtype=np.float32)
            action = np.zeros(7, dtype=np.float32)
            rec.record_step(
                obs=obs, filtered_obs=obs, action=action,
                done=(i == 4), skill_id=0, step_id=i, success=True, phase_id=0,
            )
        rec.end_episode()
        data = rec.pop_last_episode_data()
        assert data is not None
        # Core fields still present
        assert "filtered_obs" in data
        # Contact fields NOT present as arrays (not enough dims)
        assert "gripper_has_object" not in data
        # Flags indicate unavailable
        assert data["contact_signal_available_flag"] == False
        assert data["holding_signal_available_flag"] == False

    def test_old_trajectory_no_contact_still_works(self):
        """Dataset with old-format trajectory (no contact fields) still loads
        the core obs/action/phase fields correctly."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=10, include_gripper_has_object=True)
        data = rec.pop_last_episode_data()
        assert data is not None

        # Simulate an "old" dataset: drop all contact fields
        old_data = {
            k: v for k, v in data.items()
            if k not in ("gripper_has_object", "contact_signal", "holding_signal",
                         "contact_signal_available_flag", "holding_signal_available_flag")
        }

        # Save and reload as old format
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            npz_path = out / "old_traj.npz"
            np.savez_compressed(npz_path, **old_data)
            loaded = np.load(npz_path, allow_pickle=True)

            # Core fields still loadable
            assert "filtered_obs" in loaded
            assert "action" in loaded
            assert loaded["filtered_obs"].shape == (10, 11)

            # Contact fields absent (old format, no crash)
            assert "gripper_has_object" not in loaded

    def test_clean_state_after_pop(self):
        """After pop_last_episode_data, buffers are clean for next episode."""
        rec = DemonstrationRecorder(
            output_dir=tempfile.mkdtemp(),
            obs_dim=11,
            action_dim=7,
        )
        _make_fake_steps(rec, n_steps=5, include_gripper_has_object=True)
        assert rec.total_steps == 5
        data = rec.pop_last_episode_data()
        assert data is not None
        assert rec.total_steps == 0  # buffers cleared

        # Record another episode
        _make_fake_steps(rec, n_steps=3, include_gripper_has_object=True)
        assert rec.total_steps == 3
        data2 = rec.pop_last_episode_data()
        assert data2 is not None
        assert len(data2["gripper_has_object"]) == 3
