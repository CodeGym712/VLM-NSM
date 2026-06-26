"""Stage 20-S: Tests for obs[4] fallback in dataset builder.

Tests:
- Builder uses explicit holding_signal when present.
- Builder falls back to filtered_obs[:,4] when explicit signal missing.
- Builder does not create labels from phase proxy.
- Old data without obs width >4 yields mask=0.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage20_build_role_conditioned_dataset import (
    build_role_conditioned_dataset,
)


def _make_fake_trajectory_npz(
    traj_dir: Path,
    episode_id: str,
    n_steps: int = 12,
    obs_width: int = 11,
    include_explicit_holding: bool = True,
    include_explicit_contact: bool = True,
    obs4_val: float = 1.0,
):
    """Create a minimal fake trajectory for testing builder fallback."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_name = f"{episode_id}.npz"

    obs = np.zeros((n_steps, obs_width), dtype=np.float32)
    obs[:, 0] = 0.5 + np.arange(n_steps) * 0.01
    if obs_width >= 4:
        obs[:, 3] = 0.05
    if obs_width > 4:
        obs[:, 4] = obs4_val  # gripper_has_object at index 4
    if obs_width > 5:
        obs[:, 5] = 0.06
    if obs_width > 6:
        obs[:, 6] = 0.03
    if obs_width > 7:
        obs[:, 7] = 0.83
    if obs_width > 8:
        obs[:, 8] = -0.04
    if obs_width > 9:
        obs[:, 9] = 0.00
    if obs_width > 10:
        obs[:, 10] = 0.835

    action = np.zeros((n_steps, 7), dtype=np.float32)
    action[:, 0] = 0.01

    phase_ids = np.zeros(n_steps, dtype=np.int64)
    phase_ids[4:8] = 1  # grasp
    phase_ids[8:10] = 2  # lift
    phase_ids[10:] = 3  # transport

    save_dict = {
        "filtered_obs": obs,
        "obs": obs,
        "action": action,
        "phase_id": phase_ids,
    }

    if include_explicit_holding:
        holding_vals = np.zeros(n_steps, dtype=np.float32)
        holding_vals[8:] = 1.0  # holding during lift+transport
        save_dict["holding_signal"] = holding_vals
        save_dict["holding_signal_available_flag"] = np.array(True)

    if include_explicit_contact:
        contact_vals = np.zeros(n_steps, dtype=np.float32)
        contact_vals[4:10] = 1.0
        save_dict["contact_signal"] = contact_vals
        save_dict["contact_signal_available_flag"] = np.array(True)

    npz_path = traj_dir / "trajectories" / traj_name
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **save_dict)

    record = {
        "episode_id": episode_id,
        "trajectory_path": f"trajectories/{traj_name}",
        "direction": "cube1_to_cube2",
        "object_id": "cube1",
        "target_object_id": "cube2",
        "success": True,
        "teacher_profile": "scripted_only",
    }
    manifest_path = traj_dir / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")


class TestHoldingObs4Fallback:

    def test_explicit_holding_used_when_present(self):
        """When holding_signal is explicit in NPZ, it is used (not fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "explicit_holding"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=True, obs4_val=0.0)

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            assert summary["holding_label_available_rate"] == 1.0
            # Source should be explicit, not obs4 fallback
            src = summary.get("holding_signal_source_counts", {})
            assert "explicit_holding_signal" in src
            assert src.get("obs4_fallback_holding_like", 0) == 0

    def test_obs4_fallback_when_no_explicit(self):
        """When no explicit holding_signal, obs[4] fallback is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "fallback_obs4"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=False,
                include_explicit_contact=False, obs4_val=1.0)

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            assert summary["holding_label_available_rate"] == 1.0
            src = summary.get("holding_signal_source_counts", {})
            assert "obs4_fallback_holding_like" in src
            assert src.get("obs4_fallback_holding_like", 0) > 0

    def test_no_labels_from_phase_proxy(self):
        """Builder does NOT fabricate holding labels from phase_id.
        With obs_width < 11, trajectory is skipped (no windows).
        This proves no phase-proxy fabrication — builder refuses bad data
        rather than generating fake labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "no_proxy"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=False,
                include_explicit_contact=False, obs_width=3, obs4_val=0.0)

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            assert summary["no_phase_proxy_labels"] is True
            # No windows from narrow obs — no labels fabricated
            assert summary["num_windows"] == 0

    def test_old_data_narrow_obs_masks_zero(self):
        """Old data with obs width < 11 is skipped gracefully, no crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "narrow_obs"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=False,
                include_explicit_contact=False, obs_width=4)

            result = build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            # Trajectory is skipped because obs.shape[1] < 11, no crash
            assert result["num_windows"] == 0
            assert result["num_trajectories_skipped"] == 1

    def test_true_mujoco_contact_caveat_present(self):
        """Summary clearly states no true MuJoCo contact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "caveat_test"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(traj_dir, "ep01")

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            assert summary["true_mujoco_contact_available"] is False
            assert summary["per_object_contact_available"] is False
            assert "holding_label_caveat" in summary
            assert "NOT true MuJoCo contact" in summary["holding_label_caveat"]

    def test_explicit_holding_beats_obs4(self):
        """When both explicit and obs4 are available, explicit is preferred."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "explicit_beats_obs4"
            out_dir = Path(tmpdir) / "output"
            # Create data with BOTH explicit holding_signal AND obs[4]=0
            # explicit has holding=1 during phase 2-3, obs4 has all zeros
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=True,
                obs4_val=0.0)  # obs[4] is all zeros, but explicit overrides

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            src = summary.get("holding_signal_source_counts", {})
            assert src.get("explicit_holding_signal", 0) > 0
            # obs4 fallback should NOT be used (explicit is preferred)
            assert src.get("obs4_fallback_holding_like", 0) == 0
            # Positive rate should be >0 (from explicit holding values)
            assert summary["holding_positive_rate"] > 0.0

    def test_positive_rates_by_phase_correct(self):
        """Holding positive rates by phase are interpretable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "per_phase_test"
            out_dir = Path(tmpdir) / "output"
            _make_fake_trajectory_npz(
                traj_dir, "ep01", include_explicit_holding=True, obs4_val=0.0)

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4, val_ratio=0.2, seed=0, strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            rates = summary.get("holding_positive_rate_by_phase", {})
            # Phase 2 (lift) should be 100% positive
            assert rates.get("2", 0.0) == 1.0
            # Phase 3 (transport) should be 100% positive
            assert rates.get("3", 0.0) == 1.0
            # Phase 0 (reach_pick) should be 0%
            assert rates.get("0", 1.0) == 0.0
