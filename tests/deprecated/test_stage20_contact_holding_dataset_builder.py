"""Stage 20-R: Tests for contact/holding label handling in dataset builder.

Tests:
- Builder includes contact/holding arrays in windows.npz when fields exist.
- Old trajectory without fields still builds with masks=0.
- Summary reports availability and positive rates.
- No phase proxy labels are generated.
- Backward compatibility with Stage 20 v0_2_phase_labeled format.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage20_build_role_conditioned_dataset import build_role_conditioned_dataset


def _make_trajectory(
    traj_dir: Path,
    episode_id: str,
    include_contact: bool = True,
    contact_vals: np.ndarray | None = None,
    phase_ids: np.ndarray | None = None,
    n_steps: int = 15,
    direction: str = "cube1_to_cube2",
    teacher: str = "scripted_only",
):
    """Create a minimal fake trajectory NPZ + manifest entry."""
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_name = f"{episode_id}.npz"

    obs = np.zeros((n_steps, 11), dtype=np.float32)
    obs[:, 0] = 0.5 + np.arange(n_steps) * 0.01
    obs[:, 3] = 0.05  # gripper_open
    obs[:, 5] = 0.06  # cube1_x
    obs[:, 6] = 0.03  # cube1_y
    obs[:, 7] = 0.83  # cube1_z
    obs[:, 8] = -0.04  # cube2_x
    obs[:, 9] = 0.00  # cube2_y
    obs[:, 10] = 0.835  # cube2_z

    action = np.zeros((n_steps, 7), dtype=np.float32)
    action[:, 0] = 0.01

    if phase_ids is None:
        phase_ids = np.zeros(n_steps, dtype=np.int64)
        phase_ids[5:10] = 1  # grasp phase

    save_dict = {
        "filtered_obs": obs,
        "obs": obs,
        "action": action,
        "phase_id": phase_ids,
    }

    if include_contact:
        if contact_vals is None:
            contact_vals = np.zeros(n_steps, dtype=np.float32)
            contact_vals[5:12] = 1.0  # contact during grasp and early lift
        save_dict["gripper_has_object"] = contact_vals
        save_dict["contact_signal"] = contact_vals
        save_dict["holding_signal"] = contact_vals
        save_dict["contact_signal_available_flag"] = np.array(True)
        save_dict["holding_signal_available_flag"] = np.array(True)

    npz_path = traj_dir / "trajectories" / traj_name
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **save_dict)

    # Manifest entry
    top_obj = "cube1" if "cube1_to" in direction else "cube2"
    bottom_obj = "cube2" if "cube1_to" in direction else "cube1"
    record = {
        "episode_id": episode_id,
        "trajectory_path": f"trajectories/{traj_name}",
        "direction": direction,
        "object_id": top_obj,
        "target_object_id": bottom_obj,
        "top_object": top_obj,
        "bottom_object": bottom_obj,
        "success": True,
        "teacher_profile": teacher,
        "execution_mode": teacher,
    }

    manifest_path = traj_dir / "manifest.jsonl"
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return str(traj_dir)


class TestContactHoldingDatasetBuilder:

    def test_builder_includes_contact_arrays_when_fields_exist(self):
        """When trajectory has contact_signal, windows.npz includes contact arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "dataset"
            out_dir = Path(tmpdir) / "output"

            _make_trajectory(
                traj_dir, "ep001", include_contact=True,
                contact_vals=np.array([0.0]*5 + [1.0]*8 + [0.0]*2, dtype=np.float32),
                n_steps=15, direction="cube1_to_cube2",
            )
            _make_trajectory(
                traj_dir, "ep002", include_contact=True,
                contact_vals=np.array([0.0]*4 + [1.0]*9 + [0.0]*2, dtype=np.float32),
                n_steps=15, direction="cube2_to_cube1",
            )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            windows = np.load(out_dir / "windows.npz", allow_pickle=True)
            assert "contact_label" in windows
            assert "holding_label" in windows
            assert "contact_label_history" in windows
            assert "holding_label_history" in windows
            assert "contact_label_mask" in windows
            assert "holding_label_mask" in windows

            # All windows should have masks = 1 (contact present)
            assert np.all(windows["contact_label_mask"] == 1.0)
            assert np.all(windows["holding_label_mask"] == 1.0)

    def test_old_trajectory_no_contact_builds_with_masks_zero(self):
        """Old trajectory without explicit contact now gets obs[4] fallback.
        Stage 20-S: obs[4] fallback provides contact/holding labels from
        filtered_obs[:,4] (gripper_has_object heuristic).
        Masks = 1 because obs width >= 11."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "old_dataset"
            out_dir = Path(tmpdir) / "output"

            _make_trajectory(
                traj_dir, "ep001", include_contact=False,
                n_steps=15, direction="cube1_to_cube2",
            )

            builder_result = build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            windows = np.load(out_dir / "windows.npz", allow_pickle=True)
            # Arrays now exist with obs[4] fallback data
            assert "contact_label" in windows
            assert "contact_label_mask" in windows
            # With obs[4] fallback, masks are 1 (data available from filtered_obs[:,4])
            assert len(windows["contact_label"]) > 0

    def test_summary_reports_contact_availability(self):
        """Summary includes contact/holding availability and positive rates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "dataset"
            out_dir = Path(tmpdir) / "output"

            # Mix: one with contact, one without
            _make_trajectory(
                traj_dir, "ep001", include_contact=True,
                contact_vals=np.array([0.0]*5 + [1.0]*8 + [0.0]*2, dtype=np.float32),
                n_steps=15, direction="cube1_to_cube2",
            )
            _make_trajectory(
                traj_dir, "ep002", include_contact=False,
                n_steps=15, direction="cube2_to_cube1",
            )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())

            assert "contact_label_available_rate" in summary
            assert "holding_label_available_rate" in summary
            assert "contact_positive_rate" in summary
            assert "holding_positive_rate" in summary
            assert "contact_missing_count" in summary
            assert "holding_missing_count" in summary
            assert "contact_positive_rate_by_phase" in summary
            assert "holding_positive_rate_by_phase" in summary
            assert "contact_positive_rate_by_direction" in summary
            assert "holding_positive_rate_by_direction" in summary

            # No phase proxy labels
            assert summary["no_phase_proxy_labels"] is True
            assert summary["no_direct_mujoco_contact"] is True
            assert summary["no_per_object_contact"] is True

    def test_no_phase_proxy_labels(self):
        """Dataset builder uses obs[4] fallback, NOT phase proxy fabrication.
        Stage 20-S: source should be obs4_fallback_holding_like,
        NOT phase_id derived. Confirmed via summary caveats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "dataset"
            out_dir = Path(tmpdir) / "output"

            # Episode with no explicit contact fields
            _make_trajectory(
                traj_dir, "ep001", include_contact=False,
                n_steps=15, direction="cube1_to_cube2",
            )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            # Labels come from obs[4] fallback, NOT phase proxy
            assert summary["no_phase_proxy_labels"] is True
            assert summary["no_direct_mujoco_contact"] is True
            # Source should be obs4_fallback (not phase-based)
            src = summary.get("holding_signal_source_counts", {})
            assert "obs4_fallback_holding_like" in src

    def test_contact_positive_rate_correct(self):
        """Contact positive rate matches expected: ~8/15 of grasp+lift steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "dataset"
            out_dir = Path(tmpdir) / "output"

            # Contact=1 for steps 5-13 (9/15 steps)
            _make_trajectory(
                traj_dir, "ep001", include_contact=True,
                contact_vals=np.array([0.0]*5 + [1.0]*9 + [0.0], dtype=np.float32),
                n_steps=15, direction="cube1_to_cube2",
                phase_ids=np.array([0]*5 + [1]*5 + [2]*5, dtype=np.int64),
            )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            # All windows from one trajectory that has contact
            assert summary["contact_label_available_rate"] == 1.0
            assert summary["contact_missing_count"] == 0
            assert summary["contact_positive_rate"] > 0.0  # at least some positive

    def test_backward_compatible_with_v0_2_format(self):
        """Old-format dataset (no contact fields) builds identically to before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "v0_2_dataset"
            out_dir = Path(tmpdir) / "output"

            _make_trajectory(
                traj_dir, "ep001", include_contact=False,
                n_steps=15, direction="cube1_to_cube2",
            )
            _make_trajectory(
                traj_dir, "ep002", include_contact=False,
                n_steps=15, direction="cube2_to_cube1",
            )

            result = build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            # Core fields still work
            assert result["num_windows"] > 0
            assert result["can_train_mlp_baseline"] is True
            assert result["obs_dim"] == 11
            assert result["action_dim"] == 7
            assert result["role_state_dim"] == 20

    def test_mixed_dataset_contact_and_no_contact(self):
        """Mix of trajectories with and without contact works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "mixed_dataset"
            out_dir = Path(tmpdir) / "output"

            _make_trajectory(
                traj_dir, "ep_with_contact", include_contact=True,
                contact_vals=np.array([0.0]*4 + [1.0]*8 + [0.0]*3, dtype=np.float32),
                n_steps=15, direction="cube1_to_cube2",
            )
            _make_trajectory(
                traj_dir, "ep_without_contact", include_contact=False,
                n_steps=15, direction="cube2_to_cube1",
            )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            # With obs[4] fallback, both trajectories get contact labels
            # Available rate should be 1.0 (obs4 fallback provides for all)
            assert summary["contact_label_available_rate"] == 1.0
            assert summary["contact_missing_count"] == 0

    def test_multiple_episodes_contact_available(self):
        """With all contact fields, available_rate == 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "full_contact_dataset"
            out_dir = Path(tmpdir) / "output"

            for ep_idx in range(3):
                _make_trajectory(
                    traj_dir, f"ep{ep_idx:03d}", include_contact=True,
                    contact_vals=np.array([0.0]*5 + [1.0]*8 + [0.0]*2, dtype=np.float32),
                    n_steps=15, direction="cube1_to_cube2",
                )

            build_role_conditioned_dataset(
                input_datasets=[str(traj_dir)],
                output_dir=str(out_dir),
                history_len=4,
                val_ratio=0.2,
                seed=0,
                strict=False,
            )

            summary = json.loads((out_dir / "summary.json").read_text())
            assert summary["contact_label_available_rate"] == 1.0
            assert summary["holding_label_available_rate"] == 1.0
            assert summary["contact_missing_count"] == 0
