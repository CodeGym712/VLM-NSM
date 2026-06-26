import json
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage20_build_role_conditioned_dataset import (
    GOAL_DELTA_DIM,
    REACHED_GOAL_DELTA_DIM,
    ROLE_STATE_DIM,
    build_role_conditioned_dataset,
)


def _write_fake_dataset(root: Path, *, missing_obs: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    traj_dir = root / "trajectories"
    traj_dir.mkdir()
    records = []
    specs = [
        ("ep_fwd", "cube1", "cube2", "cube1_to_cube2"),
        ("ep_rev", "cube2", "cube1", "cube2_to_cube1"),
    ]
    for idx, (episode_id, top, bottom, direction) in enumerate(specs):
        t = 8
        obs = np.zeros((t, 11), dtype=np.float32)
        obs[:, 0] = np.linspace(0.0, 0.07, t)
        obs[:, 1] = 0.01 * idx
        obs[:, 2] = 1.0
        obs[:, 3] = np.linspace(1.0, 0.0, t)
        obs[:, 4] = 0.0
        obs[:, 5:8] = np.array([0.0, 0.0, 0.83], dtype=np.float32)
        obs[:, 8:11] = np.array([0.05, 0.0, 0.835], dtype=np.float32)
        obs[:, 5] += np.linspace(0.0, 0.02, t)
        obs[:, 8] -= np.linspace(0.0, 0.01, t)
        action = np.ones((t, 7), dtype=np.float32) * 0.1
        phase_id = np.array([0, 0, 1, 1, 2, 3, 4, 5], dtype=np.int64)
        arrays = {"action": action, "phase_id": phase_id}
        if not missing_obs:
            arrays["obs"] = obs
            arrays["filtered_obs"] = obs
        np.savez_compressed(traj_dir / f"{episode_id}.npz", **arrays)
        records.append({
            "episode_id": episode_id,
            "skill_name": "pick_place",
            "object_id": top,
            "target_object_id": bottom,
            "direction": direction,
            "goal_id": "red_on_green" if direction == "cube1_to_cube2" else "green_on_red",
            "success": True,
            "trajectory_path": f"trajectories/{episode_id}.npz",
            "teacher_profile": "cube2_robust",
            "initial_scene": {
                "object_poses": {
                    "cube1": {"x": 0.0, "y": 0.0, "z": 0.83},
                    "cube2": {"x": 0.05, "y": 0.0, "z": 0.835},
                }
            },
        })
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return root


def test_builder_loads_fake_manifest_and_npz(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"

    summary = build_role_conditioned_dataset(
        input_datasets=[dataset],
        output_dir=out,
        history_len=4,
        val_ratio=0.25,
        seed=0,
        strict=True,
    )

    assert summary["num_trajectories_loaded"] == 2
    assert summary["num_trajectories_used"] == 2
    assert summary["num_windows"] == 10
    assert (out / "windows.npz").exists()
    assert (out / "manifest.jsonl").exists()


def test_builder_creates_windows_with_expected_shapes(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    data = np.load(out / "windows.npz")
    assert data["obs_history"].shape == (10, 4, 11)
    assert data["role_state_history"].shape == (10, 4, ROLE_STATE_DIM)
    assert data["goal_delta_history"].shape == (10, 4, GOAL_DELTA_DIM)
    assert data["goal_delta_mask"].shape == (10, 4, GOAL_DELTA_DIM)
    assert data["reached_goal_delta_history"].shape == (10, 4, REACHED_GOAL_DELTA_DIM)
    assert data["actions"].shape == (10, 7)


def test_builder_computes_role_state_from_obs(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    data = np.load(out / "windows.npz")
    role = data["role_state_history"][0, -1]
    np.testing.assert_allclose(role[0:3], data["obs_history"][0, -1, 0:3])
    np.testing.assert_allclose(role[4:7], data["obs_history"][0, -1, 5:8])
    np.testing.assert_allclose(role[7:10], data["obs_history"][0, -1, 8:11])
    np.testing.assert_allclose(role[10:13], role[0:3] - role[4:7])


def test_builder_computes_goal_delta_and_progress(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    data = np.load(out / "windows.npz")
    assert np.isfinite(data["goal_delta_history"]).all()
    assert np.isfinite(data["progress_scalar_history"]).all()
    assert ((data["progress_scalar_history"] >= 0.0) & (data["progress_scalar_history"] <= 1.0)).all()
    assert data["goal_delta_mask"].sum() > 0


def test_strict_false_skips_missing_pose_samples_and_records_missing_fields(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input", missing_obs=True)
    out = tmp_path / "out"
    summary = build_role_conditioned_dataset(
        input_datasets=[dataset],
        output_dir=out,
        history_len=4,
        strict=False,
    )

    assert summary["num_windows"] == 0
    assert summary["missing_field_counts"]["obs"] == 2
    assert summary["can_train_mlp_baseline"] is False


def test_strict_true_raises_on_missing_required_fields(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input", missing_obs=True)
    with pytest.raises(ValueError):
        build_role_conditioned_dataset(
            input_datasets=[dataset],
            output_dir=tmp_path / "out",
            history_len=4,
            strict=True,
        )


def test_train_val_split_contains_both_direction_and_phase_when_available(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    summary = build_role_conditioned_dataset(
        input_datasets=[dataset],
        output_dir=out,
        history_len=4,
        val_ratio=0.4,
        seed=1,
    )

    train = [json.loads(line) for line in (out / "train_manifest.jsonl").read_text().splitlines()]
    val = [json.loads(line) for line in (out / "val_manifest.jsonl").read_text().splitlines()]
    assert train
    assert val
    assert set(summary["direction_counts"]) == {"cube1_to_cube2", "cube2_to_cube1"}
    assert len(summary["phase_counts"]) > 1


def test_normalization_stats_contains_finite_mean_std(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    stats = json.loads((out / "normalization_stats.json").read_text())
    for section in ["obs_history", "role_state_history", "goal_delta_history", "reached_goal_delta_history", "actions"]:
        mean = np.asarray(stats[section]["mean"], dtype=float)
        std = np.asarray(stats[section]["std"], dtype=float)
        assert np.isfinite(mean).all()
        assert np.isfinite(std).all()
        assert (std > 0).all()


def test_summary_includes_mapping_dictionaries(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    summary = build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    assert "phase_to_id" in summary
    assert "direction_to_id" in summary
    assert "placement_bin_to_id" in summary
    assert "teacher_profile_to_id" in summary
    assert summary["role_state_dim"] == ROLE_STATE_DIM


def test_readme_generated(tmp_path):
    dataset = _write_fake_dataset(tmp_path / "input")
    out = tmp_path / "out"
    build_role_conditioned_dataset(input_datasets=[dataset], output_dir=out, history_len=4)

    text = (out / "README.md").read_text()
    assert "Role-Conditioned Stack Skill Dataset" in text
    assert "Feature Layout" in text
