"""Tests for Stage 19-D3-3: Build balanced trainable dataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage19_build_balanced_train_dataset import (
    load_direction_entries,
    stratified_split,
    build_manifest_entry,
    copy_trajectory,
)


def _make_dataset(dir_path: Path, entries: list[dict], include_trajs: bool = True):
    """Create a fake skill demo dataset with manifest and trajectory files."""
    trajectories_dir = dir_path / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for idx, entry in enumerate(entries):
        traj_rel = entry.get("trajectory_path", f"trajectories/ep_{idx:06d}.npz")
        entry["trajectory_path"] = traj_rel
        manifest_lines.append(json.dumps(entry))
        if include_trajs:
            traj_path = dir_path / traj_rel
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            n_steps = entry.get("num_steps", 100)
            np.savez_compressed(
                traj_path,
                obs=np.zeros((n_steps, 11), dtype=np.float32),
                filtered_obs=np.zeros((n_steps, 11), dtype=np.float32),
                action=np.zeros((n_steps, 7), dtype=np.float32),
                done=np.zeros(n_steps, dtype=bool),
                episode_id=np.full(n_steps, idx, dtype=np.int32),
                skill_id=np.zeros(n_steps, dtype=np.int32),
                phase_id=np.zeros(n_steps, dtype=np.int32),
                step_id=np.arange(n_steps, dtype=np.int32),
                success=np.full(n_steps, entry.get("success", True), dtype=bool),
                skill_name=np.full(n_steps, entry.get("skill_name", "pick_place")),
                skill_object_id=np.full(n_steps, entry.get("object_id", "cube1")),
                skill_target_object_id=np.full(n_steps, entry.get("target_object_id", "cube2")),
            )

    manifest_path = dir_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


class TestLoadDirectionEntries:
    def test_loads_successful_with_trajectories(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _make_dataset(dataset_dir, [
            {"episode_id": "ep_001", "success": True, "num_steps": 500,
             "object_id": "cube1", "target_object_id": "cube2", "seed": 0},
            {"episode_id": "ep_002", "success": False, "num_steps": 300,
             "object_id": "cube1", "target_object_id": "cube2", "seed": 1},
            {"episode_id": "ep_003", "success": True, "num_steps": 600,
             "object_id": "cube1", "target_object_id": "cube2", "seed": 2},
        ])

        entries = load_direction_entries(dataset_dir, "forward", 50)
        assert len(entries) == 2
        assert all(e["direction"] == "forward" for e in entries)
        assert [e["episode_id"] for e in entries] == ["ep_001", "ep_003"]

    def test_respects_max_count(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        entries_data = [
            {"episode_id": f"ep_{i:03d}", "success": True, "num_steps": 500,
             "object_id": "cube1", "target_object_id": "cube2", "seed": i}
            for i in range(10)
        ]
        _make_dataset(dataset_dir, entries_data)

        entries = load_direction_entries(dataset_dir, "forward", 3)
        assert len(entries) == 3

    def test_skips_missing_trajectory(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _make_dataset(dataset_dir, [
            {"episode_id": "ep_001", "success": True, "num_steps": 500,
             "object_id": "cube1", "target_object_id": "cube2", "seed": 0},
        ], include_trajs=False)

        entries = load_direction_entries(dataset_dir, "forward", 50)
        assert len(entries) == 0

    def test_no_manifest_returns_empty(self, tmp_path):
        dataset_dir = tmp_path / "empty"
        dataset_dir.mkdir()
        entries = load_direction_entries(dataset_dir, "forward", 50)
        assert entries == []


class TestStratifiedSplit:
    def test_both_directions_in_both_splits(self):
        entries = []
        for i in range(44):
            entries.append({"episode_id": f"fwd_{i:03d}", "direction": "forward", "seed": i})
        for i in range(44):
            entries.append({"episode_id": f"rev_{i:03d}", "direction": "reverse", "seed": i})

        train, val = stratified_split(entries, val_ratio=0.2, seed=42)
        assert len(train) + len(val) == 88

        train_fwd = sum(1 for e in train if e["direction"] == "forward")
        train_rev = sum(1 for e in train if e["direction"] == "reverse")
        val_fwd = sum(1 for e in val if e["direction"] == "forward")
        val_rev = sum(1 for e in val if e["direction"] == "reverse")

        assert train_fwd > 0
        assert train_rev > 0
        assert val_fwd > 0
        assert val_rev > 0

        # Roughly 80/20 split
        assert abs(len(train) - 70) <= 5
        assert abs(len(val) - 18) <= 5

    def test_small_dataset_at_least_1_val_each(self):
        entries = [
            {"episode_id": "fwd_1", "direction": "forward"},
            {"episode_id": "fwd_2", "direction": "forward"},
            {"episode_id": "rev_1", "direction": "reverse"},
            {"episode_id": "rev_2", "direction": "reverse"},
        ]
        train, val = stratified_split(entries, val_ratio=0.2, seed=42)

        val_fwd = sum(1 for e in val if e["direction"] == "forward")
        val_rev = sum(1 for e in val if e["direction"] == "reverse")
        assert val_fwd >= 1
        assert val_rev >= 1


class TestBuildManifestEntry:
    def test_includes_direction(self):
        entry = {
            "episode_id": "rev_ep_000000", "skill_name": "pick_place",
            "object_id": "cube2", "target_object_id": "cube1",
            "relation": "on", "source": "table", "direction": "reverse",
            "success": True, "num_steps": 500, "failure_type": None,
            "seed": 42, "teacher_mode": "scripted_only",
        }
        result = build_manifest_entry(entry, "trajectories/rev_ep_000000.npz", 0)
        assert result["direction"] == "reverse"
        assert result["object_id"] == "cube2"
        assert result["target_object_id"] == "cube1"


class TestCopyTrajectory:
    def test_copies_file(self, tmp_path):
        src_dir = tmp_path / "src"
        src_traj = src_dir / "trajectories/ep_001.npz"
        src_traj.parent.mkdir(parents=True)
        np.savez_compressed(src_traj, obs=np.zeros((10, 11), dtype=np.float32),
                            action=np.zeros((10, 7), dtype=np.float32))

        dst_dir = tmp_path / "dst"
        result = copy_trajectory(str(src_traj), dst_dir, "ep_001")
        assert result == "trajectories/ep_001.npz"
        assert (dst_dir / result).exists()

    def test_nonexistent_file(self, tmp_path):
        dst_dir = tmp_path / "dst"
        result = copy_trajectory("/nonexistent/path.npz", dst_dir, "ep_001")
        assert result is None
