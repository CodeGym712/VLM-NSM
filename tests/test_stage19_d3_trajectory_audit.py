"""Tests for Stage 19-D3-1: Trajectory audit."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage19_d3_trajectory_audit import (
    audit_forward_dataset,
    audit_reverse_episodes,
    check_format_compatibility,
    run_audit,
)


def _make_forward_dataset(dir_path: Path, entries: list[dict] | None = None,
                          include_trajs: bool = True) -> dict:
    """Create a fake forward dataset with manifest and trajectory files."""
    trajectories_dir = dir_path / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    if entries is None:
        entries = [
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz",
             "success": True, "object_id": "cube1", "target_object_id": "cube2",
             "num_steps": 500, "seed": 0},
            {"episode_id": "ep_002", "trajectory_path": "trajectories/ep_002.npz",
             "success": True, "object_id": "cube1", "target_object_id": "cube2",
             "num_steps": 600, "seed": 1},
            {"episode_id": "ep_003", "trajectory_path": "trajectories/ep_003.npz",
             "success": False, "object_id": "cube1", "target_object_id": "cube2",
             "num_steps": 300, "seed": 2},
        ]

    manifest_lines = []
    for idx, entry in enumerate(entries):
        manifest_lines.append(json.dumps(entry))
        if include_trajs:
            traj_path = dir_path / entry.get("trajectory_path", "")
            if traj_path:
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
                    skill_name=np.full(n_steps, "pick_place"),
                    skill_object_id=np.full(n_steps, entry.get("object_id", "cube1")),
                    skill_target_object_id=np.full(n_steps, entry.get("target_object_id", "cube2")),
                    skill_relation=np.full(n_steps, "on"),
                    skill_source=np.full(n_steps, "table"),
                    skill_demo_episode_index=np.full(n_steps, idx, dtype=np.int32),
                    skill_demo_success=np.full(n_steps, entry.get("success", True), dtype=bool),
                )

    manifest_path = dir_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return {"entries": entries, "dir": dir_path}


def _make_reverse_episodes(file_path: Path, episodes: list[dict]) -> list[dict]:
    """Create a fake reverse episodes JSONL file."""
    lines = [json.dumps(ep) for ep in episodes]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return episodes


class TestAuditForwardDataset:
    def test_all_matched(self, tmp_path):
        dataset_dir = tmp_path / "forward"
        _make_forward_dataset(dataset_dir)

        index_items = [
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz"},
            {"episode_id": "ep_002", "trajectory_path": "trajectories/ep_002.npz"},
        ]

        result = audit_forward_dataset(dataset_dir, index_items)
        assert result["manifest_exists"] is True
        assert result["trajectories_dir_exists"] is True
        assert result["matched_trajectories"] == 2
        assert result["obs_dim"] == 11
        assert result["action_dim"] == 7
        assert result["has_phase_id"] is True
        assert result["has_skill_labels"] is True
        assert result["compatible_with_skill_dataset"] is True

    def test_missing_from_manifest(self, tmp_path):
        dataset_dir = tmp_path / "forward_missing"
        _make_forward_dataset(dataset_dir)

        index_items = [
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz"},
            {"episode_id": "ep_NONEXISTENT", "trajectory_path": "trajectories/ep_999.npz"},
        ]

        result = audit_forward_dataset(dataset_dir, index_items)
        assert result["matched_trajectories"] == 1
        assert "ep_NONEXISTENT" in result["missing_from_manifest"]

    def test_missing_trajectory_file(self, tmp_path):
        dataset_dir = tmp_path / "forward_missing_traj"
        _make_forward_dataset(dataset_dir, include_trajs=False)

        index_items = [
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz"},
        ]

        result = audit_forward_dataset(dataset_dir, index_items)
        assert result["matched_trajectories"] == 0
        assert "ep_001" in result["missing_trajectory_file"]

    def test_no_manifest(self, tmp_path):
        dataset_dir = tmp_path / "forward_nomanifest"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "trajectories").mkdir()

        index_items = [{"episode_id": "ep_001"}]
        result = audit_forward_dataset(dataset_dir, index_items)
        assert result["manifest_exists"] is False
        assert result["matched_trajectories"] == 0


class TestAuditReverseEpisodes:
    def test_all_available_in_episodes_no_trajectories(self, tmp_path):
        ep_file = tmp_path / "reverse.jsonl"
        episodes = [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 500, "trajectory_path": ""},
            {"episode_id": "rev_002", "task_success": False, "seed": 1, "steps": 300, "trajectory_path": ""},
            {"episode_id": "rev_003", "task_success": True, "seed": 2, "steps": 700, "trajectory_path": ""},
        ]
        _make_reverse_episodes(ep_file, episodes)

        index_items = [
            {"episode_id": "rev_001"},
            {"episode_id": "rev_002"},
            {"episode_id": "rev_003"},
        ]

        result = audit_reverse_episodes(ep_file, index_items)
        assert result["episodes_in_file"] == 3
        assert result["matched_in_episodes"] == 3
        assert result["has_trajectory_files"] == 0
        assert result["trajectory_available"] is False
        assert result["success_count"] == 2
        assert result["format"] == "episode_summary_only"

    def test_with_trajectory_files(self, tmp_path):
        ep_file = tmp_path / "reverse_with_traj.jsonl"
        episodes = [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 500,
             "trajectory_path": "trajectories/rev_001.npz"},
        ]
        _make_reverse_episodes(ep_file, episodes)

        index_items = [{"episode_id": "rev_001"}]

        result = audit_reverse_episodes(ep_file, index_items)
        assert result["has_trajectory_files"] == 1
        assert result["trajectory_available"] is True

    def test_missing_episode(self, tmp_path):
        ep_file = tmp_path / "reverse_missing.jsonl"
        _make_reverse_episodes(ep_file, [{"episode_id": "rev_001", "task_success": True, "steps": 500}])

        index_items = [{"episode_id": "rev_001"}, {"episode_id": "rev_NONEXISTENT"}]

        result = audit_reverse_episodes(ep_file, index_items)
        assert result["matched_in_episodes"] == 1
        assert "rev_NONEXISTENT" in result["missing_from_episodes"]


class TestCheckFormatCompatibility:
    def test_incompatible_when_reverse_missing(self):
        forward = {"compatible_with_skill_dataset": True, "obs_dim": 11, "action_dim": 7}
        reverse = {"trajectory_available": False, "has_trajectory_files": 0}
        result = check_format_compatibility(forward, reverse)
        assert result["formats_compatible"] is False
        assert result["can_build_directly"] is False
        assert result["needs_reverse_collection"] is True

    def test_compatible_when_both_ready(self):
        forward = {"compatible_with_skill_dataset": True, "obs_dim": 11, "action_dim": 7}
        reverse = {"trajectory_available": True, "has_trajectory_files": 50}
        result = check_format_compatibility(forward, reverse)
        assert result["formats_compatible"] is True
        assert result["can_build_directly"] is True
        assert result["needs_reverse_collection"] is False

    def test_incompatible_obs_dim(self):
        forward = {"compatible_with_skill_dataset": True, "obs_dim": 20, "action_dim": 7}
        reverse = {"trajectory_available": True, "has_trajectory_files": 50}
        result = check_format_compatibility(forward, reverse)
        assert "obs_dim" in str(result["issues"])


class TestRunAudit:
    def test_full_audit_with_fake_data(self, tmp_path):
        # Setup forward
        fwd_dir = tmp_path / "forward"
        _make_forward_dataset(fwd_dir, entries=[
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz",
             "success": True, "object_id": "cube1", "target_object_id": "cube2",
             "num_steps": 500, "seed": 0},
        ])

        # Setup reverse (no trajectories)
        rev_file = tmp_path / "reverse.jsonl"
        _make_reverse_episodes(rev_file, [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 500, "trajectory_path": ""},
        ])

        balanced_index = {
            "forward_items": [{"episode_id": "ep_001"}],
            "reverse_items": [{"episode_id": "rev_001"}],
        }

        result = run_audit(balanced_index, fwd_dir, rev_file)

        assert result["summary"]["forward_trajectories_available"] == 1
        assert result["summary"]["reverse_trajectories_available"] == 0
        assert result["summary"]["ready_to_build_dataset"] is False
        assert result["summary"]["needs_reverse_collection"] is True
        assert "REVERSE_TRAJECTORIES_UNAVAILABLE" in result["recommendation"]

    def test_audit_output_has_required_keys(self, tmp_path):
        fwd_dir = tmp_path / "forward_full"
        _make_forward_dataset(fwd_dir, entries=[
            {"episode_id": "ep_001", "trajectory_path": "trajectories/ep_001.npz",
             "success": True, "object_id": "cube1", "target_object_id": "cube2",
             "num_steps": 500, "seed": 0},
        ])
        rev_file = tmp_path / "reverse_full.jsonl"
        _make_reverse_episodes(rev_file, [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 500, "trajectory_path": ""},
        ])

        balanced_index = {
            "forward_items": [{"episode_id": "ep_001"}],
            "reverse_items": [{"episode_id": "rev_001"}],
        }

        result = run_audit(balanced_index, fwd_dir, rev_file)

        for key in ["audit_date", "balanced_index", "forward_audit", "reverse_audit",
                     "format_compatibility", "recommendation", "summary"]:
            assert key in result, f"Missing key: {key}"

        for key in ["forward_trajectories_available", "reverse_trajectories_available",
                     "ready_to_build_dataset", "needs_reverse_collection"]:
            assert key in result["summary"], f"Missing summary key: {key}"
