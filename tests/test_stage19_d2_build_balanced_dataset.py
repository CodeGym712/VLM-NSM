"""Tests for Stage 19-D2-4: Balanced dataset index builder."""

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.stage19_build_balanced_skill_dataset import (
    build_balanced_index,
    load_forward_entries,
    load_reverse_successes,
)


class TestLoadReverseSuccesses:
    def test_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()
            result = load_reverse_successes(Path(f.name))
        assert result == []

    def test_filters_failures(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"episode_id": "e1", "task_success": True, "seed": 0}) + "\n")
            f.write(json.dumps({"episode_id": "e2", "task_success": False, "seed": 1}) + "\n")
            f.write(json.dumps({"episode_id": "e3", "task_success": True, "seed": 2}) + "\n")
            f.write(json.dumps({"episode_id": "e4", "task_success": None, "seed": 3}) + "\n")
            f.flush()
            result = load_reverse_successes(Path(f.name))
        assert len(result) == 2
        assert [e["episode_id"] for e in result] == ["e1", "e3"]

    def test_sorts_by_seed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"episode_id": "e3", "task_success": True, "seed": 3}) + "\n")
            f.write(json.dumps({"episode_id": "e1", "task_success": True, "seed": 1}) + "\n")
            f.write(json.dumps({"episode_id": "e2", "task_success": True, "seed": 2}) + "\n")
            f.flush()
            result = load_reverse_successes(Path(f.name))
        assert [e["seed"] for e in result] == [1, 2, 3]

    def test_skips_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"episode_id": "e1", "task_success": True, "seed": 0}) + "\n")
            f.write("\n")
            f.write(json.dumps({"episode_id": "e3", "task_success": True, "seed": 2}) + "\n")
            f.flush()
            result = load_reverse_successes(Path(f.name))
        assert len(result) == 2


class TestLoadForwardEntries:
    def test_filters_success_and_direction(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Valid forward success
            f.write(json.dumps({
                "episode_id": "ep_01", "success": True,
                "object_id": "cube1", "target_object_id": "cube2",
                "skill_name": "pick_place", "trajectory_path": "t/ep_01.npz",
                "num_steps": 500, "seed": 0,
            }) + "\n")
            # Failed — skip
            f.write(json.dumps({
                "episode_id": "ep_02", "success": False,
                "object_id": "cube1", "target_object_id": "cube2",
            }) + "\n")
            # Wrong direction — skip
            f.write(json.dumps({
                "episode_id": "ep_03", "success": True,
                "object_id": "cube2", "target_object_id": "cube1",
            }) + "\n")
            # Valid forward success
            f.write(json.dumps({
                "episode_id": "ep_04", "success": True,
                "object_id": "cube1", "target_object_id": "cube2",
                "num_steps": 600, "seed": 1,
            }) + "\n")
            f.flush()
            result = load_forward_entries(Path(f.name), max_forward=50)
        assert len(result) == 2
        assert [e["episode_id"] for e in result] == ["ep_01", "ep_04"]

    def test_respects_max_forward(self):
        entries_data = [
            {"episode_id": f"ep_{i:03d}", "success": True,
             "object_id": "cube1", "target_object_id": "cube2"}
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in entries_data:
                f.write(json.dumps(entry) + "\n")
            f.flush()
            result = load_forward_entries(Path(f.name), max_forward=3)
        assert len(result) == 3


class TestBuildBalancedIndex:
    def test_basic_balanced(self):
        forward = [
            {"episode_id": "ep_001", "object_id": "cube1", "target_object_id": "cube2",
             "skill_name": "pick_place", "trajectory_path": "t/ep_001.npz",
             "success": True, "num_steps": 500, "seed": 0, "teacher_mode": "hybrid"},
        ]
        reverse = [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 600},
        ]
        result = build_balanced_index(forward, reverse, max_forward=50, max_reverse=50)
        assert result["forward_count"] == 1
        assert result["reverse_count"] == 1
        assert result["total_episodes"] == 2
        assert result["is_balanced"] is True

    def test_caps_at_max(self):
        forward = [
            {"episode_id": f"ep_{i:03d}", "object_id": "cube1",
             "target_object_id": "cube2", "success": True}
            for i in range(100)
        ]
        reverse = [
            {"episode_id": f"rev_{i:03d}", "task_success": True, "seed": i}
            for i in range(100)
        ]
        result = build_balanced_index(forward, reverse, max_forward=50, max_reverse=50)
        assert result["forward_count"] == 50
        assert result["reverse_count"] == 50
        assert result["total_episodes"] == 100
        assert result["is_balanced"] is True

    def test_unbalanced_detected(self):
        forward = [
            {"episode_id": f"ep_{i:03d}", "object_id": "cube1",
             "target_object_id": "cube2", "success": True}
            for i in range(50)
        ]
        reverse = [
            {"episode_id": f"rev_{i:03d}", "task_success": True, "seed": i}
            for i in range(5)
        ]
        result = build_balanced_index(forward, reverse, max_forward=50, max_reverse=50)
        assert result["is_balanced"] is False

    def test_reverse_items_have_correct_mapping(self):
        forward: list[dict] = []
        reverse = [
            {"episode_id": "rev_001", "task_success": True, "seed": 0, "steps": 600},
        ]
        result = build_balanced_index(forward, reverse, max_forward=50, max_reverse=50)
        rev_item = result["reverse_items"][0]
        assert rev_item["direction"] == "reverse"
        assert rev_item["object_mapping"] == {"top": "cube2", "bottom": "cube1"}
        assert rev_item["object_id"] == "cube2"
        assert rev_item["target_object_id"] == "cube1"

    def test_forward_items_have_correct_mapping(self):
        forward = [
            {"episode_id": "ep_001", "object_id": "cube1", "target_object_id": "cube2",
             "skill_name": "pick_place", "trajectory_path": "t/ep_001.npz",
             "success": True, "num_steps": 500, "seed": 0, "teacher_mode": "hybrid"},
        ]
        reverse: list[dict] = []
        result = build_balanced_index(forward, reverse, max_forward=50, max_reverse=50)
        fwd_item = result["forward_items"][0]
        assert fwd_item["direction"] == "forward"
        assert fwd_item["object_mapping"] == {"top": "cube1", "bottom": "cube2"}

    def test_output_has_required_fields(self):
        result = build_balanced_index([], [], max_forward=50, max_reverse=50)
        required_keys = [
            "description", "date", "total_episodes", "forward_count",
            "reverse_count", "is_balanced", "forward_items", "reverse_items", "notes",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
