"""Tests for eval_spatial_relation_accuracy.py — pure logic, no GPU."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test _compute_pairwise_relations
# ---------------------------------------------------------------------------


class TestComputePairwiseRelations:
    def test_basic_left_right(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.1, "y": 0.0},
        }
        rels = _compute_pairwise_relations(positions, xy_threshold=0.02)
        key = ("cube1", "cube2")
        assert rels[key]["b_wrt_a_x"] == "right"  # cube2 is to the right of cube1
        assert rels[key]["b_wrt_a_y"] == "ambiguous"  # y difference below threshold

    def test_basic_front_behind(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.0, "y": 0.1},
        }
        rels = _compute_pairwise_relations(positions, xy_threshold=0.02)
        key = ("cube1", "cube2")
        assert rels[key]["b_wrt_a_y"] == "behind"  # cube2 further from robot (positive y)
        assert rels[key]["b_wrt_a_x"] == "ambiguous"

    def test_both_ambiguous(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.005, "y": 0.005},  # both below 0.02 threshold
        }
        rels = _compute_pairwise_relations(positions, xy_threshold=0.02)
        key = ("cube1", "cube2")
        assert rels[key]["b_wrt_a_x"] == "ambiguous"
        assert rels[key]["b_wrt_a_y"] == "ambiguous"

    def test_near_far(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        near_positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.02, "y": 0.02},
        }
        rels = _compute_pairwise_relations(near_positions, near_threshold=0.05)
        key = ("cube1", "cube2")
        dist = rels[key]["distance"]
        assert dist < 0.05
        assert rels[key]["distance_label"] == "near"

        far_positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.2, "y": 0.2},
        }
        rels = _compute_pairwise_relations(far_positions, far_threshold=0.15)
        assert rels[key]["distance_label"] == "far"

    def test_mid_distance(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        positions = {
            "cube1": {"x": 0.0, "y": 0.0},
            "cube2": {"x": 0.05, "y": 0.05},
        }
        rels = _compute_pairwise_relations(positions, near_threshold=0.02, far_threshold=0.10)
        key = ("cube1", "cube2")
        assert rels[key]["distance_label"] == "mid"

    def test_three_objects(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _compute_pairwise_relations,
        )
        positions = {
            "a": {"x": 0.0, "y": 0.0},
            "b": {"x": 0.1, "y": 0.0},
            "c": {"x": 0.0, "y": 0.1},
        }
        rels = _compute_pairwise_relations(positions, xy_threshold=0.02)
        # Should have 3 pairs
        assert len(rels) == 3
        assert ("a", "b") in rels
        assert ("a", "c") in rels
        assert ("b", "c") in rels


# ---------------------------------------------------------------------------
# Test _evaluate_relation_accuracy
# ---------------------------------------------------------------------------


class TestEvaluateRelationAccuracy:
    def test_perfect_match(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _evaluate_relation_accuracy,
        )
        oracle = {
            ("a", "b"): {
                "b_wrt_a_x": "right", "b_wrt_a_y": "ambiguous",
                "distance_label": "near", "distance": 0.03,
            },
        }
        estimated = {
            ("a", "b"): {
                "b_wrt_a_x": "right", "b_wrt_a_y": "ambiguous",
                "distance_label": "near", "distance": 0.04,
            },
        }
        acc = _evaluate_relation_accuracy(oracle, estimated)
        assert acc["left_right_accuracy"] == 1.0
        assert acc["near_far_accuracy"] == 1.0
        assert acc["overall_accuracy"] == 1.0

    def test_wrong_direction(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _evaluate_relation_accuracy,
        )
        oracle = {
            ("a", "b"): {"b_wrt_a_x": "right", "b_wrt_a_y": "front",
                          "distance_label": "near", "distance": 0.03},
        }
        estimated = {
            ("a", "b"): {"b_wrt_a_x": "left", "b_wrt_a_y": "behind",
                          "distance_label": "far", "distance": 0.20},
        }
        acc = _evaluate_relation_accuracy(oracle, estimated)
        assert acc["left_right_accuracy"] == 0.0
        assert acc["front_back_accuracy"] == 0.0
        assert acc["near_far_accuracy"] == 0.0
        assert acc["wrong_left_right_count"] == 1
        assert acc["wrong_front_back_count"] == 1
        assert acc["wrong_near_far_count"] == 1

    def test_oracle_ambiguous(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            _evaluate_relation_accuracy,
        )
        oracle = {
            ("a", "b"): {"b_wrt_a_x": "ambiguous", "b_wrt_a_y": "front",
                          "distance_label": "near", "distance": 0.01},
        }
        estimated = {
            ("a", "b"): {"b_wrt_a_x": "right", "b_wrt_a_y": "front",
                          "distance_label": "near", "distance": 0.02},
        }
        acc = _evaluate_relation_accuracy(oracle, estimated)
        assert acc["ambiguous_oracle_x_count"] == 1
        assert acc["wrong_left_right_count"] == 1  # estimated says "right" but oracle is ambiguous


# ---------------------------------------------------------------------------
# Integration: end-to-end with temp JSONL
# ---------------------------------------------------------------------------


class TestEvaluateSpatialRelationAccuracy:
    def test_end_to_end_with_jsonl(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            evaluate_spatial_relation_accuracy,
        )
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "debug.jsonl"
            records = [
                {
                    "frame_idx": 0,
                    "object_id": "cube1",
                    "color": "red",
                    "oracle_position": {"x": 0.0, "y": 0.0, "z": 0.8},
                    "triangulated_position": {"x": 0.0, "y": 0.01, "z": 0.8},
                },
                {
                    "frame_idx": 0,
                    "object_id": "cube2",
                    "color": "green",
                    "oracle_position": {"x": 0.1, "y": 0.0, "z": 0.8},
                    "triangulated_position": {"x": 0.1, "y": 0.01, "z": 0.8},
                },
            ]
            with open(jsonl_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            result = evaluate_spatial_relation_accuracy(
                estimated_pose_jsonl=[str(jsonl_path)],
                output=str(Path(tmp) / "output.json"),
                xy_threshold=0.02,
            )

            assert result["num_frames_with_pairs"] == 1
            assert result["overall_relation_accuracy"] is not None

    def test_missing_file_skipped(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            evaluate_spatial_relation_accuracy,
        )
        result = evaluate_spatial_relation_accuracy(
            estimated_pose_jsonl=["/nonexistent/path.jsonl"],
        )
        assert "error" in result

    def test_insufficient_objects_per_frame(self):
        from vlm_nsm.experiments.eval_spatial_relation_accuracy import (
            evaluate_spatial_relation_accuracy,
        )
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "debug.jsonl"
            records = [
                {
                    "frame_idx": 0,
                    "object_id": "cube1",
                    "color": "red",
                    "oracle_position": {"x": 0.0, "y": 0.0, "z": 0.8},
                    "triangulated_position": {"x": 0.0, "y": 0.01, "z": 0.8},
                },
            ]
            with open(jsonl_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            result = evaluate_spatial_relation_accuracy(
                estimated_pose_jsonl=[str(jsonl_path)],
            )
            # One object = no pairs = no accuracy
            assert result["num_frames_with_pairs"] == 0
