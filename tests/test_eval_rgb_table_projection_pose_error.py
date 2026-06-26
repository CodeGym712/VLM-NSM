"""Tests for eval_rgb_table_projection_pose_error — mocked, no GPU.

Uses synthetic annotation data and a mocked detector.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
from vlm_nsm.experiments.eval_rgb_table_projection_pose_error import (
    _load_annotations,
    _check_positions,
    evaluate_frame,
    aggregate_results,
    _describe_required_fields,
)


@pytest.fixture
def sample_camera_info():
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = K[1, 1] = 600.0
    K[0, 2] = 255.5
    K[1, 2] = 255.5
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.0, 0.0, 1.0]
    return CameraInfo(K=K, T_camera_world=T, width=512, height=512, camera_name="test")


@pytest.fixture
def sample_frame():
    """A synthetic frame with one red cube and one green cube."""
    return {
        "frame_idx": 0,
        "image_width": 512,
        "image_height": 512,
        "camera_name": "agentview",
        "objects_gt": [
            {"object_id": "cube1", "color": "red", "position": {"x": 0.1, "y": -0.05, "z": 0.8}},
            {"object_id": "cube2", "color": "green", "position": {"x": -0.1, "y": 0.05, "z": 0.8}},
        ],
        "detections": [
            {
                "label": "cube1",
                "color": "red",
                "confidence": 0.95,
                "bbox": {"x1": 200, "y1": 200, "x2": 260, "y2": 260},
                "center_px": {"x": 230, "y": 230},
                "center_norm": {"x": 230 / 512, "y": 230 / 512},
            },
            {
                "label": "cube2",
                "color": "green",
                "confidence": 0.92,
                "bbox": {"x1": 300, "y1": 210, "x2": 360, "y2": 270},
                "center_px": {"x": 330, "y": 240},
                "center_norm": {"x": 330 / 512, "y": 240 / 512},
            },
        ],
    }


class TestLoadAnnotations:
    def test_load_valid_jsonl(self):
        """Valid JSONL file is loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path = Path(tmpdir) / "annotations.jsonl"
            ann_path.write_text(
                '{"image_path": "test.png", "objects": []}\n'
                '{"image_path": "test2.png", "objects": []}\n'
            )
            anns = _load_annotations(Path(tmpdir))
            assert len(anns) == 2


class TestCheckPositions:
    def test_has_positions(self):
        anns = [{"objects": [{"position": {"x": 0.1, "y": 0.0, "z": 0.8}}]}]
        assert _check_positions(anns) is True

    def test_no_positions(self):
        anns = [{"objects": [{"position": None}]}]
        assert _check_positions(anns) is False

    def test_missing_key(self):
        anns = [{"objects": [{"color": "red"}]}]
        assert _check_positions(anns) is False


class TestEvaluateFrame:
    def test_basic_projection(self, sample_camera_info, sample_frame):
        """Basic evaluation returns correct structure."""
        results = evaluate_frame(sample_frame, sample_camera_info,
                                 projection_point="center", table_z=0.0)
        assert len(results) == 2
        for r in results:
            assert "estimated_position" in r
            assert "oracle_position" in r
            assert "xy_error" in r
            assert r["projection_point"] == "center"

    def test_projection_point_modes(self, sample_camera_info, sample_frame):
        """Different projection modes produce different results."""
        center_results = evaluate_frame(sample_frame, sample_camera_info,
                                         projection_point="center")
        bottom_results = evaluate_frame(sample_frame, sample_camera_info,
                                         projection_point="bottom_center")

        assert len(center_results) == len(bottom_results)
        # bottom_center uses y2 which is larger, so v is larger
        # which should lead to different estimated positions
        center_v = center_results[0]["projection_pixel"][1]
        bottom_v = bottom_results[0]["projection_pixel"][1]
        assert bottom_v >= center_v

    def test_weighted_bottom(self, sample_camera_info, sample_frame):
        """weighted_bottom mode runs without error."""
        results = evaluate_frame(sample_frame, sample_camera_info,
                                 projection_point="weighted_bottom")
        assert len(results) == 2

    def test_no_detections(self, sample_camera_info, sample_frame):
        """Frame with no detections returns empty list."""
        empty_frame = {**sample_frame, "detections": [], "objects_gt": []}
        results = evaluate_frame(empty_frame, sample_camera_info)
        assert results == []

    def test_no_bbox_fallback_to_center_px(self, sample_camera_info):
        """Object without bbox but with center_px still works."""
        frame = {
            "objects_gt": [{"color": "red", "position": {"x": 0.0, "y": 0.0, "z": 0.8}}],
            "detections": [{
                "label": "cube1", "color": "red", "confidence": 0.9,
                "center_px": {"x": 255.5, "y": 255.5},
            }],
            "image_width": 512, "image_height": 512,
        }
        results = evaluate_frame(frame, sample_camera_info)
        assert len(results) == 1
        assert results[0]["estimated_position"] is not None

    def test_no_bbox_no_center_px_fallback_to_center_norm(self, sample_camera_info):
        """Object with only center_norm still works."""
        frame = {
            "objects_gt": [{"color": "red", "position": {"x": 0.0, "y": 0.0, "z": 0.8}}],
            "detections": [{
                "label": "cube1", "color": "red", "confidence": 0.9,
                "center_norm": {"x": 0.5, "y": 0.5},
            }],
            "image_width": 512, "image_height": 512,
        }
        results = evaluate_frame(frame, sample_camera_info)
        assert len(results) == 1
        assert results[0]["estimated_position"] is not None


class TestAggregateResults:
    def test_empty(self):
        metrics = aggregate_results([])
        assert metrics["num_objects"] == 0

    def test_basic_aggregation(self):
        records = [
            {"color": "red", "xy_error": 0.01, "x_error": 0.01, "y_error": 0.0, "z_error": 0.0},
            {"color": "red", "xy_error": 0.02, "x_error": 0.02, "y_error": 0.0, "z_error": 0.0},
            {"color": "green", "xy_error": 0.015, "x_error": 0.01, "y_error": 0.01, "z_error": 0.0},
        ]
        metrics = aggregate_results(records)
        assert metrics["num_objects"] == 3
        assert metrics["mean_xy_error"] == pytest.approx(0.015, abs=1e-4)
        assert metrics["median_xy_error"] == pytest.approx(0.015, abs=1e-4)
        assert "red" in metrics["per_color"]
        assert "green" in metrics["per_color"]
        assert metrics["per_color"]["red"]["count"] == 2

    def test_failures_counted_separately(self):
        records = [
            {"color": "red", "xy_error": 0.01, "x_error": 0.0, "y_error": 0.01, "z_error": 0.0},
            {"projection_failed": True, "color": "red"},
            {"match_failed": True, "color": "green"},
        ]
        metrics = aggregate_results(records)
        assert metrics["num_objects"] == 1
        assert metrics["projection_fail_count"] == 1
        assert metrics["match_fail_count"] == 1
