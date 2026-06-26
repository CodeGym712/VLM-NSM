"""Tests for eval_dual_camera_triangulation_pose_error.py — pure logic, no GPU."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestMatchDetectionsByColor:
    def test_perfect_match(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _match_detections_by_color,
        )
        dets_a = [{"color": "red", "label": "red_cube"}]
        dets_b = [{"color": "red", "label": "red_cube"}]
        matches = _match_detections_by_color(dets_a, dets_b)
        assert matches == [(0, 0)]

    def test_color_fallback(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _match_detections_by_color,
        )
        dets_a = [{"color": "", "label": "red_cube"}]
        dets_b = [{"color": "red", "label": ""}]
        matches = _match_detections_by_color(dets_a, dets_b)
        assert matches == [(0, 0)]

    def test_no_cross_match(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _match_detections_by_color,
        )
        dets_a = [{"color": "red", "label": ""}, {"color": "green", "label": ""}]
        dets_b = [{"color": "green", "label": ""}, {"color": "red", "label": ""}]
        matches = _match_detections_by_color(dets_a, dets_b)
        # Each detection in B used at most once
        assert len(matches) == 2
        # First red in A matches first green? No — red in A should match red in B (idx 1)
        assert (0, 1) in matches
        assert (1, 0) in matches

    def test_extra_in_one_view(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _match_detections_by_color,
        )
        dets_a = [{"color": "red", "label": ""}]
        dets_b = [{"color": "red", "label": ""}, {"color": "green", "label": ""}]
        matches = _match_detections_by_color(dets_a, dets_b)
        assert matches == [(0, 0)]


class TestExtractPixelFromDetection:
    def test_bbox_center_mode(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _extract_pixel_from_detection,
        )
        det = {"bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 250}}
        px = _extract_pixel_from_detection(det, mode="center")
        assert px is not None
        assert px[0] == pytest.approx(150.0)  # (100+200)/2
        assert px[1] == pytest.approx(200.0)  # (150+250)/2

    def test_bbox_bottom_center_mode(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _extract_pixel_from_detection,
        )
        det = {"bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 250}}
        px = _extract_pixel_from_detection(det, mode="bottom_center")
        assert px is not None
        assert px[0] == pytest.approx(150.0)
        assert px[1] == pytest.approx(250.0)  # y2

    def test_center_px_fallback(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _extract_pixel_from_detection,
        )
        det = {"center_px": {"x": 320, "y": 240}}
        px = _extract_pixel_from_detection(det, mode="center")
        assert px == (320.0, 240.0)

    def test_center_norm_fallback(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _extract_pixel_from_detection,
        )
        det = {"center_norm": {"x": 0.5, "y": 0.4}}
        px = _extract_pixel_from_detection(det, mode="center")
        assert px == (256.0, 204.8)

    def test_no_pixel_available(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _extract_pixel_from_detection,
        )
        det: dict = {}
        px = _extract_pixel_from_detection(det, mode="center")
        assert px is None


class TestAggregate:
    def test_empty_records(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _aggregate,
        )
        result = _aggregate([])
        assert result["num_objects_total"] == 0
        assert result["num_valid_triangulations"] == 0

    def test_valid_records(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _aggregate,
        )
        records = [
            {"xy_error": 0.02, "xyz_error": 0.025, "x_error": 0.01, "y_error": 0.015, "z_error": 0.01,
             "triangulation_residual": 0.001, "color": "red"},
            {"xy_error": 0.03, "xyz_error": 0.035, "x_error": 0.02, "y_error": 0.02, "z_error": 0.015,
             "triangulation_residual": 0.002, "color": "red"},
        ]
        result = _aggregate(records)
        assert result["num_objects_total"] == 2
        assert result["num_valid_triangulations"] == 2
        assert result["median_xy_error"] == pytest.approx(0.025)
        assert result["mean_xy_error"] == pytest.approx(0.025)
        assert result["per_axis_abs_error_mean"]["x"] == pytest.approx(0.015)
        assert result["per_color"]["red"]["mean_xy_error"] == pytest.approx(0.025)

    def test_mixed_valid_and_failed(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _aggregate,
        )
        records = [
            {"xy_error": 0.02, "x_error": 0.01, "y_error": 0.01, "z_error": 0.01,
             "triangulation_residual": 0.001, "color": "red"},
            {"failure": "triangulation_failed"},
            {"failure": "no_pixel"},
        ]
        result = _aggregate(records)
        assert result["num_objects_total"] == 3
        assert result["num_valid_triangulations"] == 1
        assert result["failure_count"] == 2
        assert result["triangulation_fail_count"] == 1
        assert result["no_pixel_count"] == 1


class TestNormColor:
    def test_standard(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _norm_color,
        )
        assert _norm_color("red") == "red"
        assert _norm_color("red_cube") == "red"
        assert _norm_color("GREEN") == "green"


class TestLoadCalibration:
    def test_loads_valid_calibration(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            load_calibration,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calib.json"
            calib = {
                "K": [[618.0, 0.0, 255.5], [0.0, 618.0, 255.5], [0.0, 0.0, 1.0]],
                "T_camera_world": [[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 1.35], [0, 0, 0, 1]],
                "width": 512,
                "height": 512,
                "camera_name": "agentview",
            }
            path.write_text(json.dumps(calib))
            info = load_calibration(str(path))
            assert info.camera_name == "agentview"
            assert info.K.shape == (3, 3)
            assert info.T_camera_world.shape == (4, 4)


class TestToXyz:
    def test_list_format(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _to_xyz,
        )
        pos = _to_xyz([0.1, 0.2, 0.8])
        assert np.allclose(pos, [0.1, 0.2, 0.8])

    def test_dict_format(self):
        from vlm_nsm.experiments.eval_dual_camera_triangulation_pose_error import (
            _to_xyz,
        )
        pos = _to_xyz({"x": 0.1, "y": 0.2, "z": 0.8})
        assert np.allclose(pos, [0.1, 0.2, 0.8])
