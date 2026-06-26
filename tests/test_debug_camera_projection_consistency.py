"""Tests for debug_camera_projection_consistency.py — uses mocks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from vlm_nsm.perception.camera_geometry import project_world_point_to_pixel


AGENTVIEW_K = np.array(
    [[618.0386719675123, 0.0, 255.5],
     [0.0, 618.0386719675123, 255.5],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

AGENTVIEW_T = np.array(
    [[0.0, -0.706147844353306, 0.7080644193257978, 0.5],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.7080644193257978, 0.706147844353306, 1.35],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


class TestColorMapForMatching:
    def test_standard_colors(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            color_map_for_matching,
        )
        assert color_map_for_matching("red") == "red"
        assert color_map_for_matching("red_cube") == "red"
        assert color_map_for_matching("green_cube") == "green"
        assert color_map_for_matching("blue_cube") == "blue"

    def test_unknown_color_passthrough(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            color_map_for_matching,
        )
        assert color_map_for_matching("unknown_color") == "unknown_color"


class TestFindMatch:
    def test_match_by_color(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _find_match,
        )
        detections = [
            {"label": "red_cube", "color": "red"},
            {"label": "green_cube", "color": "green"},
        ]
        match = _find_match("red", detections)
        assert match is not None
        assert match["label"] == "red_cube"

    def test_no_match(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _find_match,
        )
        detections = [{"label": "green_cube", "color": "green"}]
        match = _find_match("red", detections)
        assert match is None

    def test_empty_detections(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _find_match,
        )
        assert _find_match("red", []) is None


class TestMakeRecord:
    def test_record_with_valid_projection_and_detection(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _make_record,
        )
        from vlm_nsm.perception.robosuite_camera_utils import CameraInfo

        cam_info = CameraInfo(
            K=AGENTVIEW_K, T_camera_world=AGENTVIEW_T,
            width=512, height=512, camera_name="agentview",
        )

        # Point visible from agentview
        oracle_xyz = np.array([0.05, -0.02, 0.82], dtype=np.float64)

        # Detection at a slightly offset bbox
        proj_pixel = project_world_point_to_pixel(
            oracle_xyz, cam_info.K, cam_info.T_camera_world,
        )
        assert proj_pixel is not None
        pu, pv = proj_pixel

        detection = {
            "label": "red_cube",
            "color": "red",
            "confidence": 0.9,
            "bbox": {
                "x1": pu - 20, "y1": pv - 40,
                "x2": pu + 20, "y2": pv,
            },
            "center_px": {"x": pu, "y": pv - 20},
        }

        record = _make_record(
            0, "cube1", "red", oracle_xyz,
            detection, cam_info, "agentview", 0.8,
        )

        assert record["agentview_projected_valid"] is True
        assert record["agentview_detected"] is True
        assert record["agentview_projected_u"] == pytest.approx(pu)
        assert record["agentview_projected_v"] == pytest.approx(pv)
        assert record["agentview_detected_center_u"] == pytest.approx(pu)
        assert record["agentview_detected_bottom_u"] == pytest.approx(pu)
        assert record["agentview_detected_bottom_v"] == pytest.approx(pv)
        # Bottom pixel error should be 0 (projected matches bottom)
        assert record["agentview_bottom_pixel_error"] < 1e-10

    def test_record_no_detection(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _make_record,
        )
        from vlm_nsm.perception.robosuite_camera_utils import CameraInfo

        cam_info = CameraInfo(
            K=AGENTVIEW_K, T_camera_world=AGENTVIEW_T,
            width=512, height=512, camera_name="agentview",
        )

        oracle_xyz = np.array([0.05, -0.02, 0.82], dtype=np.float64)
        record = _make_record(
            0, "cube1", "red", oracle_xyz,
            None, cam_info, "agentview", 0.8,
        )

        assert record["agentview_projected_valid"] is True
        assert record["agentview_detected"] is False
        assert "agentview_center_pixel_error" not in record

    def test_record_behind_camera(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _make_record,
        )
        from vlm_nsm.perception.robosuite_camera_utils import CameraInfo

        cam_info = CameraInfo(
            K=AGENTVIEW_K, T_camera_world=AGENTVIEW_T,
            width=512, height=512, camera_name="agentview",
        )

        # Point behind camera
        oracle_xyz = np.array([2.0, 0.0, 1.35], dtype=np.float64)
        record = _make_record(
            0, "cube1", "red", oracle_xyz,
            None, cam_info, "agentview", 0.8,
        )

        assert record["agentview_projected_valid"] is False
        assert record["agentview_projected_failure"] == "behind_camera"


class TestAggregate:
    def test_empty_records(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _aggregate,
        )
        metrics = _aggregate([])
        assert metrics["num_objects_total"] == 0
        assert metrics["cameras"] == []

    def test_with_center_errors(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            _aggregate,
        )
        records = [
            {
                "frame_idx": 0,
                "agentview_projected_valid": True,
                "agentview_detected": True,
                "agentview_center_pixel_error": 5.0,
                "agentview_bottom_pixel_error": 3.0,
            },
            {
                "frame_idx": 1,
                "agentview_projected_valid": True,
                "agentview_detected": True,
                "agentview_center_pixel_error": 15.0,
                "agentview_bottom_pixel_error": 7.0,
            },
        ]
        metrics = _aggregate(records)
        assert len(metrics["cameras"]) == 1
        cm = metrics["cameras"][0]
        assert cm["center_pixel_error"]["mean"] == pytest.approx(10.0)
        assert cm["center_pixel_error"]["median"] == pytest.approx(10.0)
        assert cm["bottom_pixel_error"]["mean"] == pytest.approx(5.0)


class TestLoadCalibration:
    def test_load_valid_file(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            load_calibration,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "calib.json")
            with open(path, "w") as f:
                json.dump({
                    "K": AGENTVIEW_K.tolist(),
                    "T_camera_world": AGENTVIEW_T.tolist(),
                    "width": 512, "height": 512,
                    "camera_name": "agentview",
                }, f)

            info = load_calibration(path)
            assert info.camera_name == "agentview"
            assert info.width == 512
            assert np.allclose(info.K, AGENTVIEW_K)
            assert np.allclose(info.T_camera_world, AGENTVIEW_T)

    def test_load_missing_file(self):
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            load_calibration,
        )
        with pytest.raises(FileNotFoundError):
            load_calibration("/nonexistent/calib.json")

    def test_single_camera_mode(self):
        """check_consistency with only agentview calibration (no secondary)."""
        from vlm_nsm.experiments.debug_camera_projection_consistency import (
            check_consistency,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a minimal eval input
            eval_input = Path(tmpdir) / "eval_input.json"
            eval_input.write_text(json.dumps([{
                "instruction": "test",
                "image_path": "/tmp/test.png",
                "scene_info": {
                    "objects": [
                        {"object_id": "cube1", "color": "red",
                         "position": {"x": 0.1, "y": 0.0, "z": 0.8}},
                    ],
                },
            }]))

            # Write calibration
            calib_path = Path(tmpdir) / "calib_agent.json"
            calib_path.write_text(json.dumps({
                "K": AGENTVIEW_K.tolist(),
                "T_camera_world": AGENTVIEW_T.tolist(),
                "width": 512, "height": 512,
                "camera_name": "agentview",
            }))

            output_metrics = str(Path(tmpdir) / "metrics.json")
            output_jsonl = str(Path(tmpdir) / "debug.jsonl")

            # Mock detector (returns empty detections to avoid model dependency)
            with mock.patch(
                "vlm_nsm.experiments.debug_camera_projection_consistency."
                "load_detector"
            ) as mock_load:
                mock_load.return_value = None  # no detector → no detections

                metrics = check_consistency(
                    eval_input_path=str(eval_input),
                    calibration_agentview=str(calib_path),
                    calibration_secondary=None,
                    detector_model_path="/tmp/nonexistent.pt",
                    output_metrics=output_metrics,
                    output_debug_jsonl=output_jsonl,
                )

            assert metrics["num_instructions"] == 1
            assert "agentview" in str(metrics.get("cameras", []))

            # Check output files exist
            assert Path(output_metrics).exists()
            assert Path(output_jsonl).exists()

            # Check JSONL content
            with open(output_jsonl) as f:
                records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == 1
            assert records[0]["agentview_projected_valid"] is True
            assert records[0]["agentview_detected"] is False
