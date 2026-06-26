"""Tests for summarize_fixed_calibration_pose_eval.py — pure logic, no GPU."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestRecommend:
    def test_stereo_pass(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _recommend,
        )
        single = {"agentview": {"status": "PASS"}}
        stereo = {"status": "PASS"}
        spatial = {}
        assert _recommend(single, stereo, spatial) == "PROCEED_STEREO_QWEN_EVAL"

    def test_single_strong_pass_no_stereo(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _recommend,
        )
        single = {"agentview": {"status": "STRONG_PASS"}}
        stereo = {"status": "FAIL"}
        spatial = {}
        assert _recommend(single, stereo, spatial) == "ENABLE_TABLE_PROJECTION_EXPERIMENTAL"

    def test_keep_image2d(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _recommend,
        )
        single = {"agentview": {"status": "FAIL"}}
        stereo = {"status": "FAIL"}
        spatial = {}
        assert _recommend(single, stereo, spatial) == "KEEP_IMAGE2D"


class TestSummarizeSingleCamera:
    def test_with_valid_data(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _summarize_single_camera,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            # Create agentview center result
            data = {
                "mean_xy_error": 0.022, "median_xy_error": 0.022,
                "p90_xy_error": 0.023, "max_xy_error": 0.025,
                "num_objects": 90, "failure_count": 0,
            }
            (rd / "table_projection_pose_error_agentview_center_fixed_calib.json").write_text(
                json.dumps(data),
            )
            data_bc = {
                "mean_xy_error": 0.077, "median_xy_error": 0.076,
                "p90_xy_error": 0.085, "max_xy_error": 0.090,
                "num_objects": 90, "failure_count": 0,
            }
            (rd / "table_projection_pose_error_agentview_bottom_center_fixed_calib.json").write_text(
                json.dumps(data_bc),
            )
            data_wb = {
                "mean_xy_error": 0.048, "median_xy_error": 0.048,
                "p90_xy_error": 0.052, "max_xy_error": 0.055,
                "num_objects": 90, "failure_count": 0,
            }
            (rd / "table_projection_pose_error_agentview_weighted_bottom_fixed_calib.json").write_text(
                json.dumps(data_wb),
            )

            result = _summarize_single_camera(rd)
            assert "agentview" in result
            ag = result["agentview"]
            assert ag["best_mode"] == "center"
            assert ag["best_median_xy_error"] == pytest.approx(0.022)
            assert ag["status"] == "PASS"

    def test_missing_data(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _summarize_single_camera,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            result = _summarize_single_camera(rd)
            for cam in ["agentview", "frontview", "birdview"]:
                assert result[cam]["modes"]["center"].get("error") == "no_data"
                assert result[cam]["status"] == "NO_DATA"


class TestSummarizeStereo:
    def test_with_valid_data(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _summarize_stereo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            data = {
                "camera_a": "agentview", "camera_b": "frontview",
                "num_frames": 45, "num_valid_triangulations": 90,
                "median_xy_error": 0.015, "mean_xy_error": 0.014,
                "p90_xy_error": 0.025, "triangulation_residual_median": 0.005,
                "triangulation_residual_mean": 0.006,
                "per_axis_abs_error_median": {"x": 0.008, "y": 0.009, "z": 0.010},
            }
            (rd / "dual_camera_pose_error_agentview_frontview_fixed_calib.json").write_text(
                json.dumps(data),
            )
            result = _summarize_stereo(rd)
            assert result["status"] == "PASS"
            assert result["median_xy_error"] == pytest.approx(0.015)

    def test_missing(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _summarize_stereo,
        )
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            result = _summarize_stereo(rd)
            assert "error" in result


class TestGenerateMarkdown:
    def test_generates_complete_report(self):
        from vlm_nsm.experiments.summarize_fixed_calibration_pose_eval import (
            _generate_markdown,
        )
        summary = {
            "single_camera": {
                "agentview": {
                    "modes": {
                        "center": {"mean_xy_error": 0.022, "median_xy_error": 0.022,
                                   "p90_xy_error": 0.023, "max_xy_error": 0.025},
                        "bottom_center": {"mean_xy_error": 0.077, "median_xy_error": 0.076,
                                          "p90_xy_error": 0.085, "max_xy_error": 0.090},
                        "weighted_bottom": {"mean_xy_error": 0.048, "median_xy_error": 0.048,
                                            "p90_xy_error": 0.052, "max_xy_error": 0.055},
                    },
                    "best_mode": "center",
                    "best_median_xy_error": 0.022,
                    "status": "PASS",
                },
                "frontview": {
                    "modes": {
                        "center": {"error": "no_data"},
                        "bottom_center": {"error": "no_data"},
                        "weighted_bottom": {"error": "no_data"},
                    },
                    "best_mode": None,
                    "status": "NO_DATA",
                },
                "birdview": {
                    "modes": {
                        "center": {"error": "no_data"},
                        "bottom_center": {"error": "no_data"},
                        "weighted_bottom": {"error": "no_data"},
                    },
                    "best_mode": None,
                    "status": "NO_DATA",
                },
            },
            "stereo": {
                "camera_pair": "agentview+frontview",
                "num_frames": 45,
                "num_valid_triangulations": 90,
                "median_xy_error": 0.015,
                "mean_xy_error": 0.014,
                "p90_xy_error": 0.025,
                "triangulation_residual_median": 0.005,
                "status": "PASS",
            },
            "spatial_relations": {
                "overall_accuracy": 0.95,
                "left_right_accuracy": 0.97,
                "front_back_accuracy": 0.93,
                "total_pairs": 45,
            },
            "recommendation": "PROCEED_STEREO_QWEN_EVAL",
        }
        md = _generate_markdown(summary)
        assert "Stage 18-F2" in md
        assert "agentview" in md
        assert "center" in md
        assert "0.0220" in md
        assert "PASS" in md
        assert "PROCEED_STEREO_QWEN_EVAL" in md
