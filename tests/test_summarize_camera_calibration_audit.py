"""Tests for summarize_camera_calibration_audit.py — pure logic, no GPU required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fake input data matching real file formats
# ---------------------------------------------------------------------------


def _make_projection_consistency(camera_name: str, center_median: float,
                                  bottom_median: float,
                                  num_valid: int = 90,
                                  num_detected: int = 90) -> dict:
    return {
        "num_objects_total": num_valid,
        "cameras": [{
            "camera": camera_name,
            "num_valid_projections": num_valid,
            "num_detected": num_detected,
            "center_pixel_error": {
                "mean": center_median + 1.0,
                "median": center_median,
                "p90": center_median + 10.0,
                "max": center_median + 50.0,
                "min": center_median - 5.0,
                "std": 10.0,
            },
            "bottom_pixel_error": {
                "mean": bottom_median + 1.0,
                "median": bottom_median,
                "p90": bottom_median + 10.0,
                "max": bottom_median + 50.0,
                "min": bottom_median - 5.0,
                "std": 10.0,
            },
        }],
        "verdicts": [f"{camera_name}: BAD (median center error {center_median:.1f} px)"],
    }


def _make_convention_search(camera_name: str,
                             best_name: str = "row_major_cam2world_yflip",
                             best_center_median: float = 5.9,
                             best_bottom_median: float = 26.4,
                             best_y_flip: bool = True,
                             best_z_flip: bool = False,
                             vanilla_center_median: float = 152.0,
                             vanilla_bottom_median: float = 166.0) -> dict:
    top = [{
        "candidate": best_name,
        "reshape_mode": "row_major",
        "R_direction": "cam2world",
        "y_flip": best_y_flip,
        "z_flip": best_z_flip,
        "valid_projections": 90,
        "behind_camera": 0,
        "center_pixel_error_mean": best_center_median + 0.4,
        "center_pixel_error_median": best_center_median,
        "center_pixel_error_p90": best_center_median + 1.4,
        "center_pixel_error_max": best_center_median + 1.8,
        "center_pixel_error_std": 0.9,
        "bottom_pixel_error_mean": best_bottom_median + 0.3,
        "bottom_pixel_error_median": best_bottom_median,
        "bottom_pixel_error_p90": best_bottom_median + 3.9,
        "bottom_pixel_error_max": best_bottom_median + 10.0,
        "bottom_pixel_error_std": 3.6,
    }]

    # Build full all_candidates list with 16 entries
    all_candidates = []
    vanilla_inserted = False
    idx = 0
    for reshape in ["row_major", "col_major"]:
        for direction in ["cam2world", "world2cam"]:
            for yf in [False, True]:
                for zf in [False, True]:
                    name = f"{reshape}_{direction}"
                    if yf:
                        name += "_yflip"
                    if zf:
                        name += "_zflip"
                    if not yf and not zf:
                        name += "_vanilla"

                    is_vanilla = "vanilla" in name and "yflip" not in name

                    entry = {
                        "candidate": name,
                        "reshape_mode": reshape,
                        "R_direction": direction,
                        "y_flip": yf,
                        "z_flip": zf,
                        "valid_projections": 90,
                        "behind_camera": 0,
                    }
                    if name == best_name:
                        entry["center_pixel_error_median"] = best_center_median
                        entry["bottom_pixel_error_median"] = best_bottom_median
                    elif is_vanilla:
                        entry["center_pixel_error_median"] = vanilla_center_median
                        entry["bottom_pixel_error_median"] = vanilla_bottom_median
                    else:
                        entry["center_pixel_error_median"] = 200.0 + idx
                        entry["bottom_pixel_error_median"] = 200.0 + idx

                    all_candidates.append(entry)
                    idx += 1

    # Sort by center error so best is first
    all_candidates.sort(
        key=lambda r: r.get("center_pixel_error_median", float("inf")),
    )

    return {
        "camera_name": camera_name,
        "num_pairs": 90,
        "num_candidates": 16,
        "top_candidates": top,
        "all_candidates": all_candidates,
    }


def _make_world_geometry(table_z: float = 0.775,
                          surface_z: float | None = None,
                          cube1_z: float = 0.83,
                          cube2_z: float = 0.835) -> dict:
    if surface_z is None:
        surface_z = table_z + 0.025
    return {
        "table": {
            "body_name": "table",
            "position": {"x": 0.0, "y": 0.0, "z": table_z},
            "geoms": [{
                "geom_name": "table_collision",
                "position": {"x": 0.0, "y": 0.0, "z": table_z},
                "size": [0.4, 0.4, 0.025],
                "type": 6,
            }],
        },
        "cubes": {
            "cube1": {"body_name": "cube1", "position": {"x": 0.0, "y": 0.0, "z": cube1_z}},
            "cube2": {"body_name": "cube2", "position": {"x": 0.0, "y": 0.1, "z": cube2_z}},
        },
    }


# ---------------------------------------------------------------------------
# Tests: _get_vanilla_pixel_errors
# ---------------------------------------------------------------------------


class TestGetVanillaPixelErrors:
    def test_extracts_vanilla_errors(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_pixel_errors,
        )
        proj = _make_projection_consistency("agentview", 152.0, 166.0)
        result = _get_vanilla_pixel_errors(proj, "agentview")
        assert result["center_pixel_median"] == pytest.approx(152.0)
        assert result["bottom_pixel_median"] == pytest.approx(166.0)
        assert result["center_pixel_mean"] == pytest.approx(153.0)
        assert result["num_valid"] == 90
        assert result["num_detected"] == 90

    def test_returns_none_for_none_input(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_pixel_errors,
        )
        result = _get_vanilla_pixel_errors(None, "agentview")
        assert result["center_pixel_median"] is None
        assert result["bottom_pixel_median"] is None

    def test_returns_none_for_no_matching_camera(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_pixel_errors,
        )
        proj = _make_projection_consistency("agentview", 152.0, 166.0)
        result = _get_vanilla_pixel_errors(proj, "frontview")
        assert result["center_pixel_median"] is None
        assert result["bottom_pixel_median"] is None


# ---------------------------------------------------------------------------
# Tests: _get_best_convention
# ---------------------------------------------------------------------------


class TestGetBestConvention:
    def test_extracts_best_candidate(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_best_convention,
        )
        conv = _make_convention_search("agentview", best_name="row_major_cam2world_yflip",
                                        best_center_median=5.9, best_bottom_median=26.4)
        best = _get_best_convention(conv)
        assert best["name"] == "row_major_cam2world_yflip"
        assert best["reshape_mode"] == "row_major"
        assert best["R_direction"] == "cam2world"
        assert best["y_flip"] is True
        assert best["z_flip"] is False
        assert best["center_pixel_median"] == pytest.approx(5.9)
        assert best["bottom_pixel_median"] == pytest.approx(26.4)
        assert best["center_pixel_p90"] == pytest.approx(7.3)
        assert best["valid_projections"] == 90

    def test_returns_error_for_none_input(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_best_convention,
        )
        best = _get_best_convention(None)
        assert best == {"error": "no_data"}

    def test_returns_error_for_no_candidates(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_best_convention,
        )
        best = _get_best_convention({"top_candidates": []})
        assert best == {"error": "no_candidates"}


# ---------------------------------------------------------------------------
# Tests: _get_vanilla_rank
# ---------------------------------------------------------------------------


class TestGetVanillaRank:
    def test_finds_vanilla_rank(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_rank,
        )
        conv = _make_convention_search("agentview")
        rank = _get_vanilla_rank(conv)
        # "row_major_cam2world_vanilla" should appear somewhere in the sorted list
        assert rank is not None
        assert rank >= 1

    def test_returns_none_for_none_input(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_rank,
        )
        assert _get_vanilla_rank(None) is None

    def test_vanilla_rank_greater_than_1_when_better_exists(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _get_vanilla_rank,
        )
        conv = _make_convention_search("agentview",
                                        best_name="row_major_cam2world_yflip",
                                        best_center_median=5.9,
                                        vanilla_center_median=152.0)
        rank = _get_vanilla_rank(conv)
        # Vanilla should not be rank 1 since yflip variant is better
        assert rank is not None
        assert rank > 1


# ---------------------------------------------------------------------------
# Tests: _determine_status
# ---------------------------------------------------------------------------


class TestDetermineStatus:
    def test_pass_with_good_center_and_bottom(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _determine_status,
        )
        best = {"center_pixel_median": 5.9, "bottom_pixel_median": 26.4}
        vanilla = {}
        assert _determine_status(best, vanilla) == "PASS"

    def test_pass_with_only_center_good(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _determine_status,
        )
        best = {"center_pixel_median": 15.0, "bottom_pixel_median": 60.0}
        vanilla = {}
        assert _determine_status(best, vanilla) == "PASS"

    def test_marginal(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _determine_status,
        )
        best = {"center_pixel_median": 35.0, "bottom_pixel_median": 80.0}
        vanilla = {}
        assert _determine_status(best, vanilla) == "MARGINAL"

    def test_fail_with_high_errors(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _determine_status,
        )
        best = {"center_pixel_median": 150.0, "bottom_pixel_median": 160.0}
        vanilla = {}
        assert _determine_status(best, vanilla) == "FAIL"

    def test_fail_with_none_errors(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _determine_status,
        )
        best = {"center_pixel_median": None, "bottom_pixel_median": None}
        vanilla = {}
        assert _determine_status(best, vanilla) == "FAIL"


# ---------------------------------------------------------------------------
# Tests: _diagnose
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_consistent_convention_across_cameras(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"name": "row_major_cam2world_yflip"},
                "status": "PASS",
            },
            "frontview": {
                "best_convention": {"name": "row_major_cam2world_yflip"},
                "status": "PASS",
            },
        }
        diag = _diagnose(cameras)
        assert any("CONSISTENT_CONVENTION_FOUND" in d for d in diag["diagnoses"])
        assert diag["recommendation"] == "PROCEED_STEREO_REPAIR"

    def test_keep_image2d_when_all_fail(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"error": "no_data"},
                "status": "FAIL",
            },
            "frontview": {
                "best_convention": {"error": "no_data"},
                "status": "FAIL",
            },
            "birdview": {
                "best_convention": {"error": "no_data"},
                "status": "FAIL",
            },
        }
        diag = _diagnose(cameras)
        assert "DATASET_CAMERA_MISMATCH" in diag["diagnoses"]
        assert diag["recommendation"] == "KEEP_IMAGE2D"

    def test_fix_extrinsics_when_marginal(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"name": "row_major_cam2world_yflip"},
                "status": "MARGINAL",
            },
            "frontview": {
                "best_convention": {"name": "row_major_cam2world_zflip"},
                "status": "FAIL",
            },
        }
        diag = _diagnose(cameras)
        assert diag["recommendation"] == "FIX_EXTRINSICS"

    def test_table_z_suspect(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"name": "row_major_cam2world_yflip"},
                "status": "PASS",
            },
            "_world_geometry": {"table_body_z": 1.2},
        }
        diag = _diagnose(cameras)
        assert any("TABLE_Z_SUSPECT" in d for d in diag["diagnoses"])

    def test_improvement_diagnosis(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"name": "row_major_cam2world_yflip"},
                "status": "PASS",
                "improvement_factor": 26.0,
            },
        }
        diag = _diagnose(cameras)
        assert any("CAMERA_CONVENTION_SUSPECT" in d for d in diag["diagnoses"])

    def test_falls_back_to_keep_image2d_with_no_diagnoses(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _diagnose,
        )
        cameras = {
            "agentview": {
                "best_convention": {"name": "row_major"},
                "status": "PASS",
            },
        }
        diag = _diagnose(cameras)
        # Single camera can't establish consistent convention
        # so it won't trigger PROCEED_STEREO_REPAIR
        assert diag["recommendation"] in ("KEEP_IMAGE2D", "FIX_EXTRINSICS")


# ---------------------------------------------------------------------------
# Tests: _generate_markdown
# ---------------------------------------------------------------------------


class TestGenerateMarkdown:
    def test_generates_markdown_with_pass_status(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _generate_markdown,
        )
        summary = {
            "cameras": {
                "agentview": {
                    "best_convention": {
                        "name": "row_major_cam2world_yflip",
                        "center_pixel_median": 5.9,
                        "bottom_pixel_median": 26.4,
                    },
                    "status": "PASS",
                    "vanilla_rank": 4,
                    "improvement_factor": 26.0,
                    "vanilla_center_median": 152.0,
                },
                "frontview": {
                    "best_convention": {
                        "name": "row_major_cam2world_yflip",
                        "center_pixel_median": 4.5,
                        "bottom_pixel_median": 25.1,
                    },
                    "status": "PASS",
                    "vanilla_rank": 5,
                    "improvement_factor": 33.6,
                    "vanilla_center_median": 151.0,
                },
                "birdview": {
                    "best_convention": {"error": "no_data"},
                },
            },
            "world_geometry": {
                "table_body_z": 0.775,
                "table_surface_z": 0.8,
            },
            "global_diagnosis": [
                "CONSISTENT_CONVENTION_FOUND: row_major_cam2world_yflip",
            ],
            "recommendation": "PROCEED_STEREO_REPAIR",
        }
        md = _generate_markdown(summary)
        assert "Stage 18-FixCam-R" in md
        assert "agentview" in md
        assert "frontview" in md
        assert "birdview" in md
        assert "row_major_cam2world_yflip" in md
        assert "5.9" in md
        assert "PASS" in md
        assert "no_data" in md
        assert "PROCEED_STEREO_REPAIR" in md

    def test_handles_no_data_camera(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _generate_markdown,
        )
        summary = {
            "cameras": {
                "agentview": {
                    "best_convention": {"error": "no_candidates"},
                    "status": "FAIL",
                },
                "frontview": {
                    "best_convention": {"error": "no_data"},
                },
                "birdview": {
                    "best_convention": {"error": "no_data"},
                },
            },
            "world_geometry": {},
            "global_diagnosis": ["KEEP_IMAGE2D"],
            "recommendation": "KEEP_IMAGE2D",
        }
        md = _generate_markdown(summary)
        assert "no_candidates" in md or "N/A" in md
        assert "KEEP_IMAGE2D" in md

    def test_includes_stereo_disabled_message_when_not_proceed(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            _generate_markdown,
        )
        summary = {
            "cameras": {},
            "world_geometry": {},
            "global_diagnosis": ["KEEP_IMAGE2D"],
            "recommendation": "KEEP_IMAGE2D",
        }
        md = _generate_markdown(summary)
        assert "disabled" in md.lower()


# ---------------------------------------------------------------------------
# Tests: summarize_audit (end-to-end with temp files)
# ---------------------------------------------------------------------------


class TestSummarizeAudit:
    def test_full_summary_with_temp_files(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            summarize_audit,
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()

            # Write fake input files
            (results_dir / "robosuite_world_geometry_inspection.json").write_text(
                json.dumps(_make_world_geometry(0.775)),
            )
            (results_dir / "projection_consistency_agentview.json").write_text(
                json.dumps(_make_projection_consistency("agentview", 152.0, 166.0)),
            )
            (results_dir / "projection_consistency_frontview.json").write_text(
                json.dumps(_make_projection_consistency("frontview", 151.0, 145.0)),
            )
            (results_dir / "projection_consistency_birdview.json").write_text(
                json.dumps(_make_projection_consistency("birdview", 200.0, 200.0)),
            )
            (results_dir / "camera_convention_search_agentview.json").write_text(
                json.dumps(_make_convention_search("agentview")),
            )
            (results_dir / "camera_convention_search_frontview.json").write_text(
                json.dumps(_make_convention_search("frontview")),
            )
            (results_dir / "camera_convention_search_birdview.json").write_text(
                json.dumps({"camera_name": "birdview", "num_pairs": 0,
                             "top_candidates": [], "all_candidates": []}),
            )

            output = str(results_dir / "summary.json")
            md_output = str(results_dir / "summary.md")

            result = summarize_audit(
                results_dir=str(results_dir),
                output=output,
                markdown_output=md_output,
            )

            # Check result structure
            assert "cameras" in result
            assert "agentview" in result["cameras"]
            assert "frontview" in result["cameras"]
            assert "birdview" in result["cameras"]
            assert "world_geometry" in result
            assert "global_diagnosis" in result
            assert "recommendation" in result

            # Check agentview specifics
            agent = result["cameras"]["agentview"]
            assert agent["best_convention"]["name"] == "row_major_cam2world_yflip"
            assert agent["status"] == "PASS"
            assert agent["improvement_factor"] is not None
            assert agent["improvement_factor"] > 20

            # Check frontview
            front = result["cameras"]["frontview"]
            assert front["best_convention"]["y_flip"] is True

            # Check birdview has error
            bird = result["cameras"]["birdview"]
            assert "error" in bird.get("best_convention", {})

            # Check world geometry
            assert result["world_geometry"]["table_body_z"] == pytest.approx(0.775)

            # Check files written
            assert Path(output).exists()
            assert Path(md_output).exists()

            # Verify JSON is valid
            loaded = json.loads(Path(output).read_text())
            assert loaded == result

    def test_summary_with_missing_files_does_not_crash(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            summarize_audit,
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "empty_results"
            results_dir.mkdir()

            output = str(results_dir / "summary.json")

            result = summarize_audit(
                results_dir=str(results_dir),
                output=output,
                markdown_output=None,
            )

            # Should handle missing files gracefully
            assert "cameras" in result
            for cam in ["agentview", "frontview", "birdview"]:
                assert cam in result["cameras"]

    def test_improvement_factor_computation(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            summarize_audit,
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()

            (results_dir / "robosuite_world_geometry_inspection.json").write_text(
                json.dumps(_make_world_geometry(0.775)),
            )
            (results_dir / "projection_consistency_agentview.json").write_text(
                json.dumps(_make_projection_consistency("agentview", 152.0, 166.0)),
            )
            (results_dir / "projection_consistency_frontview.json").write_text(
                json.dumps(_make_projection_consistency("frontview", 151.0, 145.0)),
            )
            (results_dir / "projection_consistency_birdview.json").write_text(
                json.dumps(_make_projection_consistency("birdview", 200.0, 200.0)),
            )
            (results_dir / "camera_convention_search_agentview.json").write_text(
                json.dumps(_make_convention_search("agentview")),
            )
            (results_dir / "camera_convention_search_frontview.json").write_text(
                json.dumps(_make_convention_search("frontview")),
            )
            (results_dir / "camera_convention_search_birdview.json").write_text(
                json.dumps({"camera_name": "birdview", "num_pairs": 0,
                             "top_candidates": [], "all_candidates": []}),
            )

            output = str(results_dir / "summary.json")

            result = summarize_audit(
                results_dir=str(results_dir),
                output=output,
            )

            agent_imp = result["cameras"]["agentview"]["improvement_factor"]
            # 152.0 / 5.9 ≈ 25.76
            assert agent_imp is not None
            assert 25.0 < agent_imp < 26.0

    def test_consistent_convention_triggers_stereo_repair(self):
        from vlm_nsm.experiments.summarize_camera_calibration_audit import (
            summarize_audit,
        )
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            results_dir.mkdir()

            (results_dir / "robosuite_world_geometry_inspection.json").write_text(
                json.dumps(_make_world_geometry(0.775)),
            )
            (results_dir / "projection_consistency_agentview.json").write_text(
                json.dumps(_make_projection_consistency("agentview", 152.0, 166.0)),
            )
            (results_dir / "projection_consistency_frontview.json").write_text(
                json.dumps(_make_projection_consistency("frontview", 151.0, 145.0)),
            )
            (results_dir / "projection_consistency_birdview.json").write_text(
                json.dumps(_make_projection_consistency("birdview", 200.0, 200.0)),
            )

            # Both share the same best convention
            conv_agent = _make_convention_search("agentview",
                                                  best_name="row_major_cam2world_yflip",
                                                  best_center_median=5.9)
            conv_front = _make_convention_search("frontview",
                                                  best_name="row_major_cam2world_yflip",
                                                  best_center_median=4.5)
            conv_bird = {"camera_name": "birdview", "num_pairs": 0,
                         "top_candidates": [], "all_candidates": []}

            (results_dir / "camera_convention_search_agentview.json").write_text(
                json.dumps(conv_agent),
            )
            (results_dir / "camera_convention_search_frontview.json").write_text(
                json.dumps(conv_front),
            )
            (results_dir / "camera_convention_search_birdview.json").write_text(
                json.dumps(conv_bird),
            )

            output = str(results_dir / "summary.json")

            result = summarize_audit(
                results_dir=str(results_dir),
                output=output,
            )

            assert result["recommendation"] == "PROCEED_STEREO_REPAIR"
            assert any("CONSISTENT_CONVENTION_FOUND" in d
                       for d in result["global_diagnosis"])
