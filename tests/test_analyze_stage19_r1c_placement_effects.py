"""Tests for analyze_stage19_r1c_placement_effects.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.analyze_stage19_r1c_placement_effects import (
    _compute_mean_poses,
    _count_failure_reasons,
    _get_recommendation,
    analyze,
    generate_report,
    load_diagnostics,
)


# ---- helpers ----

def _make_episode(episode_id="ep0", goal_id="red_on_green", task_success=True,
                  seed=0, failure_reason=None,
                  initial_cube1_pose=None, initial_cube2_pose=None,
                  initial_top_pose=None, initial_bottom_pose=None,
                  initial_top_bottom_distance_xy=None,
                  initial_relative_xy_top_minus_bottom=None,
                  top_region="center", bottom_region="center"):
    """Build a synthetic diagnostics episode dict."""
    ep = {
        "episode_id": episode_id,
        "goal_id": goal_id,
        "task_success": task_success,
        "seed": seed,
        "top_region": top_region,
        "bottom_region": bottom_region,
    }
    if failure_reason:
        ep["failure_reason"] = failure_reason
    else:
        ep["failure_reason"] = "none" if task_success else "place_failed"
    if initial_cube1_pose:
        ep["initial_cube1_pose"] = initial_cube1_pose
    if initial_cube2_pose:
        ep["initial_cube2_pose"] = initial_cube2_pose
    if initial_top_pose:
        ep["initial_top_pose"] = initial_top_pose
    if initial_bottom_pose:
        ep["initial_bottom_pose"] = initial_bottom_pose
    if initial_top_bottom_distance_xy is not None:
        ep["initial_top_bottom_distance_xy"] = initial_top_bottom_distance_xy
    if initial_relative_xy_top_minus_bottom is not None:
        ep["initial_relative_xy_top_minus_bottom"] = initial_relative_xy_top_minus_bottom
    return ep


# ---- load_diagnostics ----

def test_load_diagnostics_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        tmp = f.name
    result = load_diagnostics(tmp)
    assert result == []
    Path(tmp).unlink()


def test_load_diagnostics_basic():
    eps = [_make_episode("ep0"), _make_episode("ep1", task_success=False)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
        tmp = f.name
    result = load_diagnostics(tmp)
    assert len(result) == 2
    assert result[0]["episode_id"] == "ep0"
    assert result[1]["episode_id"] == "ep1"
    Path(tmp).unlink()


def test_load_diagnostics_skips_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(_make_episode("ep0")) + "\n")
        f.write("\n")
        f.write(json.dumps(_make_episode("ep1")) + "\n")
        tmp = f.name
    result = load_diagnostics(tmp)
    assert len(result) == 2
    Path(tmp).unlink()


# ---- _count_failure_reasons ----

def test_count_failure_reasons():
    eps = [
        _make_episode("s0", task_success=True, failure_reason="none"),
        _make_episode("f0", task_success=False, failure_reason="place_failed"),
        _make_episode("f1", task_success=False, failure_reason="place_failed"),
        _make_episode("f2", task_success=False, failure_reason="grasp_failed"),
    ]
    result = _count_failure_reasons(eps)
    assert result["none"] == 1
    assert result["place_failed"] == 2
    assert result["grasp_failed"] == 1


def test_count_failure_reasons_defaults_unknown():
    eps = [_make_episode("e0")]
    eps[0].pop("failure_reason")
    result = _count_failure_reasons(eps)
    assert result["unknown"] == 1


# ---- _compute_mean_poses ----

def test_compute_mean_poses_simple():
    eps = [
        _make_episode("s0", task_success=True,
                      initial_cube1_pose={"x": 0.0, "y": 0.0, "z": 0.8},
                      initial_cube2_pose={"x": 0.15, "y": 0.0, "z": 0.8},
                      initial_top_bottom_distance_xy=0.15),
        _make_episode("s1", task_success=True,
                      initial_cube1_pose={"x": 0.1, "y": 0.0, "z": 0.8},
                      initial_cube2_pose={"x": 0.2, "y": 0.0, "z": 0.8},
                      initial_top_bottom_distance_xy=0.10),
    ]
    result = _compute_mean_poses(eps)
    assert "success" in result
    assert "failure" in result  # key always present, may be empty
    assert result["failure"] == {}
    assert result["success"]["initial_cube1_pose"]["x"] == 0.05
    assert result["success"]["initial_cube1_pose"]["y"] == 0.0
    assert result["success"]["initial_top_bottom_distance_xy"] == 0.125


def test_compute_mean_poses_mixed():
    eps = [
        _make_episode("s0", task_success=True,
                      initial_top_bottom_distance_xy=0.15),
        _make_episode("f0", task_success=False,
                      initial_top_bottom_distance_xy=0.20),
    ]
    result = _compute_mean_poses(eps)
    assert result["success"]["initial_top_bottom_distance_xy"] == 0.15
    assert result["failure"]["initial_top_bottom_distance_xy"] == 0.20


def test_compute_mean_poses_handles_missing_keys():
    eps = [_make_episode("s0", task_success=True)]
    result = _compute_mean_poses(eps)
    assert result == {"success": {}, "failure": {}}


# ---- _get_recommendation ----

def test_recommendation_high():
    rec = _get_recommendation(0.85, 0.75, "placement_variance_within_tolerance")
    assert "Scripted-only is reliable enough" in rec
    assert "Proceed to Stage 19-D2" in rec


def test_recommendation_medium():
    rec = _get_recommendation(0.75, 0.60, "placement_sensitive_scripted_control")
    assert "acceptable but" in rec
    assert "filtering by initial placement distance" in rec


def test_recommendation_low():
    rec = _get_recommendation(0.60, 0.40, "goal_order_sensitive_scripted_control")
    assert "significant placement sensitivity" in rec
    assert "fix object placement positions" in rec


# ---- analyze ----

def test_analyze_all_success():
    eps = [_make_episode(f"ep{i}", task_success=True) for i in range(10)]
    result = analyze(eps)
    assert result["total_episodes"] == 10
    assert result["success_count"] == 10
    assert result["failure_count"] == 0
    assert result["overall_success_rate"] == 1.0


def test_analyze_all_failure():
    eps = [_make_episode(f"ep{i}", task_success=False, failure_reason="grasp_failed")
           for i in range(5)]
    result = analyze(eps)
    assert result["total_episodes"] == 5
    assert result["success_count"] == 0
    assert result["overall_success_rate"] == 0.0


def test_analyze_per_goal_stats():
    eps = [
        _make_episode("r0", goal_id="red_on_green", task_success=True),
        _make_episode("r1", goal_id="red_on_green", task_success=True),
        _make_episode("r2", goal_id="red_on_green", task_success=False),
        _make_episode("g0", goal_id="green_on_red", task_success=True),
        _make_episode("g1", goal_id="green_on_red", task_success=False),
    ]
    result = analyze(eps)
    pg = result["per_goal"]
    assert pg["red_on_green"]["total"] == 3
    assert pg["red_on_green"]["success"] == 2
    assert pg["red_on_green"]["rate"] == pytest.approx(2 / 3, abs=0.01)
    assert pg["green_on_red"]["total"] == 2
    assert pg["green_on_red"]["success"] == 1
    assert pg["green_on_red"]["rate"] == 0.5


def test_analyze_per_goal_includes_failure_details():
    eps = [
        _make_episode("g0", goal_id="green_on_red", task_success=False,
                      failure_reason="place_failed", seed=5,
                      top_region="center", bottom_region="center",
                      initial_top_bottom_distance_xy=0.12),
    ]
    result = analyze(eps)
    assert result["per_goal"]["green_on_red"]["total"] == 1
    assert result["per_goal"]["green_on_red"]["success"] == 0


def test_analyze_distance_bins():
    eps = [
        _make_episode("e0", task_success=True, initial_top_bottom_distance_xy=0.03),
        _make_episode("e1", task_success=True, initial_top_bottom_distance_xy=0.04),
        _make_episode("e2", task_success=False, initial_top_bottom_distance_xy=0.12),
        _make_episode("e3", task_success=True, initial_top_bottom_distance_xy=0.25),
    ]
    result = analyze(eps)
    bins = result["distance_xy_analysis"]
    assert len(bins) >= 3
    # First bin (0.00-0.05): 2 success
    bin0 = next(b for b in bins if b["bin"] == "0.00-0.05m")
    assert bin0["total"] == 2
    assert bin0["success"] == 2
    assert bin0["rate"] == 1.0


def test_analyze_distance_bins_skips_empty():
    eps = [_make_episode(f"e{i}", initial_top_bottom_distance_xy=0.03)
           for i in range(3)]
    result = analyze(eps)
    bins = result["distance_xy_analysis"]
    # Only bin 0.00-0.05 should be present
    assert len(bins) == 1
    assert bins[0]["bin"] == "0.00-0.05m"


def test_analyze_region_analysis():
    eps = [
        _make_episode("e0", task_success=True, top_region="center"),
        _make_episode("e1", task_success=True, top_region="center"),
        _make_episode("e2", task_success=False, top_region="front-left"),
    ]
    result = analyze(eps)
    ra = result["region_analysis"]
    assert ra["center"]["total"] == 2
    assert ra["center"]["success"] == 2
    assert ra["center"]["rate"] == 1.0
    assert ra["front-left"]["total"] == 1
    assert ra["front-left"]["success"] == 0


def test_analyze_region_analysis_unknown_default():
    eps = [_make_episode("e0")]
    eps[0].pop("top_region")
    result = analyze(eps)
    assert "unknown" in result["region_analysis"]


def test_analyze_xy_sign_analysis():
    eps = [
        _make_episode("e0", task_success=True,
                      initial_relative_xy_top_minus_bottom={"dx": 0.1, "dy": 0.05}),
        _make_episode("e1", task_success=False,
                      initial_relative_xy_top_minus_bottom={"dx": -0.1, "dy": -0.05}),
    ]
    result = analyze(eps)
    xy = result["xy_sign_analysis"]
    assert "top_right_bottom_back" in xy
    assert xy["top_right_bottom_back"]["total"] == 1
    assert xy["top_right_bottom_back"]["success"] == 1
    assert "top_left_bottom_front" in xy
    assert xy["top_left_bottom_front"]["total"] == 1
    assert xy["top_left_bottom_front"]["success"] == 0


def test_analyze_skips_xy_when_rel_is_none():
    eps = [_make_episode("e0")]
    result = analyze(eps)
    assert result["xy_sign_analysis"] == {}


def test_analyze_seed_clusters():
    eps = [
        _make_episode("e0", task_success=True, seed=0),
        _make_episode("e1", task_success=False, seed=0),
        _make_episode("e2", task_success=True, seed=1),
    ]
    result = analyze(eps)
    sc = result["seed_clusters"]
    assert sc["0"]["total"] == 2
    assert sc["0"]["success"] == 1
    assert sc["1"]["total"] == 1
    assert sc["1"]["success"] == 1


def test_analyze_failure_reasons():
    eps = [
        _make_episode("e0", task_success=False, failure_reason="grasp_failed"),
        _make_episode("e1", task_success=False, failure_reason="place_failed"),
        _make_episode("e2", task_success=True, failure_reason="none"),
    ]
    result = analyze(eps)
    assert result["failure_reasons"]["grasp_failed"] == 1
    assert result["failure_reasons"]["place_failed"] == 1
    assert result["failure_reasons"]["none"] == 1


def test_analyze_root_cause_hint_placement_sensitive():
    """When green_on_red failures cluster in <=2 top regions, hint is placement_sensitive."""
    eps = []
    for i in range(10):
        eps.append(_make_episode(f"r{i}", goal_id="green_on_red",
                                 task_success=False, top_region="front-left"))
    for i in range(10):
        eps.append(_make_episode(f"red{i}", goal_id="red_on_green",
                                 task_success=True))
    result = analyze(eps)
    assert result["root_cause_hint"] == "placement_sensitive_scripted_control"


def test_analyze_root_cause_hint_variance():
    """When green rate >= 0.70, hint is variance_within_tolerance."""
    eps = []
    for i in range(8):
        eps.append(_make_episode(f"g{i}", goal_id="green_on_red", task_success=True,
                                 top_region=["fl", "c", "fr", "bl", "br"][i % 5]))
    for i in range(2):
        eps.append(_make_episode(f"gf{i}", goal_id="green_on_red", task_success=False,
                                 top_region="front-right"))
    for i in range(10):
        eps.append(_make_episode(f"r{i}", goal_id="red_on_green", task_success=True))
    result = analyze(eps)
    assert result["root_cause_hint"] == "placement_variance_within_tolerance"


def test_analyze_green_on_red_failure_clusters():
    eps = [
        _make_episode("g0", goal_id="green_on_red", task_success=False,
                      top_region="front-left", bottom_region="back-right"),
        _make_episode("g1", goal_id="green_on_red", task_success=False,
                      top_region="front-left", bottom_region="back-right"),
        _make_episode("g2", goal_id="green_on_red", task_success=False,
                      top_region="center", bottom_region="center"),
    ]
    for i in range(5):
        eps.append(_make_episode(f"r{i}", goal_id="red_on_green", task_success=True))
    result = analyze(eps)
    clusters = result["green_on_red_failure_region_clusters"]
    assert "top=front-left_bottom=back-right" in clusters
    assert clusters["top=front-left_bottom=back-right"] == 2
    assert clusters["top=center_bottom=center"] == 1


def test_analyze_mean_poses_present():
    eps = [
        _make_episode("s0", task_success=True,
                      initial_cube1_pose={"x": 0.0, "y": 0.0, "z": 0.8},
                      initial_top_bottom_distance_xy=0.15),
        _make_episode("f0", task_success=False,
                      initial_cube1_pose={"x": 0.1, "y": 0.1, "z": 0.8},
                      initial_top_bottom_distance_xy=0.20),
    ]
    result = analyze(eps)
    assert "success" in result["mean_poses"]
    assert "failure" in result["mean_poses"]
    assert result["mean_poses"]["success"]["initial_cube1_pose"]["x"] == 0.0
    assert result["mean_poses"]["failure"]["initial_cube1_pose"]["x"] == 0.1


def test_analyze_recommendation_field():
    eps = [_make_episode(f"e{i}", task_success=(i < 8)) for i in range(10)]
    result = analyze(eps)
    assert "recommendation" in result
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0


# ---- generate_report ----

def test_generate_report_writes_files():
    eps = [_make_episode(f"ep{i}", task_success=(i < 6)) for i in range(10)]
    analysis = analyze(eps)
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "report.json"
        md_path = Path(tmpdir) / "report.md"
        generate_report(analysis, str(json_path), str(md_path))
        assert json_path.exists()
        assert md_path.exists()
        # Verify JSON is valid and contains keys
        data = json.loads(json_path.read_text())
        assert data["total_episodes"] == 10
        # Verify markdown has expected sections
        md = md_path.read_text()
        assert "Stage 19-R1c" in md
        assert "Per-Goal Success" in md
        assert "Failure Reasons" in md
        assert "Distance XY Analysis" in md
        assert "Recommendation" in md


def test_generate_report_creates_parent_dir():
    eps = [_make_episode("ep0")]
    analysis = analyze(eps)
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "report.json"
        generate_report(analysis, str(nested), str(nested.with_suffix(".md")))
        assert nested.exists()
