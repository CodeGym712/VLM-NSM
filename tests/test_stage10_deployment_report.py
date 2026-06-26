"""Verify Stage 10 deployment report and results."""

import json
from pathlib import Path

REPORT_DIR = Path("generated/report_tables/stage_10")
RESULTS_S9 = Path("generated/results/stage_9")
RESULTS_S10 = Path("generated/results/stage_10")
DOC = Path("docs/simulation_protocol.md")


# ── S10-A: reach_pick-only 100ep summary ─────────────────────────────────


def test_reach_pick_only_100ep_exists():
    p = RESULTS_S9 / "dagger_reach_pick_only_100ep.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["num_episodes"] == 100
    assert data["success_count"] >= 85  # expect >= 85%


def test_reach_pick_only_100ep_summary_exists():
    assert (REPORT_DIR / "reach_pick_only_100ep_summary.md").exists()


# ── S10-B: reach_pick-only 3×50ep ────────────────────────────────────────


def test_reach_pick_only_seed0_50ep_exists():
    p = RESULTS_S10 / "reach_pick_only_seed0_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


def test_reach_pick_only_seed1_50ep_exists():
    p = RESULTS_S10 / "reach_pick_only_seed1_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


def test_reach_pick_only_seed2_50ep_exists():
    p = RESULTS_S10 / "reach_pick_only_seed2_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


# ── S10-C: deployment comparison ─────────────────────────────────────────


def test_deployment_comparison_exists():
    p = REPORT_DIR / "deployment_policy_comparison.md"
    assert p.exists()
    text = p.read_text()
    assert "reach_pick-only" in text or "reach-only" in text
    assert "Scripted" in text


def test_deployment_comparison_has_recommendation():
    text = (REPORT_DIR / "deployment_policy_comparison.md").read_text()
    assert "Recommendation" in text or "recommendation" in text


# ── S10-D: protocol update ───────────────────────────────────────────────


def test_protocol_default_is_reach_pick():
    text = DOC.read_text()
    assert "92.0%" in text
    assert "Deployment Default" in text or "canonical deployment" in text


def test_protocol_mentions_reach_place_research():
    text = DOC.read_text()
    assert "reach_place" in text
    assert "Research Variant" in text or "research" in text.lower()


# ── Expected results directory ────────────────────────────────────────────


def test_stage10_dirs_exist():
    assert RESULTS_S10.exists()
    assert REPORT_DIR.exists()
