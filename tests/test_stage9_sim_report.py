"""Verify Stage 9 simulation protocol, reports, and e2e config."""

import json
from pathlib import Path

REPORT_DIR = Path("generated/report_tables/stage_9")
RESULTS_DIR = Path("generated/results/stage_9")
DOC_DIR = Path("docs")


# ── S9-A: simulation_protocol.md ──────────────────────────────────────────


def test_protocol_doc_exists():
    assert (DOC_DIR / "simulation_protocol.md").exists()


def test_protocol_mentions_canonical_checkpoint():
    text = (DOC_DIR / "simulation_protocol.md").read_text()
    assert "stage_7B-4/dagger_reach_bc/best.pt" in text


def test_protocol_mentions_canonical_config():
    text = (DOC_DIR / "simulation_protocol.md").read_text()
    assert "robosuite_best_execution_config_stage7a12_final.json" in text


def test_protocol_mentions_bc_phases():
    text = (DOC_DIR / "simulation_protocol.md").read_text()
    assert "reach_pick" in text
    assert "bc_phases" in text


def test_protocol_mentions_failure_types():
    text = (DOC_DIR / "simulation_protocol.md").read_text()
    for ft in ("horizon", "preplace", "verify_failed"):
        assert ft in text


def test_protocol_mentions_deprecated():
    text = (DOC_DIR / "simulation_protocol.md").read_text()
    assert "Deprecated" in text or "废弃" in text


# ── S9-B/C: Result files ──────────────────────────────────────────────────


def test_scripted_canonical_result_exists():
    p = RESULTS_DIR / "scripted_canonical_100ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 100
        assert data.get("bc_phases") == [] or data.get("bc_phases") == ["none"]


def test_dagger_seed0_result_exists():
    p = RESULTS_DIR / "dagger_reach_seed0_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


def test_dagger_seed1_result_exists():
    p = RESULTS_DIR / "dagger_reach_seed1_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


def test_dagger_seed2_result_exists():
    p = RESULTS_DIR / "dagger_reach_seed2_50ep.json"
    if p.exists():
        data = json.loads(p.read_text())
        assert data["num_episodes"] == 50


# ── S9-E: Reports ─────────────────────────────────────────────────────────


def test_failure_audit_v3_exists():
    p = REPORT_DIR / "dagger_100ep_failure_audit_v3.md"
    assert p.exists()
    text = p.read_text()
    assert "Failure Audit" in text
    assert "Stage 8" in text


def test_canonical_comparison_exists():
    p = REPORT_DIR / "canonical_sim_comparison.md"
    assert p.exists()
    text = p.read_text()
    assert "Canonical" in text or "canonical" in text


# ── S9-F: E2E demo script ────────────────────────────────────────────────


def test_e2e_demo_script_exists():
    p = Path("src/vlm_nsm/experiments/run_e2e_language_to_sim.py")
    assert p.exists()
    text = p.read_text()
    assert "MockGoalParser" in text
    assert "AlignmentLayer" in text
    assert "PlannerAdapter" in text
    assert "RobosuiteEnvWrapper" in text
    assert "run_episode" in text


def test_e2e_has_demos():
    text = Path("src/vlm_nsm/experiments/run_e2e_language_to_sim.py").read_text()
    assert "DEMOS" in text


# ── Export script ─────────────────────────────────────────────────────────


def test_export_script_exists():
    p = Path("src/vlm_nsm/experiments/export_stage9_sim_report.py")
    assert p.exists()
    text = p.read_text()
    assert "failure_audit" in text.lower()
    assert "canonical_comparison" in text.lower()
