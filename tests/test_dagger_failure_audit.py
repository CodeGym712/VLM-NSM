"""Verify DAgger failure audit data."""

import json
from pathlib import Path


RESULTS_DIR = Path("generated/results/stage_7B-5")


def test_dagger_failures_collected():
    """All DAgger seed results should have failure_type on failed episodes."""
    for seed in range(3):
        data = json.loads((RESULTS_DIR / f"dagger_reach_seed{seed}_20ep.json").read_text())
        for ep in data["episode_results"]:
            if not ep["success"]:
                assert ep.get("failure_type") is not None, \
                    f"seed={seed} failed episode has no failure_type"


def test_dagger_failure_rate_under_20_percent():
    for seed in range(3):
        data = json.loads((RESULTS_DIR / f"dagger_reach_seed{seed}_20ep.json").read_text())
        fails = sum(1 for ep in data["episode_results"] if not ep["success"])
        assert fails <= 4, f"seed={seed} has {fails} failures (>4)"


def test_bc_main_failure_types_all_horizon():
    """bc_main failures should all be horizon (not preplace)."""
    for seed in range(3):
        data = json.loads((RESULTS_DIR / f"bc_main_seed{seed}_20ep.json").read_text())
        for ep in data["episode_results"]:
            if not ep["success"]:
                assert ep["failure_type"] == "horizon", \
                    f"bc_main seed={seed} has non-horizon failure: {ep['failure_type']}"


def test_failure_audit_table_exists():
    p = Path("generated/report_tables/stage_7B-5/dagger_failure_audit.md")
    assert p.exists()
    text = p.read_text()
    assert "horizon" in text or "Horizon" in text
