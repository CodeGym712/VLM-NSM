"""Verify Stage 7B-5 report tables and JSON results exist."""

from pathlib import Path


RESULTS_DIR = Path("generated/results/stage_7B-5")
TABLES_DIR = Path("generated/report_tables/stage_7B-5")


def test_multiseed_json_exists():
    for seed in range(3):
        for prefix in ["scripted", "bc_main", "dagger_reach"]:
            p = RESULTS_DIR / f"{prefix}_seed{seed}_20ep.json"
            assert p.exists(), f"Missing {p}"


def test_dagger_50ep_json_exists():
    p = RESULTS_DIR / "dagger_reach_50ep.json"
    assert p.exists()


def test_dagger_only_bc_pt_exists():
    p = RESULTS_DIR / "dagger_only_bc" / "best.pt"
    assert p.exists()


def test_dagger_only_bc_eval_json_exists():
    p = RESULTS_DIR / "dagger_only_bc_eval_20ep.json"
    assert p.exists()
    import json
    data = json.loads(p.read_text())
    assert data["success_rate"] == 0.0
    assert data["num_episodes"] == 20


def test_report_table_multiseed_exists():
    p = TABLES_DIR / "dagger_multiseed_comparison.md"
    assert p.exists()
    text = p.read_text()
    assert "DAgger" in text


def test_report_table_data_ablation_exists():
    p = TABLES_DIR / "dagger_data_ablation.md"
    assert p.exists()
    text = p.read_text()
    assert "DAgger-only" in text or "dagger_only" in text or "0%" in text


def test_report_table_phase_ablation_exists():
    p = TABLES_DIR / "phase_ablation.md"
    assert p.exists()
    text = p.read_text()
    assert "reach_pick" in text


def test_report_table_failure_audit_exists():
    p = TABLES_DIR / "dagger_failure_audit.md"
    assert p.exists()
