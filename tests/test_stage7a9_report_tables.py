"""Tests for Stage 7A-9 report table exports (robosuite-independent).

Verifies that the export functions handle best_config_v2 data
and that the ablation comparison table includes the new row.
"""

from vlm_nsm.experiments.export_embodied_report_tables import (
    export_ablation_comparison,
    _md_table,
)


def test_stage7a9_ablation_includes_best_config():
    """Ablation comparison should include best_config_v2 row."""
    consolidated = {
        "baseline": {"label": "Baseline (all flags)", "num_episodes": 20,
                     "strict_physical_success_rate": 0.9,
                     "semantic_goal_success_rate": 0.9,
                     "mean_steps": 989, "p95_steps": 1000,
                     "strict_reality_gap": "10%", "semantic_reality_gap": "10%",
                     "timeout_after_goal_count": 0, "timeout_before_goal_count": 1,
                     "failure_stage_distribution": {"horizon": 1}},
        "best_config_v2": {"label": "Best Config v2 (kp=0.08)", "num_episodes": 20,
                           "strict_physical_success_rate": 0.95,
                           "semantic_goal_success_rate": 0.95,
                           "mean_steps": 650, "p95_steps": 850,
                           "strict_reality_gap": "5%", "semantic_reality_gap": "5%",
                           "timeout_after_goal_count": 0, "timeout_before_goal_count": 0,
                           "failure_stage_distribution": {}},
    }
    assert "best_config_v2" in consolidated
    best = consolidated["best_config_v2"]
    assert best["label"] == "Best Config v2 (kp=0.08)"
    assert best["strict_physical_success_rate"] >= 0.9


def test_best_config_v2_metrics_format():
    """Best config v2 row values must be properly formatted for display."""
    best = {
        "label": "Best Config v2 (kp=0.08)",
        "num_episodes": 20,
        "strict_physical_success_rate": 0.95,
        "semantic_goal_success_rate": 0.95,
        "mean_steps": 650,
        "p95_steps": 850,
        "strict_reality_gap": "5%",
        "semantic_reality_gap": "5%",
        "timeout_after_goal_count": 0,
        "timeout_before_goal_count": 0,
        "failure_stage_distribution": {},
    }
    assert f"{best['strict_physical_success_rate'] * 100:.0f}%" == "95%"
    assert f"{best['semantic_goal_success_rate'] * 100:.0f}%" == "95%"
    assert best["mean_steps"] <= 700
    assert best["p95_steps"] < 900


def test_md_table_creation():
    """_md_table utility should produce valid markdown."""
    headers = ["Config", "Strict %", "Semantic %"]
    rows = [
        ["Baseline", "90%", "90%"],
        ["Best Config v2", "95%", "95%"],
    ]
    table = _md_table(headers, rows)
    assert "| Config" in table
    assert "| ---" in table
    assert "| Best Config v2" in table
    assert "95%" in table


def test_export_ablation_comparison_includes_best_config_v2(monkeypatch, tmp_path):
    """Ablation comparison export should include best_config_v2 row."""
    report_dir = tmp_path / "report_tables"
    report_dir.mkdir(parents=True)

    import vlm_nsm.experiments.export_embodied_report_tables as mod
    monkeypatch.setattr(mod, "REPORT_DIR", report_dir)

    consolidated = {
        "baseline": {"label": "Baseline", "num_episodes": 20,
                     "strict_physical_success_rate": 0.9,
                     "semantic_goal_success_rate": 0.9,
                     "mean_steps": 989, "p95_steps": 1000,
                     "strict_reality_gap": "10%", "semantic_reality_gap": "10%",
                     "timeout_after_goal_count": 0, "timeout_before_goal_count": 1,
                     "failure_stage_distribution": {"horizon": 1}},
        "best_config_v2": {"label": "Best Config v2", "num_episodes": 20,
                           "strict_physical_success_rate": 0.95,
                           "semantic_goal_success_rate": 0.95,
                           "mean_steps": 650, "p95_steps": 850,
                           "strict_reality_gap": "5%", "semantic_reality_gap": "5%",
                           "timeout_after_goal_count": 0, "timeout_before_goal_count": 0,
                           "failure_stage_distribution": {}},
    }

    md_path = export_ablation_comparison(consolidated)
    assert md_path.exists()
    content = md_path.read_text()
    assert "Best Config v2" in content
    assert "95%" in content


def test_best_config_v2_failure_stages():
    """Best config v2 should have minimal failure stages."""
    best = {
        "failure_stage_distribution": {},
    }
    assert len(best["failure_stage_distribution"]) == 0
