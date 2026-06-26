"""Tests for Stage 7A-11 report table exports (robosuite-independent)."""

from vlm_nsm.experiments.export_embodied_report_tables import (
    export_termination_stats_table,
    export_phase_budget_table,
    REPORT_DIR,
)


def test_export_termination_stats_table_exists():
    """export_termination_stats_table should be callable."""
    assert callable(export_termination_stats_table)


def test_export_termination_stats_with_none():
    """Should handle None input gracefully."""
    path = export_termination_stats_table(None)
    assert path.exists()
    content = path.read_text()
    assert "No data" in content


def test_export_termination_stats_with_data():
    """Should render table with termination stats."""
    mock = {
        "num_episodes": 20,
        "termination_stats": {
            "terminate_on_goal_count": 15,
            "terminate_after_safe_release_count": 2,
            "skip_retreat_if_goal_satisfied_count": 3,
            "normal_completion_count": 0,
            "horizon_termination_count": 0,
        },
    }
    path = export_termination_stats_table(mock)
    assert path.exists()
    content = path.read_text()
    assert "Terminate on goal" in content
    assert "15" in content


def test_export_phase_budget_table_exists():
    assert callable(export_phase_budget_table)


def test_export_phase_budget_with_none():
    path = export_phase_budget_table(None)
    assert path.exists()
    assert "No data" in path.read_text()


def test_export_phase_budget_with_phase_trace():
    """Should render phase budget from per-episode phase_trace data."""
    mock = {
        "mean_steps": 900,
        "per_episode": [
            {"phase_trace": {"reach_pick": 150, "grasp": 300, "lift": 180,
                              "reach_place": 160, "release": 50, "retreat": 60}},
            {"phase_trace": {"reach_pick": 140, "grasp": 280, "lift": 170,
                              "reach_place": 150, "release": 45, "retreat": 55}},
        ],
    }
    path = export_phase_budget_table(mock, label="test")
    assert path.exists()
    content = path.read_text()
    assert "Reach Pick" in content or "Reach" in content
    assert "Grasp" in content


def test_export_phase_budget_file_label():
    """Should create file with correct label."""
    mock = {"mean_steps": 500, "per_episode": []}
    path = export_phase_budget_table(mock, label="stage7a11_20ep")
    assert "stage7a11_20ep" in str(path)
