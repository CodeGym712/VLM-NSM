"""Tests for export_embodied_report_tables.py."""

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.export_embodied_report_tables import (
    export_embodied_mock_summary,
    export_demo_dataset_summary,
    export_replay_sanity_summary,
)


def _make_mock_json(path: Path, data: dict, name: str):
    p = path / name
    p.write_text(json.dumps(data))
    return p


def test_export_embodied_mock_summary_creates_files(monkeypatch, tmp_path):
    """Report tables should be generated from mock data."""
    with tempfile.TemporaryDirectory() as td:
        report_dir = Path(td) / "report_tables"
        report_dir.mkdir(parents=True)

        # Mock REPORT_DIR
        import vlm_nsm.experiments.export_embodied_report_tables as mod
        monkeypatch.setattr(mod, "REPORT_DIR", report_dir)

        clean_data = {
            "step_success_rate": 1.0,
            "skill_success_rate": 1.0,
            "episode_success_rate": 1.0,
            "goal_success_rate": 1.0,
            "predicate_match_rate": 1.0,
            "reality_gap": 0.0,
            "failure_stage_distribution": {},
        }

        md_path, csv_path = export_embodied_mock_summary(clean_data, None)
        assert md_path.exists()
        assert csv_path.exists()
        content = md_path.read_text()
        assert "100.0%" in content
        assert "Clean" in content


def test_export_demo_dataset_summary_creates_file(tmp_path):
    """Demo dataset summary should be generated."""
    report_dir = tmp_path / "report_tables"
    report_dir.mkdir(parents=True)

    import vlm_nsm.experiments.export_embodied_report_tables as mod
    import vlm_nsm.core.utils as utils
    original = utils.PROJECT_ROOT
    utils.PROJECT_ROOT = tmp_path

    try:
        # Direct call - doesn't use PROJECT_ROOT for writing
        demos = {"num_episodes": 20, "num_success": 20, "episode_success_rate": 1.0,
                 "failure_stage_distribution": {}, "full_obs_saved": True,
                 "full_obs_dim": 36, "filtered_obs_dim": 16, "reduction_ratio": 0.555}
        filter_stats = {"full_obs_available": True, "full_obs_dim": 36, "filtered_obs_dim": 16,
                        "reduction_ratio": 0.555, "num_transitions": 240,
                        "obs_mean": 0.1, "obs_std": 0.3, "step_distribution": {"step_0": 40}}
        replay = {"total_steps_in_data": 240, "total_episodes_in_data": 20,
                  "data_success_rate": 1.0, "replay_success_rate": 1.0,
                  "episode_count_match": True, "obs_dim_match": True,
                  "action_dim_match": True, "step_sequence_valid": True}

        from vlm_nsm.experiments.export_embodied_report_tables import REPORT_DIR as rd
        # Override with our tmp
        rd_actual = report_dir

        md_path = export_demo_dataset_summary(demos, replay, filter_stats)
        assert md_path is not None
        assert "240" in md_path.read_text()
    finally:
        utils.PROJECT_ROOT = original


def test_export_replay_sanity_summary_creates_file(tmp_path):
    """Replay sanity summary should be generated."""
    report_dir = tmp_path / "report_tables"
    report_dir.mkdir(parents=True)

    import vlm_nsm.experiments.export_embodied_report_tables as mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "REPORT_DIR", report_dir)

    try:
        replay = {"total_steps_in_data": 240, "total_episodes_in_data": 20,
                  "data_success_rate": 1.0, "replay_success_rate": 1.0,
                  "episode_count_match": True}

        md_path = export_replay_sanity_summary(replay)
        assert md_path.exists()
        content = md_path.read_text()
        assert "240" in content
        assert "1.0" in content
    finally:
        monkeypatch.undo()
