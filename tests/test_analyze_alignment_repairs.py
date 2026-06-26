"""Test analyze_alignment_repairs with mock data."""

import json
from pathlib import Path

from vlm_nsm.experiments.analyze_alignment_repairs import (
    collect_results,
    extract_repair_stats,
)


def _make_result(
    sample_id="test_001",
    model_name="mock",
    task="hanoi",
    difficulty="easy",
    prompt_variant="full",
    alignment_mode="full",
    repaired=False,
    repair_types=None,
    goal_success=True,
):
    return {
        "sample_id": sample_id,
        "model_name": model_name,
        "task": task,
        "difficulty": difficulty,
        "prompt_variant": prompt_variant,
        "alignment_mode": alignment_mode,
        "repaired": repaired,
        "repair_types": repair_types or [],
        "goal_success": goal_success,
        "instruction": "test instruction",
        "raw_output_path": "",
    }


def test_extract_repair_stats_on_empty(tmp_path):
    """No repairs should produce empty stats list."""
    data = [
        _make_result(sample_id="s1", repaired=False),
        _make_result(sample_id="s2", repaired=False),
    ]
    fpath = tmp_path / "language_variation_hanoi_easy_mock_full_full.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")

    stats = extract_repair_stats(collect_results(tmp_path))
    assert len(stats) == 0


def test_extract_repair_stats_with_repairs(tmp_path):
    data = [
        _make_result(sample_id="s1", repaired=True,
                      repair_types=["predicate_alias"]),
        _make_result(sample_id="s2", repaired=True,
                      repair_types=["predicate_alias"]),
        _make_result(sample_id="s3", repaired=True,
                      repair_types=["goal_completion"]),
        _make_result(sample_id="s4", repaired=False),
    ]
    fpath = tmp_path / "language_variation_hanoi_hard_mock_full_full.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")

    stats = extract_repair_stats(collect_results(tmp_path))
    assert len(stats) == 2
    for s in stats:
        if s["repair_type"] == "predicate_alias":
            assert s["count"] == 2
            assert s["task"] == "hanoi"
            assert "difficulty" in s
            assert "prompt_variant" in s
            assert "alignment_mode" in s
        elif s["repair_type"] == "goal_completion":
            assert s["count"] == 1


def test_collect_results_skips_summary(tmp_path):
    data = [_make_result()]
    fpath = tmp_path / "language_variation_hanoi_easy_mock_full_full.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")
    summary_path = tmp_path / "language_variation_summary.json"
    summary_path.write_text(json.dumps(data), encoding="utf-8")

    rows = collect_results(tmp_path)
    assert len(rows) == 1


def test_repair_stats_with_difficulty(tmp_path):
    data = [
        _make_result(sample_id="s1", difficulty="hard",
                      repaired=True, repair_types=["predicate_alias"]),
    ]
    fpath = tmp_path / "language_variation_hanoi_hard_mock_full_full.json"
    fpath.write_text(json.dumps(data), encoding="utf-8")

    stats = extract_repair_stats(collect_results(tmp_path))
    assert len(stats) == 1
    assert stats[0]["difficulty"] == "hard"
    assert stats[0]["prompt_variant"] == "full"
