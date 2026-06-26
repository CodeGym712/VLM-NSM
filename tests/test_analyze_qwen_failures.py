"""Test analyze_qwen_failures with mock result data (no real Qwen)."""

import json
from pathlib import Path

from vlm_nsm.experiments.analyze_qwen_failures import (
    classify_failure,
    collect_results,
    extract_failure_cases,
    extract_repair_stats,
    short_reason,
)


def _make_result(
    sample_id="test_001",
    model_name="mock",
    task="hanoi",
    category="zh_standard",
    json_parse_success=True,
    schema_validity=None,
    alignment_success=True,
    planner_success=True,
    execution_success=True,
    goal_success=True,
    error_type=None,
    repaired=False,
    repair_types=None,
    instruction="test instruction",
    raw_output_path=None,
    inference_time=1.0,
    input_tokens=100,
    output_tokens=50,
):
    return {
        "sample_id": sample_id,
        "model_name": model_name,
        "task": task,
        "category": category,
        "instruction": instruction,
        "json_parse_success": json_parse_success,
        "schema_validity": schema_validity if schema_validity is not None else json_parse_success,
        "alignment_success": alignment_success,
        "planner_success": planner_success,
        "execution_success": execution_success,
        "goal_success": goal_success,
        "error_type": error_type,
        "repaired": repaired,
        "repair_types": repair_types or [],
        "raw_output_path": raw_output_path,
        "inference_time": inference_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def test_classify_schema_invalid():
    r = _make_result(json_parse_success=True, schema_validity=False,
                      error_type="schema_invalid")
    assert classify_failure(r) == "schema_invalid"


def test_short_reason_schema_invalid():
    r = _make_result(json_parse_success=True, schema_validity=False,
                      error_type="schema_invalid")
    assert "Schema invalid" in short_reason(r)


def test_classify_json_parse_error():
    r = _make_result(json_parse_success=False, error_type="json_parse_error")
    assert classify_failure(r) == "json_parse_error"


def test_classify_invalid_object():
    r = _make_result(alignment_success=False, error_type="invalid_object")
    assert classify_failure(r) == "invalid_object"


def test_classify_invalid_predicate():
    r = _make_result(alignment_success=False, error_type="invalid_predicate")
    assert classify_failure(r) == "invalid_predicate"


def test_classify_type_error():
    r = _make_result(alignment_success=False, error_type="type_error")
    assert classify_failure(r) == "type_error"


def test_classify_incomplete_goal():
    r = _make_result(alignment_success=False, error_type="incomplete_goal")
    assert classify_failure(r) == "incomplete_goal"


def test_classify_planner_fail():
    r = _make_result(planner_success=False, error_type="planner_error")
    assert classify_failure(r) == "planner_fail"


def test_classify_execution_fail():
    r = _make_result(
        planner_success=True, execution_success=False, goal_success=False,
        error_type=None,
    )
    assert classify_failure(r) == "execution_fail"


def test_classify_goal_not_satisfied():
    r = _make_result(
        planner_success=True, execution_success=True, goal_success=False,
        error_type=None,
    )
    assert classify_failure(r) == "goal_not_satisfied"


def test_short_reason_json_parse():
    r = _make_result(json_parse_success=False)
    assert "JSON" in short_reason(r)


def test_short_reason_invalid_object():
    r = _make_result(alignment_success=False, error_type="invalid_object")
    assert "Invalid object" in short_reason(r)


def test_short_reason_incomplete_goal():
    r = _make_result(alignment_success=False, error_type="incomplete_goal")
    assert "Incomplete goal" in short_reason(r)


def test_extract_failure_cases():
    rows = [
        _make_result(sample_id="ok", goal_success=True),
        _make_result(sample_id="fail1", goal_success=False, error_type="incomplete_goal"),
        _make_result(sample_id="fail2", goal_success=False, error_type="planner_error",
                      planner_success=False),
    ]
    cases = extract_failure_cases(rows)
    assert len(cases) == 2
    assert cases[0]["sample_id"] == "fail1"
    assert cases[1]["sample_id"] == "fail2"
    assert "failure_type" in cases[0]
    assert "short_reason" in cases[0]


def test_extract_failure_cases_all_success():
    rows = [_make_result(sample_id="ok", goal_success=True) for _ in range(5)]
    cases = extract_failure_cases(rows)
    assert len(cases) == 0


def test_extract_repair_stats():
    rows = [
        _make_result(sample_id="r1", repaired=True, repair_types=["predicate_alias"]),
        _make_result(sample_id="r2", repaired=True, repair_types=["predicate_alias"]),
        _make_result(sample_id="r3", repaired=True, repair_types=["goal_completion"]),
        _make_result(sample_id="r4", repaired=False, repair_types=[]),
    ]
    stats = extract_repair_stats(rows)
    assert len(stats) == 2  # two distinct repair_types
    for s in stats:
        if s["repair_type"] == "predicate_alias":
            assert s["count"] == 2
        elif s["repair_type"] == "goal_completion":
            assert s["count"] == 1


def test_collect_results(tmp_path):
    """Test collect_results with a mock JSON file."""
    sample_data = [
        _make_result(sample_id="test1", model_name="mock_model"),
        _make_result(sample_id="test2", model_name="mock_model"),
    ]
    fpath = tmp_path / "language_variation_hanoi_mock_model.json"
    fpath.write_text(json.dumps(sample_data), encoding="utf-8")

    rows = collect_results(tmp_path)
    assert len(rows) == 2


def test_collect_results_skips_summary(tmp_path):
    """Verify files ending with _summary.json are skipped."""
    sample_data = [_make_result()]
    fpath = tmp_path / "language_variation_hanoi_mock_model.json"
    fpath.write_text(json.dumps(sample_data), encoding="utf-8")
    summary_path = tmp_path / "language_variation_summary.json"
    summary_path.write_text(json.dumps(sample_data), encoding="utf-8")

    rows = collect_results(tmp_path)
    assert len(rows) == 1  # summary file should be skipped
