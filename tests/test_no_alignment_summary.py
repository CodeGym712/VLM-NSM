"""Test no_alignment metrics semantics in compute_summary."""

from vlm_nsm.experiments.summarize_language_variation import compute_summary


def _make_result(
    alignment_mode="full",
    json_parse_success=True,
    schema_validity=True,
    alignment_success=True,
    alignment_applicable=True,
    repaired=False,
    planner_success=True,
    execution_success=True,
    goal_success=True,
):
    return {
        "model_name": "mock",
        "parser": "mock",
        "task": "hanoi",
        "n_blocks": 3,
        "difficulty": "hard",
        "category": "zh_standard",
        "prompt_variant": "full",
        "alignment_mode": alignment_mode,
        "alignment_applicable": alignment_applicable,
        "json_parse_success": json_parse_success,
        "schema_validity": schema_validity,
        "alignment_success": alignment_success,
        "repaired": repaired,
        "repair_types": [],
        "planner_success": planner_success,
        "execution_success": execution_success,
        "goal_success": goal_success,
        "error_type": None,
        "inference_time": 1.0,
        "input_tokens": 100,
        "output_tokens": 50,
    }


def test_no_alignment_alignment_rate_not_100():
    """no_alignment should not report 100% alignment success."""
    rows = [
        _make_result(alignment_mode="no_alignment", alignment_applicable=False,
                      alignment_success=None),
        _make_result(alignment_mode="no_alignment", alignment_applicable=False,
                      alignment_success=None),
    ]
    summary = compute_summary(rows)
    assert len(summary) == 1
    assert summary[0]["alignment_success_pct"] is None
    assert summary[0]["alignment_success_rate"] == "N/A"


def test_full_alignment_reports_rate():
    rows = [
        _make_result(alignment_mode="full"),
        _make_result(alignment_mode="full", alignment_success=True),
    ]
    summary = compute_summary(rows)
    assert len(summary) == 1
    assert summary[0]["alignment_success_pct"] is not None
    assert summary[0]["alignment_success_rate"] == "2/2"


def test_mixed_alignment_applicable():
    """Full alignment rows are counted, no_alignment rows are not."""
    rows = [
        _make_result(alignment_mode="full", alignment_success=True),
        _make_result(alignment_mode="no_alignment", alignment_applicable=False,
                      alignment_success=None),
    ]
    summary = compute_summary(rows)
    assert len(summary) == 2  # two different alignment_modes


def test_schema_validity_in_summary():
    rows = [
        _make_result(schema_validity=True),
        _make_result(schema_validity=False),
    ]
    summary = compute_summary(rows)
    assert len(summary) == 1
    assert summary[0]["schema_validity_rate"] == "1/2"
    assert summary[0]["schema_validity_pct"] == 50.0


def test_schema_validity_all_invalid():
    rows = [
        _make_result(json_parse_success=False, schema_validity=False),
        _make_result(json_parse_success=False, schema_validity=False),
    ]
    summary = compute_summary(rows)
    assert len(summary) == 1
    assert summary[0]["schema_validity_rate"] == "0/2"
    assert summary[0]["schema_validity_pct"] == 0.0
