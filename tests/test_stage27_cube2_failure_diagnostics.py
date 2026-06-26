"""Test Stage 27 cube2 failure diagnostics with minimal mock data."""

import json
from pathlib import Path

from vlm_nsm.experiments.stage27_cube2_failure_diagnostics import (
    _classify_failure_phase,
    _is_failure,
    _table_column_for_reason,
    _task_direction,
    diagnose,
)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_task_direction_green_on_red():
    assert _task_direction({"task_direction": "green_on_red"}) == "green_on_red"


def test_task_direction_red_on_green():
    assert _task_direction({"task_direction": "red_on_green"}) == "red_on_green"


def test_task_direction_from_goal_id():
    assert (
        _task_direction({"goal_id": "s27_green_on_red_s00"}) == "green_on_red"
    )
    assert (
        _task_direction({"episode_id": "s27_red_on_green_s05"}) == "red_on_green"
    )


def test_task_direction_unknown():
    assert _task_direction({"goal_id": "unrelated"}) == "unknown"


def test_is_failure_true():
    assert _is_failure({"task_success": False}) is True


def test_is_failure_not_false():
    assert _is_failure({"task_success": True}) is False
    assert _is_failure({}) is False


def test_classify_failure_phase_grasp():
    assert _classify_failure_phase({"failure_reason": "nocontact"}) == "grasp_failure"
    assert _classify_failure_phase({"failure_reason": "grasp"}) == "grasp_failure"
    assert _classify_failure_phase({"failure_reason": "slip"}) == "grasp_failure"


def test_classify_failure_phase_timeout():
    assert _classify_failure_phase({"failure_reason": "timeout"}) == "timeout"


def test_classify_failure_phase_place():
    assert _classify_failure_phase({"failure_reason": "place"}) == "place_failure"


def test_classify_failure_phase_none():
    assert _classify_failure_phase({"failure_reason": "none"}) == "success"


def test_classify_failure_phase_unknown():
    assert _classify_failure_phase({"failure_reason": "something_else"}) == "unknown"
    assert _classify_failure_phase({}) == "unknown"


def test_table_column_for_reason():
    assert _table_column_for_reason("nocontact") == "nocontact"
    assert _table_column_for_reason("grasp") == "slip"
    assert _table_column_for_reason("slip") == "slip"
    assert _table_column_for_reason("timeout") == "timeout"
    assert _table_column_for_reason("place") == "place"
    assert _table_column_for_reason("") == "unknown"
    assert _table_column_for_reason("weird") == "unknown"


# ---------------------------------------------------------------------------
# Integration test: empty failure list handled gracefully
# ---------------------------------------------------------------------------


def test_diagnostics_handles_empty_failure_list_gracefully(tmp_path: Path):
    """When no green-on-red failures exist, diagnostics should report 0
    failures and produce valid output without crashing."""

    # Build oracle JSONL: only red_on_green successes and one green success.
    oracle_rows = [
        {
            "episode_id": "s27_red_on_green_s00",
            "seed": 0,
            "task_direction": "red_on_green",
            "task_success": True,
        },
        {
            "episode_id": "s27_red_on_green_s01",
            "seed": 1,
            "task_direction": "red_on_green",
            "task_success": True,
        },
        {
            "episode_id": "s27_green_on_red_s00",
            "seed": 0,
            "task_direction": "green_on_red",
            "task_success": True,  # green success, not a failure
        },
    ]

    # Build live JSONL: one red failure, one green success — no green failures.
    live_rows = [
        {
            "episode_id": "s27_red_on_green_s02",
            "seed": 2,
            "task_direction": "red_on_green",
            "task_success": False,
            "failure_reason": "timeout",
        },
        {
            "episode_id": "s27_green_on_red_s01",
            "seed": 1,
            "task_direction": "green_on_red",
            "task_success": True,
        },
    ]

    oracle_path = tmp_path / "oracle_episodes.jsonl"
    live_path = tmp_path / "live_episodes.jsonl"

    oracle_path.write_text(
        "\n".join(json.dumps(r) for r in oracle_rows) + "\n",
        encoding="utf-8",
    )
    live_path.write_text(
        "\n".join(json.dumps(r) for r in live_rows) + "\n",
        encoding="utf-8",
    )

    result = diagnose(
        oracle_jsonl=str(oracle_path),
        live_jsonl=str(live_path),
    )

    # Core invariants when there are zero green failures.
    assert result["n_green_failures"] == 0
    assert result["n_green_success"] == 2  # one oracle, one live
    assert len(result["failure_details"]) == 0
    assert result["green_failure_phase_distribution"] == {}
    assert result["green_failure_reason_distribution"] == {}
    assert result["n_green_total"] == 2
    assert result["n_red_total"] == 3

    # Summary table should still exist for both directions.
    assert isinstance(result["summary_table"], list)

    # green_on_red / oracle: 1 total, 0 failures.
    gor_oracle = [
        s
        for s in result["summary_table"]
        if s["task_direction"] == "green_on_red" and s["condition"] == "oracle"
    ][0]
    assert gor_oracle["n_total"] == 1
    assert gor_oracle["n_failures"] == 0

    # green_on_red / live: 1 total, 0 failures.
    gor_live = [
        s
        for s in result["summary_table"]
        if s["task_direction"] == "green_on_red" and s["condition"] == "live"
    ][0]
    assert gor_live["n_total"] == 1
    assert gor_live["n_failures"] == 0
