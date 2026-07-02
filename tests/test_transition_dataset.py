"""Tests for transition JSONL validation and trajectory-safe splitting."""

import json

import pytest

from vlm_nsm.data.transition_dataset import (
    TransitionDatasetError,
    audit_records,
    deduplicate_records,
    group_split,
    load_jsonl,
    load_schema,
)


def record(transition_id: str, run_id: str, sequence_index: int = 0) -> dict:
    timestamp = "2026-07-02T01:02:03Z"
    predicate = {
        "name": "handempty",
        "args": [],
        "truth_value": True,
        "source": "oracle",
        "confidence": 1.0,
        "observed_at": timestamp,
        "freshness_seconds": 0.0,
        "evidence_id": f"{transition_id}-pre",
    }
    return {
        "schema_version": "embodied_transition/v1",
        "transition_id": transition_id,
        "run_id": run_id,
        "sequence_index": sequence_index,
        "timestamp": timestamp,
        "task_id": "stack-2",
        "scene_id": f"scene-{run_id}",
        "seed": sequence_index,
        "pre_predicates": [predicate],
        "post_predicates": [predicate | {"evidence_id": f"{transition_id}-post"}],
        "grounded_option": {"name": "pick", "arguments": ["cube1"]},
        "expected_effect": {"add": [], "delete": []},
        "observed_effect": {"add": [], "delete": []},
        "observer": {"name": "mujoco_oracle", "version": "v1"},
        "executor": {"name": "atomic_pid", "version": "executor_v1"},
        "continuous_context": {},
        "controller_diagnostics": {},
        "success": True,
        "raw_failure_type": None,
        "attribution": "SUCCESS",
        "attribution_confidence": 1.0,
        "evidence_ids": [f"{transition_id}-pre", f"{transition_id}-post"],
        "metadata": {},
    }


def test_load_jsonl_reports_malformed_line_number(tmp_path):
    path = tmp_path / "transitions.jsonl"
    path.write_text(json.dumps(record("t1", "r1")) + "\n{broken\n", encoding="utf-8")

    with pytest.raises(TransitionDatasetError, match="line 2"):
        load_jsonl(path)


def test_audit_reports_schema_path_and_duplicate_id():
    invalid = record("bad", "r2")
    del invalid["executor"]["version"]
    records = [record("t1", "r1"), record("t1", "r1", 1), invalid]

    report = audit_records(records, load_schema())

    assert report.record_count == 3
    assert report.invalid_count == 1
    assert report.duplicate_transition_ids == ("t1",)
    assert report.ok is False
    assert any(issue.line_number == 3 and "executor.version" in issue.path for issue in report.issues)


def test_deduplicate_records_preserves_first_occurrence():
    first = record("t1", "r1", 0)
    duplicate = record("t1", "r1", 1)
    unique = record("t2", "r1", 2)

    result = deduplicate_records([first, duplicate, unique])

    assert result == [first, unique]


def test_group_split_is_deterministic_and_never_leaks_runs():
    records = [
        record(f"t-{run}-{step}", f"run-{run}", step)
        for run in range(6)
        for step in range(2)
    ]

    first = group_split(records, train_ratio=0.5, validation_ratio=0.25, seed=11)
    second = group_split(records, train_ratio=0.5, validation_ratio=0.25, seed=11)

    assert first == second
    assert sum(len(items) for items in first.values()) == len(records)
    run_sets = [{item["run_id"] for item in first[name]} for name in ("train", "validation", "test")]
    assert run_sets[0].isdisjoint(run_sets[1])
    assert run_sets[0].isdisjoint(run_sets[2])
    assert run_sets[1].isdisjoint(run_sets[2])
    assert all(run_sets)


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio"),
    [(0.0, 0.2), (0.8, 0.3), (1.0, 0.0)],
)
def test_group_split_rejects_ratios_without_three_partitions(train_ratio, validation_ratio):
    with pytest.raises(ValueError, match="ratios"):
        group_split([record("t1", "r1")], train_ratio, validation_ratio)
