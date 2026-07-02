"""CLI tests for embodied transition dataset auditing."""

import json

from vlm_nsm.experiments.audit_embodied_transitions import main


def record(transition_id: str, run_id: str) -> dict:
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
        "sequence_index": 0,
        "timestamp": timestamp,
        "task_id": "stack-2",
        "scene_id": f"scene-{run_id}",
        "seed": 7,
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


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def test_cli_writes_report_and_trajectory_safe_splits(tmp_path):
    source = tmp_path / "transitions.jsonl"
    report_path = tmp_path / "audit.json"
    split_dir = tmp_path / "splits"
    write_jsonl(source, [record(f"t{index}", f"run{index}") for index in range(6)])

    result = main([
        "--input", str(source),
        "--report", str(report_path),
        "--split-dir", str(split_dir),
        "--seed", "19",
    ])

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["record_count"] == 6
    assert {path.name for path in split_dir.glob("*.jsonl")} == {
        "train.jsonl", "validation.jsonl", "test.jsonl"
    }
    split_runs = []
    for name in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (split_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
        split_runs.append({row["run_id"] for row in rows})
    assert split_runs[0].isdisjoint(split_runs[1] | split_runs[2])
    assert split_runs[1].isdisjoint(split_runs[2])


def test_cli_returns_nonzero_for_duplicate_transition_ids(tmp_path):
    source = tmp_path / "duplicates.jsonl"
    report_path = tmp_path / "audit.json"
    write_jsonl(source, [record("same-id", "run1"), record("same-id", "run2")])

    result = main(["--input", str(source), "--report", str(report_path)])

    assert result == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["duplicate_transition_ids"] == ["same-id"]
