from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_server_r2_r3_handoff import audit_r2_package, audit_r3_package


def _write_manifest(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.sha256":
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (directory / "manifest.sha256").write_text("".join(entries), encoding="utf-8")


def _r2_package(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "r2"
    directory.mkdir()
    event = {
        "candidate_id": "demo.hdf5::demo_0::0",
        "family": "anonymous_family_000",
        "binding_scope": "event_effect_anchored",
        "replay_label": "non_applicable",
    }
    (directory / "event_bound_validation.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (directory / "family_existential_ablation.jsonl").write_text(json.dumps({**event, "replay_label": "applicable"}) + "\n", encoding="utf-8")
    (directory / "unresolved_cases.jsonl").write_text("", encoding="utf-8")
    report = {
        "total_candidates": 1,
        "non_applicable": 1,
        "applicable": 0,
        "unresolved": 0,
        "unresolved_rate": 0.0,
        "unresolved_breakdown": {"binding_missing": 0, "oracle_unresolved": 0, "data_missing": 0},
        "pass_conditions": {"unresolved_rate_actual": 0.0},
        "per_family": {"anonymous_family_000": {"total": 1, "non_applicable": 1, "applicable": 0, "unresolved": 0}},
    }
    (directory / "r2_negative_validation_report_v3.json").write_text(json.dumps(report), encoding="utf-8")
    resolver = {"total_predicate_entities": 1, "resolved": 1, "unresolved": 0, "hit_rate": 1.0, "entries": {"cube": {"resolved": "cube_main"}}}
    (directory / "oracle_body_resolver_report.json").write_text(json.dumps(resolver), encoding="utf-8")
    (directory / "environment_versions.json").write_text("{}", encoding="utf-8")
    _write_manifest(directory)
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"file": "demo.hdf5", "demo": "demo_0", "frame_t": 0, "family": "anonymous_family_000"}) + "\n", encoding="utf-8")
    return directory, candidates


def test_r2_audit_accepts_consistent_rows_and_report(tmp_path):
    directory, candidates = _r2_package(tmp_path)
    report = audit_r2_package(directory, expected_candidates=1, local_candidates=candidates)
    assert report["status"] == "PASS"
    assert report["candidate_multiset_matches_local"] is True


def test_r2_audit_rejects_stale_per_family_aggregate(tmp_path):
    directory, candidates = _r2_package(tmp_path)
    report_path = directory / "r2_negative_validation_report_v3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["per_family"]["anonymous_family_000"]["unresolved"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_manifest(directory)
    audited = audit_r2_package(directory, expected_candidates=1, local_candidates=candidates)
    assert audited["status"] == "FAIL"
    assert "report per_family disagrees with event rows" in audited["errors"]


def test_r3_audit_accepts_complete_counterfactual_batch(tmp_path):
    directory = tmp_path / "r3"
    directory.mkdir()
    witness = {"witness_id": "w_0001"}
    case = {**witness, "before_shortcut_succeeds": True, "after_shortcut_rejected": True, "after_full_plan_succeeds": True}
    (directory / "r3_witness_cases.jsonl").write_text(json.dumps(witness) + "\n", encoding="utf-8")
    report = {
        "candidate_count": 1,
        "evaluated_count": 1,
        "blocked_count": 0,
        "before_shortcut_success": 1,
        "after_shortcut_rejected": 1,
        "after_full_plan_success": 1,
        "input_sha256": {"domain": "x"},
        "cases": [case],
    }
    (directory / "r3_witness_acceptance_report.json").write_text(json.dumps(report), encoding="utf-8")
    _write_manifest(directory)
    audited = audit_r3_package(directory, expected_witnesses=1)
    assert audited["status"] == "PASS"


def test_r3_audit_accepts_server_schema_and_results_jsonl(tmp_path):
    directory = tmp_path / "r3_server"
    directory.mkdir()
    witness = {"witness_id": "w_0001"}
    case = {**witness, "before_shortcut": True, "after_shortcut_rejected": True, "after_full_plan": True}
    result = {**witness, "before_shortcut_succeeds": True, "after_shortcut_rejected": True, "after_full_plan_succeeds": True}
    (directory / "r3_witness_cases.jsonl").write_text(json.dumps(witness) + "\n", encoding="utf-8")
    (directory / "r3_witness_acceptance_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    report = {
        "overall_status": "PASS",
        "candidate_count": 1,
        "evaluated_count": 1,
        "blocked_count": 0,
        "input_sha256": "x",
        "stage_results": {"before_shortcut": {"passed": 1}, "after_shortcut": {"rejected": 1}, "after_full_plan": {"succeeded": 1}},
        "cases": [case],
    }
    (directory / "r3_witness_acceptance_report.json").write_text(json.dumps(report), encoding="utf-8")
    _write_manifest(directory)
    audited = audit_r3_package(directory, expected_witnesses=1)
    assert audited["status"] == "PASS"
    assert audited["computed"]["schema"] == "server_v2"
