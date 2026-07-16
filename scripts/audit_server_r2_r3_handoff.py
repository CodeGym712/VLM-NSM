"""Independently audit server-returned R2/R3 research artifact packages.

The audit treats JSONL evidence as authoritative and rejects reports whose
aggregate fields disagree with the rows they summarize.  It is deliberately
read-only with respect to the returned package: its report is written only to
the caller-provided local output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


UNRESOLVED_REASONS = ("binding_missing", "oracle_unresolved", "data_missing")
R2_REQUIRED_FILES = {
    "event_bound_validation.jsonl",
    "family_existential_ablation.jsonl",
    "r2_negative_validation_report_v3.json",
    "oracle_body_resolver_report.json",
    "unresolved_cases.jsonl",
    "environment_versions.json",
    "manifest.sha256",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.sha256"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"ok": False, "errors": ["missing manifest.sha256"], "entries": 0}

    expected: dict[str, str] = {}
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"invalid manifest entry on line {number}")
            continue
        digest, name = parts
        path = directory / name
        if path.parent != directory or not path.is_file():
            errors.append(f"manifest references missing or nested file: {name}")
            continue
        expected[name] = digest.lower()
        if sha256(path) != digest.lower():
            errors.append(f"sha256 mismatch: {name}")

    actual_files = {path.name for path in directory.iterdir() if path.is_file() and path.name != "manifest.sha256"}
    if actual_files != set(expected):
        errors.append("manifest file set does not match package files")
    return {"ok": not errors, "errors": errors, "entries": len(expected)}


def _label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("replay_label", "missing")) for row in rows)


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family", ""))].append(row)
    result: dict[str, dict[str, int]] = {}
    for family, members in grouped.items():
        labels = _label_counts(members)
        result[family] = {
            "total": len(members),
            "non_applicable": labels["non_applicable"],
            "applicable": labels["applicable"],
            "unresolved": labels["unresolved"],
        }
    return dict(sorted(result.items()))


def _candidate_keys(rows: list[dict[str, Any]], *, server: bool) -> Counter[str]:
    if server:
        return Counter(f"{row.get('candidate_id', '')}\t{row.get('family', '')}" for row in rows)
    return Counter(
        f"{row.get('file', '')}::{row.get('demo', '')}::{row.get('frame_t', '')}\t{row.get('family', '')}"
        for row in rows
    )


def audit_r2_package(directory: Path, expected_candidates: int, local_candidates: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    missing_files = sorted(name for name in R2_REQUIRED_FILES if not (directory / name).is_file())
    if missing_files:
        return {"status": "FAIL", "errors": [f"missing required files: {', '.join(missing_files)}"]}

    manifest = audit_manifest(directory)
    errors.extend(manifest["errors"])
    events = read_jsonl(directory / "event_bound_validation.jsonl")
    ablation = read_jsonl(directory / "family_existential_ablation.jsonl")
    unresolved_cases = read_jsonl(directory / "unresolved_cases.jsonl")
    report = read_json(directory / "r2_negative_validation_report_v3.json")
    resolver = read_json(directory / "oracle_body_resolver_report.json")

    event_labels = _label_counts(events)
    ablation_labels = _label_counts(ablation)
    family_counts = _family_counts(events)
    unresolved_breakdown = Counter(str(row.get("unresolved_reason", "missing")) for row in unresolved_cases)
    computed = {
        "total_candidates": len(events),
        "non_applicable": event_labels["non_applicable"],
        "applicable": event_labels["applicable"],
        "unresolved": event_labels["unresolved"],
        "unresolved_rate": event_labels["unresolved"] / len(events) if events else None,
        "event_effect_anchored_rows": sum(row.get("binding_scope") == "event_effect_anchored" for row in events),
        "ablation_total": len(ablation),
        "ablation_applicable": ablation_labels["applicable"],
        "ablation_non_applicable": ablation_labels["non_applicable"],
        "unresolved_breakdown": {reason: unresolved_breakdown[reason] for reason in UNRESOLVED_REASONS},
        "per_family": family_counts,
    }

    if len(events) != expected_candidates:
        errors.append(f"event row count {len(events)} != expected {expected_candidates}")
    if len(ablation) != len(events):
        errors.append("ablation row count differs from event-bound row count")
    if computed["event_effect_anchored_rows"] != len(events):
        errors.append("some event rows are not event_effect_anchored")
    if len(unresolved_cases) != computed["unresolved"]:
        errors.append("unresolved_cases row count disagrees with event rows")
    for field in ("total_candidates", "non_applicable", "applicable", "unresolved"):
        if report.get(field) != computed[field]:
            errors.append(f"report {field} disagrees with event rows")
    if report.get("unresolved_rate") != computed["unresolved_rate"]:
        errors.append("report unresolved_rate disagrees with event rows")
    if report.get("pass_conditions", {}).get("unresolved_rate_actual") != computed["unresolved_rate"]:
        errors.append("report pass_conditions.unresolved_rate_actual disagrees with event rows")
    if report.get("unresolved_breakdown") != computed["unresolved_breakdown"]:
        errors.append("report unresolved_breakdown disagrees with unresolved_cases")
    if report.get("per_family") != family_counts:
        errors.append("report per_family disagrees with event rows")

    entries = resolver.get("entries", {})
    if resolver.get("total_predicate_entities") != len(entries):
        errors.append("resolver total_predicate_entities disagrees with entries")
    resolved_entries = sum(item.get("resolved") is not None for item in entries.values())
    if resolver.get("resolved") != resolved_entries:
        errors.append("resolver resolved count disagrees with entries")
    if resolver.get("unresolved") != len(entries) - resolved_entries:
        errors.append("resolver unresolved count disagrees with entries")
    if resolver.get("hit_rate") != (resolved_entries / len(entries) if entries else 0.0):
        errors.append("resolver hit_rate disagrees with entries")

    candidate_match: bool | None = None
    if local_candidates is not None:
        local_keys = _candidate_keys(read_jsonl(local_candidates), server=False)
        server_keys = _candidate_keys(events, server=True)
        candidate_match = server_keys == local_keys
        if not candidate_match:
            errors.append("server candidate/family multiset differs from local input")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest": manifest,
        "computed": computed,
        "candidate_multiset_matches_local": candidate_match,
        "resolver": {"entries": len(entries), "resolved": resolved_entries, "unresolved": len(entries) - resolved_entries},
    }


def audit_r3_package(directory: Path, expected_witnesses: int) -> dict[str, Any]:
    errors: list[str] = []
    report_path = directory / "r3_witness_acceptance_report.json"
    witnesses_path = directory / "r3_witness_cases.jsonl"
    if not report_path.is_file() or not witnesses_path.is_file():
        return {"status": "FAIL", "errors": ["missing R3 acceptance report or witness JSONL"]}
    manifest = audit_manifest(directory)
    errors.extend(manifest["errors"])
    report = read_json(report_path)
    witnesses = read_jsonl(witnesses_path)
    cases = report.get("cases", [])
    witness_ids = [str(row.get("witness_id", "")) for row in witnesses]
    case_ids = [str(row.get("witness_id", "")) for row in cases]
    evaluated = [row for row in cases if row.get("status") is None]
    server_schema = "overall_status" in report or "stage_results" in report

    def case_value(row: dict[str, Any], canonical: str) -> bool:
        fields = {
            "before_shortcut_success": ("before_shortcut_succeeds", "before_shortcut"),
            "after_shortcut_rejected": ("after_shortcut_rejected",),
            "after_full_plan_success": ("after_full_plan_succeeds", "after_full_plan"),
        }
        return bool(next((row[field] for field in fields[canonical] if field in row), False))

    condition_counts = {
        "before_shortcut_success": sum(case_value(row, "before_shortcut_success") for row in evaluated),
        "after_shortcut_rejected": sum(case_value(row, "after_shortcut_rejected") for row in evaluated),
        "after_full_plan_success": sum(case_value(row, "after_full_plan_success") for row in evaluated),
    }
    if len(witnesses) != expected_witnesses:
        errors.append(f"witness row count {len(witnesses)} != expected {expected_witnesses}")
    if len(set(witness_ids)) != len(witness_ids) or "" in witness_ids:
        errors.append("witness JSONL has missing or duplicate witness_id")
    if Counter(witness_ids) != Counter(case_ids):
        errors.append("report cases do not match witness JSONL")
    for field, expected in (("candidate_count", len(witnesses)), ("evaluated_count", len(evaluated)), ("blocked_count", len(cases) - len(evaluated))):
        if report.get(field) != expected:
            errors.append(f"report {field} disagrees with cases")
    if server_schema:
        stages = report.get("stage_results", {})
        expected_stage_counts = {
            "before_shortcut_success": stages.get("before_shortcut", {}).get("passed"),
            "after_shortcut_rejected": stages.get("after_shortcut", {}).get("rejected"),
            "after_full_plan_success": stages.get("after_full_plan", {}).get("succeeded"),
        }
        for field, value in condition_counts.items():
            if expected_stage_counts[field] != value:
                errors.append(f"report stage_results disagrees with cases for {field}")
        if report.get("overall_status") != "PASS":
            errors.append("report overall_status is not PASS")
    else:
        for field, value in condition_counts.items():
            if report.get(field) != value:
                errors.append(f"report {field} disagrees with cases")
    if len(evaluated) != len(witnesses) or any(value != len(witnesses) for value in condition_counts.values()):
        errors.append("not every witness passes all three counterfactual checks")
    if not report.get("input_sha256"):
        errors.append("report is missing input_sha256")
    results_path = directory / "r3_witness_acceptance_results.jsonl"
    result_row_count: int | None = None
    if results_path.is_file():
        results = read_jsonl(results_path)
        result_row_count = len(results)
        result_ids = [str(row.get("witness_id", "")) for row in results]
        if Counter(result_ids) != Counter(witness_ids):
            errors.append("result JSONL does not match witness JSONL")
        for row in results:
            case = next((item for item in cases if item.get("witness_id") == row.get("witness_id")), None)
            if case is None or any(case_value(case, field) != case_value(row, field) for field in condition_counts):
                errors.append("result JSONL disagrees with report cases")
                break
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest": manifest,
        "computed": {
            "schema": "server_v2" if server_schema else "local_v1",
            "witness_count": len(witnesses),
            "evaluated_count": len(evaluated),
            "result_row_count": result_row_count,
            **condition_counts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-r2-candidates", type=Path)
    parser.add_argument("--expected-r2-candidates", type=int, default=519)
    parser.add_argument("--r3-dir", type=Path)
    parser.add_argument("--expected-r3-witnesses", type=int, default=57)
    args = parser.parse_args()

    r2 = audit_r2_package(args.r2_dir, args.expected_r2_candidates, args.local_r2_candidates)
    report: dict[str, Any] = {"r2": r2}
    if args.r3_dir is not None:
        report["r3"] = audit_r3_package(args.r3_dir, args.expected_r3_witnesses)
    report["status"] = "PASS" if all(section["status"] == "PASS" for section in report.values() if isinstance(section, dict)) else "FAIL"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "server_handoff_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "r2": r2["status"], "r3": report.get("r3", {}).get("status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
