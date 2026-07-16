"""Run confound-free local R3 target and role-binding specificity checks.

Every rejected ``place_in`` case explicitly satisfies ``holding(?obj)``.
Consequently, a rejection is accepted only when the missing precondition is
exactly ``open(?container)`` for the grounded destination container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from run_r3_witness_revision_smoke import replay
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as a package
    from scripts.run_r3_witness_revision_smoke import replay


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_confound_free_checks(
    domain: dict[str, Any],
    *,
    obj: str = "akita_black_bowl_1",
    top: str = "wooden_cabinet_1_top_level",
    middle: str = "wooden_cabinet_1_middle_level",
    bottom: str = "wooden_cabinet_1_bottom_level",
) -> dict[str, Any]:
    place_in = [("place_in", [obj, top])]
    cases = [
        ("target_closed__all_closed", "target_closed", {("holding", obj), ("closed", top), ("closed", middle), ("closed", bottom)}, "REJECT", top),
        ("target_closed__middle_open", "target_closed", {("holding", obj), ("closed", top), ("open", middle)}, "REJECT", top),
        ("target_closed__bottom_open", "target_closed", {("holding", obj), ("closed", top), ("open", bottom)}, "REJECT", top),
        ("target_open__top_open", "target_open", {("holding", obj), ("open", top), ("closed", middle), ("closed", bottom)}, "ALLOW", top),
        ("target_open__all_open", "target_open", {("holding", obj), ("open", top), ("open", middle), ("open", bottom)}, "ALLOW", top),
        ("binding__middle_open_target_top_closed", "binding_specificity", {("holding", obj), ("closed", top), ("open", middle)}, "REJECT", top),
        ("binding__bottom_open_target_top_closed", "binding_specificity", {("holding", obj), ("closed", top), ("open", bottom)}, "REJECT", top),
        ("binding__middle_bottom_open_target_top_closed", "binding_specificity", {("holding", obj), ("closed", top), ("open", middle), ("open", bottom)}, "REJECT", top),
        ("binding__top_open_target_middle_closed", "binding_specificity", {("holding", obj), ("open", top), ("closed", middle)}, "REJECT", middle),
    ]
    rows: list[dict[str, Any]] = []
    for test_id, family, state, expected, target in cases:
        result = replay(domain, [("place_in", [obj, target])], state)
        expected_open = ("open", target)
        if expected == "REJECT":
            missing = result["steps"][-1]["reason"] if result["steps"] else "empty_plan"
            passed = not result["ok"] and missing == f"missing_positive:{expected_open}"
        else:
            passed = result["ok"]
        rows.append(
            {
                "test_id": test_id,
                "test_family": family,
                "target_container": target,
                "state_atoms": [list(atom) for atom in sorted(state)],
                "expected": expected,
                "actual": "ALLOW" if result["ok"] else "REJECT",
                "actual_reason": result["steps"][-1]["reason"] if result["steps"] else "empty_plan",
                "holding_satisfied": ("holding", obj) in state,
                "pass": passed,
            }
        )
    family_summary = {
        family: {"total": len([row for row in rows if row["test_family"] == family]), "passed": len([row for row in rows if row["test_family"] == family and row["pass"]])}
        for family in sorted({row["test_family"] for row in rows})
    }
    return {
        "stage": "R3",
        "phase": "specificity_confound_free_local",
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "candidate_edit": ["open", "?container"],
        "rows": rows,
        "family_summary": family_summary,
        "scope_note": "Local symbolic precondition audit only; not evidence of learned revision or OOD generalization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, default=Path("generated/libero_audit/p4_domain_export/p4_domain.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/libero_audit/r3_witness_specificity_confound_free_local"))
    args = parser.parse_args()
    report = run_confound_free_checks(json.loads(args.domain.read_text(encoding="utf-8")))
    report["input_sha256"] = {"domain": sha256(args.domain)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "r3_specificity_confound_free_results.jsonl"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in report.pop("rows")), encoding="utf-8")
    report_path = args.output_dir / "r3_specificity_confound_free_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = [path for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": report["status"], "family_summary": report["family_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
