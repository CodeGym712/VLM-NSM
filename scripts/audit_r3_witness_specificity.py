"""Run local symbolic specificity checks for the R3 ``place_in`` edit.

This is a contract-level audit: it establishes that ``open(?container)`` is
bound to the destination container and leaves unrelated operators unchanged.
It is not a learned-revision or out-of-distribution result.
"""

from __future__ import annotations

import argparse
import copy
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


def without_open_container_precondition(domain: dict[str, Any]) -> dict[str, Any]:
    ablated = copy.deepcopy(domain)
    place_in = next(item for item in ablated["operators"] if item["name"] == "place_in")
    place_in["preconditions_pos"] = [atom for atom in place_in["preconditions_pos"] if atom != ["open", "?container"]]
    return ablated


def with_wrong_object_role_precondition(domain: dict[str, Any]) -> dict[str, Any]:
    revised = without_open_container_precondition(domain)
    place_in = next(item for item in revised["operators"] if item["name"] == "place_in")
    place_in["preconditions_pos"].append(["open", "?obj"])
    return revised


def run_specificity(domain: dict[str, Any], *, obj: str = "bowl", target: str = "drawer", other: str = "other_drawer", support: str = "table") -> dict[str, Any]:
    ablated = without_open_container_precondition(domain)
    revised = copy.deepcopy(domain)
    wrong_role = with_wrong_object_role_precondition(domain)
    closed = {("handempty",), ("on", obj, support), ("closed", target)}
    target_open = {("handempty",), ("on", obj, support), ("open", target)}
    wrong_open = {("handempty",), ("on", obj, support), ("closed", target), ("open", other)}
    shortcut = [("pick", [obj]), ("place_in", [obj, target])]
    place_on = [("pick", [obj]), ("place_on", [obj, support])]

    ablated_shortcut = replay(ablated, shortcut, closed)
    target_closed = replay(revised, shortcut, closed)
    target_open_result = replay(revised, shortcut, target_open)
    wrong_open_result = replay(revised, shortcut, wrong_open)
    wrong_role_result = replay(wrong_role, shortcut, target_open)
    pick_unchanged = replay(ablated, [("pick", [obj])], closed) == replay(revised, [("pick", [obj])], closed)
    open_unchanged = replay(ablated, [("open", [target])], closed) == replay(revised, [("open", [target])], closed)
    place_on_unchanged = replay(ablated, place_on, closed) == replay(revised, place_on, closed)
    checks = {
        "ablated_shortcut_succeeds": ablated_shortcut["ok"],
        "target_closed_rejected": not target_closed["ok"],
        "target_open_allowed": target_open_result["ok"],
        "wrong_container_open_rejected": not wrong_open_result["ok"],
        "wrong_object_role_rejected": not wrong_role_result["ok"],
        "pick_unchanged": pick_unchanged,
        "open_unchanged": open_unchanged,
        "place_on_unchanged": place_on_unchanged,
    }
    return {
        "stage": "R3",
        "status": "SPECIFICITY_PASS" if all(checks.values()) else "SPECIFICITY_FAIL",
        "candidate_edit": ["open", "?container"],
        "checks": checks,
        "evidence": {
            "target_closed": target_closed,
            "target_open": target_open_result,
            "wrong_container_open": wrong_open_result,
            "wrong_object_role": wrong_role_result,
        },
        "scope_note": "Local symbolic contract audit only; not evidence of learned revision or OOD generalization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, default=Path("generated/libero_audit/p4_domain_export/p4_domain.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/libero_audit/r3_witness_specificity_local"))
    args = parser.parse_args()
    report = run_specificity(json.loads(args.domain.read_text(encoding="utf-8")))
    report["input_sha256"] = {"domain": sha256(args.domain)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "r3_specificity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "r3_specificity_report.md").write_text(
        "# R3 witness edit specificity audit\n\n"
        f"Status: **{report['status']}**\n\n"
        "The audit checks target-container binding, wrong-container rejection, a wrong-role contrast, and non-place_in operator invariance. "
        "It does not claim learned revision or OOD generalization.\n",
        encoding="utf-8",
    )
    files = [path for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name != "manifest.sha256"]
    (args.output_dir / "manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
