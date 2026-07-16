"""Evaluate the batch R3 candidate witness cases with symbolic counterfactual replay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from run_r3_witness_revision_smoke import replay


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, default=Path("generated/libero_audit/p4_domain_export/p4_domain.json"))
    parser.add_argument("--witnesses", type=Path, default=Path("generated/libero_audit/r3_witness_cases/r3_witness_cases.jsonl"))
    parser.add_argument("--initial-results", type=Path, default=Path("generated/libero_audit/p5_symbolic_planning/p5_initial_planning_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("generated/libero_audit/r3_witness_cases/r3_witness_acceptance_report.json"))
    args = parser.parse_args()
    domain = json.loads(args.domain.read_text(encoding="utf-8"))
    initial = {row["demo_id"]: set(tuple(atom) for atom in row.get("initial_atoms", [])) for row in read_jsonl(args.initial_results)}
    rows = read_jsonl(args.witnesses)
    cases = []
    for witness in rows:
        state = initial.get(witness.get("source_demo_id"))
        if state is None:
            cases.append({"witness_id": witness["witness_id"], "status": "blocked_missing_initial_state"})
            continue
        ablated = copy.deepcopy(domain)
        place_in = next(item for item in ablated["operators"] if item["name"] == "place_in")
        place_in["preconditions_pos"] = [atom for atom in place_in["preconditions_pos"] if atom != ["open", "?container"]]
        full_plan = []
        for step in witness["observed_plan"]:
            op = next(item for item in domain["operators"] if item["name"] == step["operator"])
            binding = step.get("bindings", {})
            ground_args = [binding[parameter["name"]] for parameter in op["parameters"]]
            full_plan.append((step["operator"], ground_args))
        support_index = witness["supporting_step_index"]
        shortcut_plan = [step for index, step in enumerate(full_plan) if index != support_index]
        before = replay(ablated, shortcut_plan, state)
        revised = copy.deepcopy(ablated)
        place_in_revised = next(item for item in revised["operators"] if item["name"] == "place_in")
        if ["open", "?container"] not in place_in_revised["preconditions_pos"]:
            place_in_revised["preconditions_pos"].append(["open", "?container"])
        after_shortcut = replay(revised, shortcut_plan, state)
        after_full = replay(revised, full_plan, state)
        cases.append(
            {
                "witness_id": witness["witness_id"],
                "source_demo_id": witness["source_demo_id"],
                "before_shortcut_succeeds": before["ok"],
                "after_shortcut_rejected": not after_shortcut["ok"],
                "after_full_plan_succeeds": after_full["ok"],
                "before_reason": before["steps"][-1]["reason"] if before["steps"] else "empty_plan",
                "after_shortcut_reason": after_shortcut["steps"][-1]["reason"] if after_shortcut["steps"] else "empty_plan",
            }
        )
    valid = [case for case in cases if case.get("status") is None]
    report = {
        "stage": "R3",
        "status": "BATCH_SMOKE_PASS" if valid and all(case["before_shortcut_succeeds"] and case["after_shortcut_rejected"] and case["after_full_plan_succeeds"] for case in valid) else "BATCH_SMOKE_PARTIAL",
        "candidate_count": len(rows),
        "evaluated_count": len(valid),
        "blocked_count": len(cases) - len(valid),
        "before_shortcut_success": sum(case.get("before_shortcut_succeeds", False) for case in valid),
        "after_shortcut_rejected": sum(case.get("after_shortcut_rejected", False) for case in valid),
        "after_full_plan_success": sum(case.get("after_full_plan_succeeds", False) for case in valid),
        "input_sha256": {
            "domain": sha256(args.domain),
            "witnesses": sha256(args.witnesses),
            "initial_results": sha256(args.initial_results),
        },
        "scope_note": "Template-level counterfactual replay; not held-out learning evidence.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_paths = [path for path in sorted(args.output.parent.iterdir()) if path.is_file() and path.name != "manifest.sha256"]
    (args.output.parent / "manifest.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_paths), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
