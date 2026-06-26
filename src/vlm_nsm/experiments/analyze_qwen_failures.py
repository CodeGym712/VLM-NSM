from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


FAILURE_CLASSIFICATION = [
    ("json_parse_error", lambda r: not r.get("json_parse_success")),
    ("schema_invalid", lambda r: r.get("json_parse_success") and not r.get("schema_validity")),
    ("invalid_object", lambda r: r.get("error_type") == "invalid_object"),
    ("invalid_predicate", lambda r: r.get("error_type") == "invalid_predicate"),
    ("type_error", lambda r: r.get("error_type") == "type_error"),
    ("incomplete_goal", lambda r: r.get("error_type") == "incomplete_goal"),
    ("planner_fail", lambda r: r.get("error_type") == "planner_error"),
    ("execution_fail", lambda r: not r.get("execution_success") and r.get("planner_success")),
    ("goal_not_satisfied", lambda r: not r.get("goal_success") and r.get("execution_success")),
]

REPAIR_TYPE_ALIASES = {
    "alias_repair": "predicate_alias",
    "goal_completion": "goal_completion",
    "predicate_canonicalization": "predicate_canonicalization",
    "object_rejection": "object_rejection",
    "predicate_rejection": "predicate_rejection",
    "type_error_detected": "type_error_detected",
    "incomplete_goal_detected": "incomplete_goal_detected",
}


def classify_failure(result: dict) -> str:
    for reason, check in FAILURE_CLASSIFICATION:
        if check(result):
            return reason
    if result.get("error_type"):
        return f"unknown_{result['error_type']}"
    return "unknown_error"


def short_reason(result: dict) -> str:
    if not result.get("json_parse_success"):
        return "JSON parse failed"
    if not result.get("schema_validity"):
        return "Schema invalid (no goal_predicates)"
    if not result.get("alignment_success") and result.get("alignment_applicable", True):
        error = result.get("error_type", "alignment_error")
        if error == "invalid_object":
            return "Invalid object in goal"
        if error == "invalid_predicate":
            return "Invalid predicate in goal"
        if error == "type_error":
            return "Type error in predicate"
        if error == "incomplete_goal":
            return "Incomplete goal (missing predicates)"
        return f"Alignment failed: {error}"
    if not result.get("planner_success"):
        return "Planner failed to find plan"
    if not result.get("execution_success"):
        return "Symbolic execution failed"
    if not result.get("goal_success"):
        return "Goal not satisfied after execution"
    return "Success"


def collect_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fpath in sorted(results_dir.iterdir()):
        if not fpath.name.startswith("language_variation_") or not fpath.name.endswith(".json"):
            continue
        if fpath.name.endswith("_summary.json"):
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows.extend(data)
        except (json.JSONDecodeError, OSError):
            pass
    return rows


def extract_failure_cases(rows: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for r in rows:
        if r.get("goal_success"):
            continue
        cases.append({
            "sample_id": r.get("sample_id", ""),
            "model_name": r.get("model_name", ""),
            "task": r.get("task", ""),
            "difficulty": r.get("difficulty", ""),
            "category": r.get("category", ""),
            "prompt_variant": r.get("prompt_variant", ""),
            "alignment_mode": r.get("alignment_mode", ""),
            "instruction": r.get("instruction", ""),
            "json_parse_success": r.get("json_parse_success", False),
            "schema_validity": r.get("schema_validity", False),
            "alignment_success": r.get("alignment_success", False),
            "planner_success": r.get("planner_success", False),
            "execution_success": r.get("execution_success", False),
            "goal_success": r.get("goal_success", False),
            "error_type": r.get("error_type", ""),
            "repair_types": r.get("repair_types", []),
            "raw_output_path": r.get("raw_output_path", ""),
            "failure_type": classify_failure(r),
            "short_reason": short_reason(r),
        })
    return cases


def extract_repair_stats(rows: list[dict]) -> list[dict]:
    stats: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []

    for r in rows:
        if not r.get("repaired"):
            continue
        repair_types = r.get("repair_types", [])
        if isinstance(repair_types, list):
            for rt in repair_types:
                raw.append({
                    "model_name": r.get("model_name", ""),
                    "task": r.get("task", ""),
                    "difficulty": r.get("difficulty", ""),
                    "prompt_variant": r.get("prompt_variant", ""),
                    "alignment_mode": r.get("alignment_mode", ""),
                    "repair_type": rt,
                    "sample_id": r.get("sample_id", ""),
                    "instruction": r.get("instruction", ""),
                    "raw_output_path": r.get("raw_output_path", ""),
                })

    # Aggregate
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "count": 0, "sample_ids": [], "instructions": [], "raw_output_paths": [],
    })
    for item in raw:
        key = (item["model_name"], item["task"], item["difficulty"],
               item["prompt_variant"], item["alignment_mode"], item["repair_type"])
        groups[key]["count"] += 1
        groups[key]["sample_ids"].append(item["sample_id"])
        groups[key]["instructions"].append(item["instruction"])
        groups[key]["raw_output_paths"].append(item["raw_output_path"])

    for (model_name, task, difficulty, prompt_variant, alignment_mode, repair_type), g in sorted(groups.items()):
        stats.append({
            "model_name": model_name,
            "task": task,
            "difficulty": difficulty,
            "prompt_variant": prompt_variant,
            "alignment_mode": alignment_mode,
            "repair_type": repair_type,
            "count": g["count"],
            "example_sample_id": g["sample_ids"][0],
            "example_instruction": g["instructions"][0],
            "raw_output_path": g["raw_output_paths"][0] if g["raw_output_paths"] else "",
        })

    return stats


def main():
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    rows = collect_results(results_dir)
    if not rows:
        print("No language variation results found.")
        return

    # Failure cases
    failures = extract_failure_cases(rows)
    fail_json_path = results_dir / "qwen_failure_cases.json"
    fail_json_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Failure cases written to {fail_json_path} ({len(failures)} cases)")

    fail_csv_path = results_dir / "qwen_failure_cases.csv"
    fieldnames = [
        "sample_id", "model_name", "task", "difficulty", "category",
        "prompt_variant", "alignment_mode", "instruction",
        "json_parse_success", "schema_validity", "alignment_success",
        "planner_success", "execution_success", "goal_success", "error_type",
        "repair_types", "raw_output_path", "failure_type", "short_reason",
    ]
    with fail_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if failures:
            clean = []
            for fc in failures:
                row = {k: fc.get(k, "") for k in fieldnames}
                row["repair_types"] = json.dumps(fc.get("repair_types", []),
                                                  ensure_ascii=False)
                clean.append(row)
            writer.writerows(clean)
    print(f"Failure cases CSV written to {fail_csv_path} ({len(failures)} cases)")

    # Repair stats
    repair_stats = extract_repair_stats(rows)
    repair_json_path = results_dir / "qwen_alignment_repair_stats.json"
    repair_json_path.write_text(json.dumps(repair_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Repair stats written to {repair_json_path} ({len(repair_stats)} rows)")

    repair_csv_path = results_dir / "qwen_alignment_repair_stats.csv"
    if repair_stats:
        fieldnames = list(repair_stats[0].keys())
        with repair_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(repair_stats)
    else:
        fieldnames = [
            "model_name", "task", "difficulty", "prompt_variant",
            "alignment_mode", "repair_type", "count",
            "example_sample_id", "example_instruction", "raw_output_path",
        ]
        with repair_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Write a row indicating no repairs observed
            writer.writerow({
                "repair_type": "none_observed",
                "count": 0,
            })
    print(f"Repair stats CSV written to {repair_csv_path}")

    # Summary print
    total = len(rows)
    fail_count = len(failures)
    repaired_count = sum(1 for r in rows if r.get("repaired"))
    print(f"\nTotal samples: {total}, Failures: {fail_count}, Repaired: {repaired_count}")

    if failures:
        print("\nFailure by type:")
        type_counts: dict[str, int] = defaultdict(int)
        for f in failures:
            type_counts[f["failure_type"]] += 1
        for ftype, count in sorted(type_counts.items()):
            print(f"  {ftype}: {count}")

    if repair_stats:
        print("\nRepair stats:")
        for rs in repair_stats:
            print(f"  {rs['model_name']} / {rs['task']} / {rs['repair_type']}: {rs['count']} times (e.g. {rs['example_sample_id']})")


if __name__ == "__main__":
    main()
