#!/usr/bin/env python3
"""Analyze referential grounding failures from scene grounding eval results.

Categorizes failures by the type of referential expression used in the
instruction, helping identify which predicate types are most needed.

Usage:
    python scripts/analyze_referential_failures.py \\
        --results generated/results/stage_18/qwen_plus_rgb_detector_stereo_fixed/summary.json \\
        --instructions generated/eval_inputs/stage_18/scene_grounding_rgb_detector.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# Keywords indicating different reference types in instructions
REFERENCE_PATTERNS = {
    "spatial_directional": [
        "left", "right", "leftmost", "rightmost",
        "on the left", "on the right", "to the left", "to the right",
    ],
    "spatial_relative": [
        "closer to", "closest to", "farther from", "farthest from",
        "near", "front", "back", "behind",
        "further", "further away",
    ],
    "edge_anchor": [
        "edge", "corner", "side of the table", "end of the table",
        "front of the table", "back of the table",
    ],
    "color": [
        "colored", "color", "lighter", "darker", "bright", "dull",
        "red", "green", "blue", "yellow", "purple", "orange", "pink",
        "dark", "light",
    ],
    "stacking_order": [
        "on top", "onto", "stack", "on the", "bottom", "top block",
        "bottom block", "below", "under",
    ],
}


def classify_instruction(instruction: str) -> list[str]:
    """Return list of reference type labels found in the instruction."""
    instruction_lower = instruction.lower()
    types: list[str] = []
    for ref_type, patterns in REFERENCE_PATTERNS.items():
        for pattern in patterns:
            if pattern in instruction_lower:
                types.append(ref_type)
                break
    if not types:
        types.append("other")
    return types


def main():
    parser = argparse.ArgumentParser(
        description="Analyze referential grounding failures",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to eval summary.json or results JSON",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Path to instructions JSON (for reference classification)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for analysis JSON (optional)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)

    results = json.loads(results_path.read_text())

    # Map instruction text to result
    instruction_results: dict[str, dict] = {}
    for r in results.get("results", []):
        inst = r.get("instruction", "")
        instruction_results[inst] = r

    # Classify each instruction
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})

    for inst, res in instruction_results.items():
        types = classify_instruction(inst)
        success = res.get("fully_successful", res.get("alignment_success", False))
        for t in types:
            by_type[t]["total"] += 1
            if success:
                by_type[t]["success"] += 1
            else:
                by_type[t]["fail"] += 1

    # Print breakdown
    total_inst = len(instruction_results)
    total_ok = sum(1 for r in instruction_results.values()
                   if r.get("fully_successful", r.get("alignment_success", False)))
    total_fail = total_inst - total_ok

    print(f"{'=' * 70}")
    print("Referential Grounding Failure Analysis")
    print(f"{'=' * 70}")
    print(f"Total instructions: {total_inst}")
    print(f"Successful: {total_ok}  Failed: {total_fail}")
    print(f"Overall success rate: {total_ok / total_inst * 100:.1f}%" if total_inst else "")
    print()

    print(f"{'Reference Type':<25s} {'Total':>6s} {'OK':>6s} {'Fail':>6s} {'Rate':>8s}")
    print("-" * 55)
    for ref_type in sorted(by_type.keys()):
        stats = by_type[ref_type]
        rate = stats["success"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {ref_type:<23s} {stats['total']:>6d} {stats['success']:>6d} "
              f"{stats['fail']:>6d} {rate:>7.1f}%")
    print()

    # Failures by category from the results
    categories = defaultdict(int)
    for r in results.get("results", []):
        cat = r.get("category", r.get("failure_type", "unknown"))
        if not r.get("fully_successful", r.get("alignment_success", False)):
            categories[cat] += 1

    if categories:
        print("Failure categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    # List individual failures with reference types
    print()
    print("Individual failures:")
    for inst, res in sorted(instruction_results.items()):
        if not res.get("fully_successful", res.get("alignment_success", False)):
            types = classify_instruction(inst)
            print(f"  [{', '.join(types)}] {inst}")
            err = res.get("error_message", res.get("ungroundable_reason", ""))
            if err:
                print(f"    -> {err}")

    # Output JSON if requested
    if args.output:
        output_data = {
            "total": total_inst,
            "successful": total_ok,
            "failed": total_fail,
            "overall_rate": total_ok / total_inst if total_inst else 0,
            "by_reference_type": {
                ref_type: {
                    "total": s["total"],
                    "success": s["success"],
                    "fail": s["fail"],
                    "rate": s["success"] / s["total"] if s["total"] else 0,
                }
                for ref_type, s in by_type.items()
            },
            "by_category": dict(categories),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
