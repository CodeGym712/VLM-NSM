"""Stage 19 dry-run symbolic execution — validates goals without robosuite.

Uses the real AlignmentLayer and target_stack_from_predicates to validate
symbolic goals, but does NOT run the controller or instantiate a sim.

CLI:
  python -m vlm_nsm.experiments.stage19_dry_run_symbolic_execution \
    --goal_suite generated/eval_inputs/stage_19/symbolic_goal_suite.json \
    --output generated/results/stage_19/stage19_dry_run_symbolic_execution.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.utils import ensure_dir
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates

VALID_PREDICATES = ["on", "clear", "free-gripper", "grasped", "smaller"]
TYPE_TABLE = {
    "cube1": "block", "cube2": "block", "table": "table",
}
PREDICATE_SIGNATURE = {
    "on": [["block", "block"], ["block", "table"]],
    "clear": [["block"]],
    "free-gripper": [[]],
    "grasped": [["block"]],
    "smaller": [["block", "block"], ["block", "table"]],
}


def load_goal_suite(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def validate_goal(goal_entry: dict, available_objects: list[str]) -> dict:
    """Validate a single symbolic goal through the real pipeline gates.

    Returns a dict with per-gate results and final decision.
    """
    goal_id = goal_entry["goal_id"]
    parser_output = goal_entry["parser_output"]
    expected = goal_entry.get("expected_execution", "execute")

    result: dict = {
        "goal_id": goal_id,
        "description": goal_entry.get("description", ""),
        "expected_execution": expected,
        "checks": {},
        "decision": "reject",
        "decision_reason": "",
        "correct": False,
    }

    # Gate 1: Ungroundable status check
    status = parser_output.get("status", "ok")
    result["checks"]["status_check"] = {
        "passed": status != "ungroundable",
        "detail": f"status={status}",
    }
    if status == "ungroundable":
        result["decision"] = "reject"
        result["decision_reason"] = f"parser status is 'ungroundable': {parser_output.get('reason', 'no reason')}"
        result["correct"] = (expected == "reject")
        return result

    # Gate 2: Schema validation
    alignment = AlignmentLayer()
    align_result = alignment.align_and_validate(
        goal_json=parser_output,
        valid_objects=available_objects + ["table"],
        valid_predicates=VALID_PREDICATES,
        type_table={k: v for k, v in TYPE_TABLE.items() if k in available_objects or k == "table"},
        predicate_signature=PREDICATE_SIGNATURE,
        task_type=parser_output.get("task_type", "stacking"),
        n_blocks=len(available_objects),
    )

    result["checks"]["alignment"] = {
        "passed": align_result.success,
        "detail": align_result.error if not align_result.success else "ok",
    }

    if not align_result.success:
        result["decision"] = "reject"
        result["decision_reason"] = f"alignment failed: {align_result.error}"
        result["correct"] = (expected == "reject")
        return result

    # Gate 3: Target stack derivation
    on_predicates = [
        (p.name, p.args[0], p.args[1])
        for p in align_result.goal_predicates
        if p.name == "on" and len(p.args) == 2
    ]

    try:
        target_stack = target_stack_from_predicates(on_predicates, available_objects)
    except ValueError as e:
        result["checks"]["target_stack"] = {
            "passed": False,
            "detail": str(e),
        }
        result["decision"] = "reject"
        result["decision_reason"] = f"target stack derivation failed: {e}"
        result["correct"] = (expected == "reject")
        return result

    result["checks"]["target_stack"] = {
        "passed": True,
        "detail": f"target_stack={target_stack}",
    }

    # Gate 4: Safety checks on resolved objects
    object_bindings = parser_output.get("object_bindings", {})
    top = object_bindings.get("top block", "")
    bottom = object_bindings.get("bottom block", "")

    result["checks"]["safety"] = {}

    # Self-reference check
    if top and bottom and top == bottom:
        result["checks"]["safety"]["self_reference"] = {"passed": False, "detail": f"top==bottom=={top}"}
        result["decision"] = "reject"
        result["decision_reason"] = f"self-referential binding: top and bottom are both {top}"
        result["correct"] = (expected == "reject")
        return result
    result["checks"]["safety"]["self_reference"] = {"passed": True, "detail": "ok"}

    # Object availability check
    target_objects = set(target_stack)
    unknown = target_objects - set(available_objects)
    if unknown:
        result["checks"]["safety"]["object_availability"] = {
            "passed": False,
            "detail": f"unknown objects: {sorted(unknown)}",
        }
        result["decision"] = "reject"
        result["decision_reason"] = f"unknown objects in target stack: {sorted(unknown)}"
        result["correct"] = (expected == "reject")
        return result
    result["checks"]["safety"]["object_availability"] = {"passed": True, "detail": "ok"}

    # All gates passed
    result["decision"] = "execute"
    result["decision_reason"] = f"all gates passed, target_stack={target_stack}"
    result["correct"] = (expected == "execute")
    result["target_stack"] = target_stack
    return result


def run(goal_suite_path: str, output_path: str, available_objects: list[str] | None = None) -> dict:
    suite = load_goal_suite(goal_suite_path)
    goals = suite.get("goals", [])
    available_objects = available_objects or suite.get("available_objects", ["cube1", "cube2"])

    results: list[dict] = []
    for goal in goals:
        result = validate_goal(goal, available_objects)
        results.append(result)

    # Aggregate counts
    allowed = [r for r in results if r["decision"] == "execute"]
    rejected = [r for r in results if r["decision"] == "reject"]
    correct = [r for r in results if r["correct"]]
    incorrect = [r for r in results if not r["correct"]]

    # Unsafe leaks: goals that should have been rejected but were allowed
    unsafe_leaks = [
        r for r in results
        if r["expected_execution"] == "reject" and r["decision"] == "execute"
    ]

    summary = {
        "suite_description": suite.get("description", ""),
        "date": suite.get("date", ""),
        "available_objects": available_objects,
        "total_goals": len(results),
        "allowed_count": len(allowed),
        "rejected_count": len(rejected),
        "invalid_goal_count": len(rejected),
        "unsafe_goal_leak_count": len(unsafe_leaks),
        "unsafe_leaks": [r["goal_id"] for r in unsafe_leaks],
        "correct_decisions": len(correct),
        "incorrect_decisions": len(incorrect),
        "accuracy": round(len(correct) / len(results), 4) if results else 0,
        "results": results,
    }

    output = Path(output_path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 19 dry-run symbolic execution")
    parser.add_argument(
        "--goal_suite",
        default="generated/eval_inputs/stage_19/symbolic_goal_suite.json",
        help="Path to symbolic goal suite JSON",
    )
    parser.add_argument(
        "--output",
        default="generated/results/stage_19/stage19_dry_run_symbolic_execution.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--available_objects",
        nargs="+",
        default=["cube1", "cube2"],
        help="Available object IDs",
    )
    args = parser.parse_args()

    summary = run(args.goal_suite, args.output, args.available_objects)

    print(f"\n=== Stage 19 Dry-Run Symbolic Execution ===")
    print(f"Total goals:       {summary['total_goals']}")
    print(f"Allowed:           {summary['allowed_count']}")
    print(f"Rejected:          {summary['rejected_count']}")
    print(f"Unsafe leaks:      {summary['unsafe_goal_leak_count']}")
    print(f"Correct decisions: {summary['correct_decisions']}/{summary['total_goals']}")
    print(f"Accuracy:          {summary['accuracy']:.1%}")

    if summary["unsafe_goal_leak_count"] > 0:
        print(f"\n⚠️  UNSAFE LEAKS DETECTED: {summary['unsafe_leaks']}")
    else:
        print(f"\n✅ No unsafe leaks — all rejectable goals blocked.")

    # Per-goal details
    print(f"\nPer-goal decisions:")
    for r in summary["results"]:
        status = "✅" if r["correct"] else "❌"
        decision_mark = "EXECUTE" if r["decision"] == "execute" else "REJECT"
        print(f"  {status} {r['goal_id']:40s} → {decision_mark:8s} ({r['decision_reason'][:60]})")


if __name__ == "__main__":
    main()
