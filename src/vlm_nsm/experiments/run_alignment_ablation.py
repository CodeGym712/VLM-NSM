from __future__ import annotations

import argparse
import json
from typing import Any

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.failure_analyzer import classify_failure
from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor

VALID_PREDICATES_HANOI = ["on", "clear", "smaller", "free-gripper", "grasped"]
VALID_PREDICATES_STACKING = ["on", "clear", "smaller", "free-gripper", "grasped"]


def _hanoi_objects(n_blocks):
    return [f"cube{i}" for i in range(1, n_blocks + 1)] + ["peg1", "peg2", "peg3"]


def _hanoi_types(n_blocks):
    return {f"cube{i}": "disk" for i in range(1, n_blocks + 1)} | {"peg1": "peg", "peg2": "peg", "peg3": "peg"}


def _hanoi_signature():
    return {
        "on": [["disk", "disk|peg"]],
        "clear": [["disk|peg"]],
        "smaller": [["disk", "disk|peg"]],
        "free-gripper": [[]],
        "grasped": [["disk"]],
    }


def _stacking_objects(n_blocks):
    return [f"cube{i}" for i in range(1, n_blocks + 1)] + ["table"]


def _stacking_types(n_blocks):
    return {f"cube{i}": "block" for i in range(1, n_blocks + 1)} | {"table": "table"}


def _stacking_signature():
    return {
        "on": [["block", "block|table"]],
        "clear": [["block|table"]],
        "smaller": [["block", "block|table"]],
        "free-gripper": [[]],
        "grasped": [["block"]],
    }


SYNTHETIC_GOALS: dict[str, list[dict[str, Any]]] = {
    "full_alignment": [
        {
            "label": "valid_hanoi",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "cube1", "cube2"],
                    ["on", "cube2", "cube3"],
                    ["on", "cube3", "peg3"],
                ],
            },
            "expected_errors": [],
        },
        {
            "label": "valid_stacking",
            "goal": {
                "task_type": "stacking",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "cube1", "cube2"],
                    ["on", "cube2", "cube3"],
                    ["on", "cube3", "table"],
                ],
            },
            "expected_errors": [],
        },
    ],
    "no_alignment": [
        {
            "label": "alias_predicate",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["above", "cube1", "cube2"],
                    ["on", "cube2", "cube3"],
                    ["on", "cube3", "peg3"],
                ],
            },
            "expected_errors": [],
            "repair_expected": True,
        },
        {
            "label": "alias_handempty",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["handempty"],
                    ["on", "cube1", "cube2"],
                    ["on", "cube2", "cube3"],
                    ["on", "cube3", "peg3"],
                ],
            },
            "expected_errors": [],
            "repair_expected": True,
        },
    ],
    "no_alias_repair": [
        {
            "label": "invalid_object",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "blockA", "cube2"],
                ],
            },
            "expected_errors": ["invalid_object"],
        },
        {
            "label": "invalid_predicate",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["stacked_on", "cube1", "cube2"],
                ],
            },
            "expected_errors": ["invalid_predicate"],
        },
        {
            "label": "type_error",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "peg1", "cube1"],
                ],
            },
            "expected_errors": ["type_error"],
        },
    ],
    "no_completeness_repair": [
        {
            "label": "incomplete_hanoi",
            "goal": {
                "task_type": "hanoi",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "cube1", "cube2"],
                ],
            },
            "expected_errors": ["incomplete_goal"],
        },
        {
            "label": "incomplete_stacking",
            "goal": {
                "task_type": "stacking",
                "object_bindings": {},
                "goal_predicates": [
                    ["on", "cube1", "cube2"],
                ],
            },
            "expected_errors": ["incomplete_goal"],
        },
    ],
}


def run_single_alignment(
    task: str,
    n_blocks: int,
    goal_json: dict,
) -> dict[str, Any]:
    if task == "hanoi":
        objects = _hanoi_objects(n_blocks)
        types = _hanoi_types(n_blocks)
        preds = VALID_PREDICATES_HANOI
        sig = _hanoi_signature()
        target_peg = "peg3"
    else:
        objects = _stacking_objects(n_blocks)
        types = _stacking_types(n_blocks)
        preds = VALID_PREDICATES_STACKING
        sig = _stacking_signature()
        target_peg = None

    aligner = AlignmentLayer()
    alignment = aligner.align_and_validate(
        goal_json,
        valid_objects=objects,
        valid_predicates=preds,
        type_table=types,
        predicate_signature=sig,
        task_type=task,
        target_peg=target_peg,
        n_blocks=n_blocks,
    )

    result: dict[str, Any] = {
        "json_parse_success": True,
        "object_validity": alignment.success,
        "predicate_validity": alignment.success,
        "type_check_pass": alignment.success,
        "goal_completeness": alignment.success,
        "repaired": alignment.repaired,
        "repair_types": [],
        "planner_success": False,
        "execution_success": False,
        "goal_success": False,
        "error_type": "none" if alignment.success else alignment.error,
    }

    if alignment.repaired:
        result["repair_types"] = ["predicate_alias"]

    if not alignment.success:
        return result

    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    if task == "hanoi":
        init = [Predicate("free-gripper", [])] + [Predicate("on", [cubes[-1], "peg1"])]
        for i in range(n_blocks - 1, 0, -1):
            init.append(Predicate("on", [cubes[i - 1], cubes[i]]))
        init.extend(Predicate("clear", [p]) for p in ["peg1", "peg2", "peg3"] if p != "peg1")
        init.append(Predicate("clear", [cubes[0]]))
        problem = PlanningProblem("hanoi", objects, types, init, alignment.goal_predicates)
        plan_result = PlannerAdapter("internal_hanoi").solve(n_blocks=n_blocks)
    else:
        init = [Predicate("free-gripper", [])]
        for c in cubes:
            init.append(Predicate("on", [c, "table"]))
            init.append(Predicate("clear", [c]))
        target_stack = list(reversed(cubes))
        problem = PlanningProblem("stacking", objects, types, init, alignment.goal_predicates)
        plan_result = PlannerAdapter("bfs").solve(cubes=cubes, target_stack=target_stack)

    with_smaller = PDDLGenerator()._with_smaller_predicates(problem, list(init))
    executor = SymbolicStateExecutor()
    executor.reset([p.as_tuple() for p in with_smaller])
    execution = executor.execute_plan(plan_result.actions) if plan_result.success else None
    goal_ok = goal_satisfied(
        execution.final_state if execution else executor.get_state(),
        alignment.goal_predicates,
    ) if plan_result.success and execution else False

    result.update({
        "planner_success": plan_result.success,
        "execution_success": execution.success if execution else False,
        "goal_success": goal_ok,
    })
    return result


def run_alignment_ablation(task: str, n_blocks: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for ablation_mode, goals in SYNTHETIC_GOALS.items():
        for goal_spec in goals:
            # Skip invalid scenario type mismatches
            spec_task = "hanoi" if "cube3" in str(goal_spec) else "stacking"
            if "table" in str(goal_spec.get("goal", {})):
                spec_task = "stacking"
            if spec_task != task:
                continue
            # Skip stacking aliases for hanoi (above/on is hanoi specific)
            if task == "stacking" and goal_spec["label"] == "alias_predicate":
                continue
            if task == "hanoi" and "stacking" in goal_spec["label"]:
                continue
            if task == "stacking" and "hanoi" in goal_spec["label"]:
                continue

            trial = run_single_alignment(task, n_blocks, goal_spec["goal"])
            trial["ablation_mode"] = ablation_mode
            trial["label"] = goal_spec["label"]
            results.append(trial)

    total = len(results)
    summary = {
        "task": task,
        "n_blocks": n_blocks,
        "total_trials": total,
        "alignment_success_rate": sum(1 for r in results if r["object_validity"]) / total if total else 0.0,
        "repair_rate": sum(1 for r in results if r["repaired"]) / total if total else 0.0,
        "planner_success_rate": sum(1 for r in results if r["planner_success"]) / total if total else 0.0,
        "goal_success_rate": sum(1 for r in results if r["goal_success"]) / total if total else 0.0,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hanoi", "stacking"])
    parser.add_argument("--n_blocks", type=int, required=True, choices=[3, 4, 5])
    args = parser.parse_args()

    summary = run_alignment_ablation(args.task, args.n_blocks)
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"alignment_ablation_{args.task}_{args.n_blocks}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"result_path={out_path}")


if __name__ == "__main__":
    main()
