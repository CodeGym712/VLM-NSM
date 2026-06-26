from __future__ import annotations

import argparse
import json
from typing import Any

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.failure_analyzer import classify_failure
from vlm_nsm.core.goal_parser import MockGoalParser, QwenTextGoalParser
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor

VALID_PREDICATES = ["on", "clear", "smaller", "free-gripper", "grasped"]
DEFAULT_INSTRUCTION = "stack the cubes onto each other"


def build_stacking_problem(n_blocks: int):
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    objects = cubes + ["table"]
    types = {c: "block" for c in cubes} | {"table": "table"}
    init = [Predicate("free-gripper", [])]
    for c in cubes:
        init.append(Predicate("on", [c, "table"]))
        init.append(Predicate("clear", [c]))
    return cubes, objects, types, init


def predicate_signature():
    return {
        "on": [["block", "block|table"]],
        "clear": [["block|table"]],
        "smaller": [["block", "block|table"]],
        "free-gripper": [[]],
        "grasped": [["block"]],
    }


def run_stacking_pipeline(
    n_blocks: int,
    parser_name: str = "mock",
    goal_parser: MockGoalParser | QwenTextGoalParser | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    planner_name: str = "internal_stacking",
    executor_name: str = "symbolic",
    write_result: bool = True,
    result_name: str | None = None,
) -> dict[str, Any]:
    if executor_name != "symbolic":
        raise ValueError(f"unsupported executor: {executor_name}")

    cubes, objects, types, init = build_stacking_problem(n_blocks)
    if goal_parser is None:
        if parser_name == "mock":
            goal_parser = MockGoalParser()
        else:
            raise ValueError(f"unknown parser: {parser_name}")

    run_id = result_name or f"stacking_{n_blocks}_{parser_name}"
    target_stack = list(reversed(cubes))
    parse_result = goal_parser.parse(
        instruction=instruction,
        task_type="stacking",
        objects=objects,
        n_blocks=n_blocks,
    )

    result: dict[str, Any] = {
        "task": "stacking",
        "n_blocks": n_blocks,
        "instruction": instruction,
        "parser": parser_name,
        "planner": planner_name,
        "executor": executor_name,
        "goal_json": parse_result.goal_json,
        "json_parse_success": parse_result.success,
        "alignment_success": False,
        "alignment_repaired": False,
        "repaired": False,
        "repair_types": [],
        "planner_success": False,
        "atomic_action_length": 0,
        "logical_plan_length": 0,
        "optimal_logical_length": n_blocks - 1,
        "plan_optimality_ratio": 0.0,
        "execution_success": False,
        "goal_success": False,
        "error_type": None,
    }

    if not parse_result.success:
        result["error_type"] = "json_parse_error"
        return _write_result(result, n_blocks, run_id, write_result)

    aligner = AlignmentLayer()
    alignment = aligner.align_and_validate(
        parse_result.goal_json,
        valid_objects=objects,
        valid_predicates=VALID_PREDICATES,
        type_table=types,
        predicate_signature=predicate_signature(),
        task_type="stacking",
        n_blocks=n_blocks,
    )
    result.update(
        {
            "alignment_success": alignment.success,
            "alignment_repaired": alignment.repaired,
            "repaired": alignment.repaired,
            "repair_types": _repair_types(alignment),
            "alignment_error": alignment.error,
            "alignment_details": alignment.details,
        }
    )
    if not alignment.success:
        result["error_type"] = classify_failure(alignment_result=alignment)
        return _write_result(result, n_blocks, run_id, write_result)

    problem = PlanningProblem(
        domain_name="stacking",
        objects=objects,
        types=types,
        init_predicates=init,
        goal_predicates=alignment.goal_predicates,
        problem_name=f"problem_generated_stacking_{n_blocks}_{_safe_name(run_id)}",
    )
    problem_path = PDDLGenerator().generate(problem)

    plan_result = PlannerAdapter(planner_name).solve(
        cubes=cubes,
        target_stack=target_stack,
    )

    executor = SymbolicStateExecutor()
    with_smaller = PDDLGenerator()._with_smaller_predicates(problem, list(init))
    executor.reset([p.as_tuple() for p in with_smaller])
    execution = executor.execute_plan(plan_result.actions) if plan_result.success else None
    final_state = execution.final_state if execution else executor.get_state()
    goal_ok = goal_satisfied(final_state, alignment.goal_predicates) if plan_result.success and execution else False

    atomic_count = len(plan_result.actions)
    logical_count = atomic_count  # stacking: atomic = logical
    optimal_logical = n_blocks - 1
    ratio = logical_count / optimal_logical if optimal_logical > 0 else 0.0

    result.update(
        {
            "generated_pddl_path": str(problem_path),
            "planner_success": plan_result.success,
            "planner_error": plan_result.error,
            "actions": [{"name": a.name, "args": a.args} for a in plan_result.actions],
            "atomic_action_length": atomic_count,
            "logical_plan_length": logical_count,
            "optimal_logical_length": optimal_logical,
            "plan_optimality_ratio": ratio,
            "execution_success": execution.success if execution else False,
            "execution_error": execution.error if execution else None,
            "goal_success": goal_ok,
            "metrics": {
                "n_blocks": n_blocks,
                "planner_success": plan_result.success,
                "atomic_action_length": atomic_count,
                "logical_plan_length": logical_count,
                "optimal_logical_length": optimal_logical,
                "plan_optimality_ratio": ratio,
                "execution_success": execution.success if execution else False,
                "goal_success": goal_ok,
            },
        }
    )
    result["error_type"] = classify_failure(
        alignment_result=alignment,
        plan_result=plan_result,
        execution_result=execution,
        goal_success=goal_ok,
    )
    return _write_result(result, n_blocks, run_id, write_result)


def _default_result_path(n_blocks: int, parser: str, planner: str, executor: str) -> Path:
    return ensure_dir(PROJECT_ROOT / "generated" / "results") / f"stacking_{n_blocks}_{parser}_{planner}_{executor}.json"


def _write_result(result, n_blocks, run_id, write_result):
    if not write_result:
        return result
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"{_safe_name(run_id) or f'stacking_{n_blocks}_symbolic'}.json"
    result["result_path"] = str(out_path)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _repair_types(alignment) -> list[str]:
    if not alignment.repaired:
        return []
    return ["predicate_alias"]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_blocks", type=int, required=True, choices=[3, 4, 5])
    parser.add_argument("--parser", default="mock", choices=["mock", "qwen_text"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--planner", default="internal_stacking", choices=["internal_stacking"])
    parser.add_argument("--executor", default="symbolic", choices=["symbolic"])
    args = parser.parse_args()

    result = run_stacking_pipeline(
        n_blocks=args.n_blocks,
        parser_name=args.parser,
        instruction=args.instruction,
        planner_name=args.planner,
        executor_name=args.executor,
        write_result=True,
        result_name=f"stacking_{args.n_blocks}_{args.parser}_{args.planner}_{args.executor}",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("result_path"):
        print(f"result_path={result['result_path']}")


if __name__ == "__main__":
    main()
