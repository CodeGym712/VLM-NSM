from __future__ import annotations

import json
from typing import Any

from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor


def fixed_hanoi_goal(n_blocks: int, target_peg: str = "peg3") -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    goals = [Predicate("on", [cubes[i], cubes[i + 1]]) for i in range(len(cubes) - 1)]
    goals.append(Predicate("on", [cubes[-1], target_peg]))
    return goals


def fixed_stacking_goal(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    goals = [Predicate("on", [cubes[i], cubes[i + 1]]) for i in range(len(cubes) - 1)]
    goals.append(Predicate("on", [cubes[-1], "table"]))
    return goals


def _hanoi_problem(n_blocks: int) -> PlanningProblem:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    pegs = ["peg1", "peg2", "peg3"]
    objects = cubes + pegs
    types = {c: "disk" for c in cubes} | {p: "peg" for p in pegs}
    init = [Predicate("free-gripper", [])]
    init.append(Predicate("on", [cubes[-1], "peg1"]))
    for i in range(n_blocks - 1, 0, -1):
        init.append(Predicate("on", [cubes[i - 1], cubes[i]]))
    init.extend(Predicate("clear", [p]) for p in pegs if p != "peg1")
    init.append(Predicate("clear", [cubes[0]]))
    return PlanningProblem("hanoi", objects, types, init, fixed_hanoi_goal(n_blocks))


def _stacking_problem(n_blocks: int) -> PlanningProblem:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    objects = cubes + ["table"]
    types = {c: "block" for c in cubes} | {"table": "table"}
    init = [Predicate("free-gripper", [])]
    for c in cubes:
        init.append(Predicate("on", [c, "table"]))
        init.append(Predicate("clear", [c]))
    return PlanningProblem("stacking", objects, types, init, fixed_stacking_goal(n_blocks))


def run_fixed_goal_baseline(task: str, n_blocks: int) -> dict[str, Any]:
    if task == "hanoi":
        problem = _hanoi_problem(n_blocks)
        plan_result = PlannerAdapter("internal_hanoi").solve(n_blocks=n_blocks)
        optimal_length = 2 ** n_blocks - 1
    else:
        problem = _stacking_problem(n_blocks)
        cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
        target_stack = list(reversed(cubes))
        plan_result = PlannerAdapter("bfs").solve(cubes=cubes, target_stack=target_stack)
        optimal_length = n_blocks - 1

    with_smaller = PDDLGenerator()._with_smaller_predicates(problem, list(problem.init_predicates))
    executor = SymbolicStateExecutor()
    executor.reset([p.as_tuple() for p in with_smaller])
    execution = executor.execute_plan(plan_result.actions) if plan_result.success else None
    final_state = execution.final_state if execution else executor.get_state()
    goal_ok = goal_satisfied(final_state, problem.goal_predicates) if plan_result.success and execution else False

    action_count = len(plan_result.actions) if plan_result.success else 0
    return {
        "task": task,
        "n_blocks": n_blocks,
        "method": "fixed_goal",
        "planner_success": plan_result.success,
        "action_count": action_count,
        "optimal_length": optimal_length,
        "plan_optimality_ratio": action_count / optimal_length if plan_result.success and optimal_length > 0 else 0.0,
        "execution_success": execution.success if execution else False,
        "goal_success": goal_ok,
        "error": plan_result.error if not plan_result.success else (execution.error if execution and not execution.success else None),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hanoi", "stacking"])
    parser.add_argument("--n_blocks", type=int, required=True, choices=[3, 4, 5])
    args = parser.parse_args()

    result = run_fixed_goal_baseline(args.task, args.n_blocks)
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"fixed_goal_{args.task}_{args.n_blocks}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"result_path={out_path}")


if __name__ == "__main__":
    main()
