from __future__ import annotations

import json
from typing import Any

from vlm_nsm.core.data_types import Action, Predicate
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor


def _hanoi_init(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    init = [Predicate("free-gripper", [])]
    init.append(Predicate("on", [cubes[-1], "peg1"]))
    for i in range(n_blocks - 1, 0, -1):
        init.append(Predicate("on", [cubes[i - 1], cubes[i]]))
    init.extend(Predicate("clear", [p]) for p in ["peg1", "peg2", "peg3"] if p != "peg1")
    init.append(Predicate("clear", [cubes[0]]))
    return init


def _hanoi_types(n_blocks: int) -> dict:
    return {f"cube{i}": "disk" for i in range(1, n_blocks + 1)} | {"peg1": "peg", "peg2": "peg", "peg3": "peg"}


def _hanoi_smaller(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    pegs = ["peg1", "peg2", "peg3"]
    smaller = []
    for i, d in enumerate(cubes):
        for ld in cubes[i + 1:]:
            smaller.append(Predicate("smaller", [d, ld]))
        for p in pegs:
            smaller.append(Predicate("smaller", [d, p]))
    return smaller


def _stacking_init(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    init = [Predicate("free-gripper", [])]
    for c in cubes:
        init.append(Predicate("on", [c, "table"]))
        init.append(Predicate("clear", [c]))
    return init


def _stacking_types(n_blocks: int) -> dict:
    return {f"cube{i}": "block" for i in range(1, n_blocks + 1)} | {"table": "table"}


def _stacking_smaller(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    smaller = []
    for i, d in enumerate(cubes):
        for ld in cubes[i + 1:]:
            smaller.append(Predicate("smaller", [d, ld]))
        smaller.append(Predicate("smaller", [d, "table"]))
    return smaller


def oracle_hanoi_plan(n_blocks: int) -> list[Action]:
    from vlm_nsm.planners.internal_hanoi_planner import InternalHanoiPlanner
    result = InternalHanoiPlanner().solve(n_blocks)
    return result.actions


def oracle_stacking_plan(n_blocks: int) -> list[Action]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = list(reversed(cubes))
    from vlm_nsm.planners.internal_stacking_planner import InternalStackingPlanner
    result = InternalStackingPlanner().solve(cubes=cubes, target_stack=target_stack)
    return result.actions


def _hanoi_goal(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    goals = [Predicate("on", [cubes[i], cubes[i + 1]]) for i in range(len(cubes) - 1)]
    goals.append(Predicate("on", [cubes[-1], "peg3"]))
    return goals


def _stacking_goal(n_blocks: int) -> list[Predicate]:
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    goals = [Predicate("on", [cubes[i], cubes[i + 1]]) for i in range(len(cubes) - 1)]
    goals.append(Predicate("on", [cubes[-1], "table"]))
    return goals


def _state_from_init(n_blocks: int, task: str) -> set:
    if task == "hanoi":
        init = _hanoi_init(n_blocks) + _hanoi_smaller(n_blocks)
    else:
        init = _stacking_init(n_blocks) + _stacking_smaller(n_blocks)
    return {p.as_tuple() for p in init}


class CorruptedPlanGenerator:
    @staticmethod
    def missing_action(n_blocks: int, task: str) -> list[Action]:
        oracle = oracle_hanoi_plan(n_blocks) if task == "hanoi" else oracle_stacking_plan(n_blocks)
        return oracle[:-1] if len(oracle) > 1 else []

    @staticmethod
    def invalid_object(n_blocks: int, task: str) -> list[Action]:
        oracle = oracle_hanoi_plan(n_blocks) if task == "hanoi" else oracle_stacking_plan(n_blocks)
        if not oracle:
            return []
        bad = list(oracle)
        # Replace first object arg with an invalid one
        args = list(bad[0].args)
        if args:
            args[0] = "blockX"
            bad[0] = Action(bad[0].name, args)
        return bad

    @staticmethod
    def invalid_action_format(n_blocks: int, task: str) -> list[Action]:
        oracle = oracle_hanoi_plan(n_blocks) if task == "hanoi" else oracle_stacking_plan(n_blocks)
        if not oracle:
            return []
        return [Action("fly", ["moon"])] + oracle

    @staticmethod
    def precondition_violation(n_blocks: int, task: str) -> list[Action]:
        """Try to pick a disk that is not clear (has another on top)."""
        if task == "hanoi" and n_blocks >= 3:
            # Try to pick cube3 (bottom disk) directly — it's on peg1 but cube1 is on top
            return [Action("pick", ["cube3", "peg1"])]
        return []

    @staticmethod
    def wrong_order(n_blocks: int, task: str) -> list[Action]:
        oracle = oracle_hanoi_plan(n_blocks) if task == "hanoi" else oracle_stacking_plan(n_blocks)
        return list(reversed(oracle))


def get_all_corrupted_plans(n_blocks: int, task: str) -> dict[str, list[Action]]:
    gen = CorruptedPlanGenerator()
    return {
        "missing_action": gen.missing_action(n_blocks, task),
        "invalid_object": gen.invalid_object(n_blocks, task),
        "invalid_action_format": gen.invalid_action_format(n_blocks, task),
        "precondition_violation": gen.precondition_violation(n_blocks, task),
        "wrong_order": gen.wrong_order(n_blocks, task),
    }


def run_direct_plan(
    task: str,
    n_blocks: int,
    actions: list[Action],
) -> dict[str, Any]:
    state = _state_from_init(n_blocks, task)
    goals = _hanoi_goal(n_blocks) if task == "hanoi" else _stacking_goal(n_blocks)

    executor = SymbolicStateExecutor()
    executor.reset(state)

    action_format_valid = True
    object_validity = True
    precondition_violation = False
    failed_step = None
    execution_success = True
    execution_error = None

    for idx, action in enumerate(actions):
        try:
            executor.execute(action)
        except ValueError as exc:
            execution_success = False
            failed_step = idx
            execution_error = str(exc)
            err_msg = str(exc).lower()
            if "precondition" in err_msg or "failed" in err_msg:
                precondition_violation = True
            break
        except Exception as exc:
            execution_success = False
            failed_step = idx
            execution_error = str(exc)
            break

    final_state = executor.get_state()
    goal_ok = goal_satisfied(final_state, goals) if execution_success else False

    return {
        "task": task,
        "n_blocks": n_blocks,
        "method": "direct_plan",
        "action_count": len(actions),
        "action_format_validity": action_format_valid,
        "object_validity": object_validity,
        "precondition_violation_rate": 1.0 if precondition_violation else 0.0,
        "plan_completeness": goal_ok,
        "execution_success": execution_success,
        "goal_success": goal_ok,
        "failed_step": failed_step,
        "error_type": execution_error,
    }


def run_baseline(task: str, n_blocks: int, mode: str = "oracle_direct_plan") -> dict[str, Any]:
    if mode == "oracle_direct_plan":
        if task == "hanoi":
            actions = oracle_hanoi_plan(n_blocks)
        else:
            actions = oracle_stacking_plan(n_blocks)
        result = run_direct_plan(task, n_blocks, actions)
        result["mode"] = "oracle_direct_plan"
        return result

    elif mode == "corrupted_direct_plan":
        corrupted = get_all_corrupted_plans(n_blocks, task)
        results = []
        for corr_name, corr_actions in corrupted.items():
            trial = run_direct_plan(task, n_blocks, corr_actions)
            trial["corruption_type"] = corr_name
            results.append(trial)

        summary = {
            "task": task,
            "n_blocks": n_blocks,
            "method": "direct_plan",
            "mode": "corrupted_direct_plan",
            "total_trials": len(results),
            "execution_success_rate": sum(1 for r in results if r["execution_success"]) / len(results) if results else 0.0,
            "goal_success_rate": sum(1 for r in results if r["goal_success"]) / len(results) if results else 0.0,
            "results": results,
        }
        return summary

    else:
        raise ValueError(f"unknown mode: {mode}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hanoi", "stacking"])
    parser.add_argument("--n_blocks", type=int, required=True, choices=[3, 4, 5])
    parser.add_argument("--mode", default="oracle_direct_plan", choices=["oracle_direct_plan", "corrupted_direct_plan"])
    args = parser.parse_args()

    result = run_baseline(args.task, args.n_blocks, args.mode)
    mode_tag = "oracle" if args.mode == "oracle_direct_plan" else "corrupted"
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"llm_direct_plan_{args.task}_{args.n_blocks}_{mode_tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"result_path={out_path}")


if __name__ == "__main__":
    main()
