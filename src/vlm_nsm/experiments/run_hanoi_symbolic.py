from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import GoalParseResult, PlanningProblem, Predicate
from vlm_nsm.core.failure_analyzer import classify_failure
from vlm_nsm.core.goal_parser import MockGoalParser, QwenTextGoalParser
from vlm_nsm.core.metrics import hanoi_optimal_moves, summarize_run
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor

VALID_PREDICATES = ["on", "clear", "smaller", "free-gripper", "grasped"]
DEFAULT_INSTRUCTION = "move the tower from peg1 to peg3"


def build_hanoi_problem(n_blocks: int, target_peg: str = "peg3"):
    disks = [f"cube{i}" for i in range(1, n_blocks + 1)]
    pegs = ["peg1", "peg2", "peg3"]
    objects = disks + pegs
    types = {d: "disk" for d in disks} | {p: "peg" for p in pegs}
    init = [Predicate("free-gripper", [])]
    init.append(Predicate("on", [disks[-1], "peg1"]))
    for i in range(n_blocks - 1, 0, -1):
        init.append(Predicate("on", [disks[i - 1], disks[i]]))
    init.extend(Predicate("clear", [p]) for p in pegs if p != "peg1")
    init.append(Predicate("clear", [disks[0]]))
    return disks, pegs, objects, types, init


def predicate_signature():
    return {
        "on": [["disk", "disk|peg"]],
        "clear": [["disk|peg"]],
        "smaller": [["disk", "disk|peg"]],
        "free-gripper": [[]],
        "grasped": [["disk"]],
    }


def make_goal_parser(args) -> MockGoalParser | QwenTextGoalParser:
    if args.parser == "mock":
        return MockGoalParser()
    prompt_path = args.prompt_path or str(PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt")
    if not args.model_path:
        raise SystemExit("--model_path is required when --parser qwen_text")
    return QwenTextGoalParser(
        model_path=args.model_path,
        prompt_path=prompt_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )


def parse_goal(
    goal_parser: MockGoalParser | QwenTextGoalParser,
    parser_name: str,
    instruction: str,
    objects: list[str],
    types: dict[str, str],
    n_blocks: int,
    target_peg: str,
) -> GoalParseResult:
    if parser_name == "mock":
        return goal_parser.parse(
            instruction=instruction,
            task_type="hanoi",
            objects=objects,
            n_blocks=n_blocks,
            source_peg="peg1",
            target_peg=target_peg,
        )
    return goal_parser.parse(
        instruction=instruction,
        objects=objects,
        valid_predicates=VALID_PREDICATES,
        type_table=types,
        task_type="hanoi",
        n_blocks=n_blocks,
        target_peg=target_peg,
    )


def run_hanoi_pipeline(
    n_blocks: int,
    parser_name: str = "mock",
    goal_parser: MockGoalParser | QwenTextGoalParser | None = None,
    model_path: str | None = None,
    prompt_path: str | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    planner_name: str = "internal_hanoi",
    executor_name: str = "symbolic",
    metric_ff_binary: str | None = None,
    save_raw_output: bool = False,
    write_result: bool = True,
    result_name: str | None = None,
) -> dict[str, Any]:
    if executor_name != "symbolic":
        raise ValueError(f"unsupported executor: {executor_name}")

    target_peg = "peg3"
    disks, pegs, objects, types, init = build_hanoi_problem(n_blocks, target_peg=target_peg)
    if goal_parser is None:
        if parser_name == "mock":
            goal_parser = MockGoalParser()
        elif parser_name == "qwen_text":
            if not model_path:
                raise ValueError("model_path is required for qwen_text")
            goal_parser = QwenTextGoalParser(
                model_path=model_path,
                prompt_path=prompt_path or str(PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt"),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"unknown parser: {parser_name}")

    run_id = result_name or f"hanoi_{n_blocks}_{parser_name}"
    parse_result = parse_goal(goal_parser, parser_name, instruction, objects, types, n_blocks, target_peg)
    if save_raw_output or parser_name == "qwen_text":
        _save_raw_output(run_id, parse_result.raw_output)

    result: dict[str, Any] = {
        "task": "hanoi",
        "n_blocks": n_blocks,
        "instruction": instruction,
        "parser": parser_name,
        "goal_json": parse_result.goal_json,
        "raw_output": parse_result.raw_output if save_raw_output else None,
        "json_parse_success": parse_result.success,
        "inference_time": parse_result.inference_time,
        "input_tokens": parse_result.input_tokens,
        "output_tokens": parse_result.output_tokens,
        "alignment_success": False,
        "alignment_repaired": False,
        "repaired": False,
        "repair_types": [],
        "planner_success": False,
        "atomic_action_length": 0,
        "logical_plan_length": 0,
        "optimal_logical_length": hanoi_optimal_moves(n_blocks),
        "plan_optimality_ratio": 0.0,
        "execution_success": False,
        "goal_success": False,
        "error_type": None,
    }

    if not parse_result.success:
        result["error_type"] = classify_failure(parse_result=parse_result)
        return _write_result_if_requested(result, n_blocks, run_id, write_result)

    aligner = AlignmentLayer()
    alignment = aligner.align_and_validate(
        parse_result.goal_json,
        valid_objects=objects,
        valid_predicates=VALID_PREDICATES,
        type_table=types,
        predicate_signature=predicate_signature(),
        task_type="hanoi",
        target_peg=target_peg,
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
        result["error_type"] = classify_failure(parse_result=parse_result, alignment_result=alignment)
        return _write_result_if_requested(result, n_blocks, run_id, write_result)

    problem = PlanningProblem(
        domain_name="hanoi",
        objects=objects,
        types=types,
        init_predicates=init,
        goal_predicates=alignment.goal_predicates,
        problem_name=f"problem_generated_hanoi_{n_blocks}_{_safe_name(run_id)}",
    )
    problem_path = PDDLGenerator().generate(problem)
    domain_path = PROJECT_ROOT / "domains" / "hanoi" / "domain.pddl"
    plan_result = PlannerAdapter(planner_name, metric_ff_binary=metric_ff_binary).solve(
        domain_path,
        problem_path,
        n_blocks=n_blocks,
    )

    executor = SymbolicStateExecutor()
    generated_init = PDDLGenerator()._with_smaller_predicates(problem, list(init))
    executor.reset([p.as_tuple() for p in generated_init])
    execution = executor.execute_plan(plan_result.actions) if plan_result.success else None
    final_state = execution.final_state if execution else executor.get_state()
    goal_ok = goal_satisfied(final_state, alignment.goal_predicates) if plan_result.success and execution else False

    atomic_count = len(plan_result.actions)
    logical_count = atomic_count // 2 if atomic_count else 0
    optimal_logical = hanoi_optimal_moves(n_blocks)
    ratio = logical_count / optimal_logical if optimal_logical > 0 else 0.0

    metrics = summarize_run(n_blocks, atomic_count, plan_result.success, execution.success if execution else False, goal_ok)
    result.update(
        {
            "problem_path": str(problem_path),
            "planner": planner_name,
            "planner_success": plan_result.success,
            "planner_error": plan_result.error,
            "plan": [{"name": a.name, "args": a.args} for a in plan_result.actions],
            "atomic_action_length": atomic_count,
            "logical_plan_length": logical_count,
            "optimal_logical_length": optimal_logical,
            "plan_optimality_ratio": ratio,
            "execution_success": execution.success if execution else False,
            "execution_error": execution.error if execution else None,
            "goal_success": goal_ok,
            "metrics": metrics,
        }
    )
    result["error_type"] = classify_failure(
        parse_result=parse_result,
        alignment_result=alignment,
        plan_result=plan_result,
        execution_result=execution,
        goal_success=goal_ok,
    )
    return _write_result_if_requested(result, n_blocks, run_id, write_result)


def _save_raw_output(run_id: str, raw_output: str) -> Path:
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "logs")
    path = out_dir / f"{_safe_name(run_id)}_{time.time_ns()}.raw.txt"
    path.write_text(raw_output or "", encoding="utf-8")
    return path


def _default_result_path(n_blocks: int, parser: str, planner: str, executor: str) -> Path:
    return ensure_dir(PROJECT_ROOT / "generated" / "results") / f"hanoi_{n_blocks}_{parser}_{planner}_{executor}.json"


def _write_result_if_requested(result: dict[str, Any], n_blocks: int, run_id: str, write_result: bool) -> dict[str, Any]:
    if not write_result:
        return result
    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"{_safe_name(run_id) or f'hanoi_{n_blocks}_symbolic'}.json"
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
    parser.add_argument("--prompt_path", default=str(PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt"))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--planner", default="internal_hanoi", choices=["internal_hanoi", "metric_ff", "fast_downward"])
    parser.add_argument("--save_raw_output", action="store_true")
    parser.add_argument("--executor", default="symbolic", choices=["symbolic"])
    args = parser.parse_args()

    result_name = f"hanoi_{args.n_blocks}_{args.parser}_{args.planner}_{args.executor}"
    result = run_hanoi_pipeline(
        n_blocks=args.n_blocks,
        parser_name=args.parser,
        model_path=args.model_path,
        prompt_path=args.prompt_path,
        instruction=args.instruction,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        planner_name=args.planner,
        executor_name=args.executor,
        save_raw_output=args.save_raw_output,
        write_result=True,
        result_name=result_name,
    )
    print(json.dumps(result.get("metrics", result), indent=2, ensure_ascii=False))
    if result.get("result_path"):
        print(f"result_path={result['result_path']}")


if __name__ == "__main__":
    main()
