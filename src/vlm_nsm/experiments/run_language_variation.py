from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import PlanningProblem, Predicate
from vlm_nsm.core.goal_parser import MockGoalParser, QwenTextGoalParser
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor

VALID_PREDICATES = ["on", "clear", "smaller", "free-gripper", "grasped"]


def _check_schema_validity(goal_json: dict | None) -> bool:
    """Check whether parsed JSON has valid goal_predicates (list of lists)."""
    if goal_json is None:
        return False
    preds = goal_json.get("goal_predicates")
    if not isinstance(preds, list):
        return False
    if len(preds) == 0:
        return False
    for p in preds:
        if not isinstance(p, list) or len(p) < 1:
            return False
        if not isinstance(p[0], str):
            return False
    return True


def hanoi_optimal_moves(n: int) -> int:
    return 2**n - 1


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


def hanoi_predicate_signature():
    return {
        "on": [["disk", "disk|peg"]],
        "clear": [["disk|peg"]],
        "smaller": [["disk", "disk|peg"]],
        "free-gripper": [[]],
        "grasped": [["disk"]],
    }


def build_stacking_problem(n_blocks: int):
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    objects = cubes + ["table"]
    types = {c: "block" for c in cubes} | {"table": "table"}
    init = [Predicate("free-gripper", [])]
    for c in cubes:
        init.append(Predicate("on", [c, "table"]))
        init.append(Predicate("clear", [c]))
    return cubes, objects, types, init


def stacking_predicate_signature():
    return {
        "on": [["block", "block|table"]],
        "clear": [["block|table"]],
        "smaller": [["block", "block|table"]],
        "free-gripper": [[]],
        "grasped": [["block"]],
    }


def _predicates_from_goal_json(goal_json: dict) -> list[Predicate]:
    """Convert raw goal_predicates from parsed JSON to Predicate objects."""
    return [Predicate(p[0], p[1:]) for p in goal_json.get("goal_predicates", [])]


def run_single_sample(
    sample: dict[str, Any],
    task: str,
    n_blocks: int,
    parser_name: str,
    model_path: str | None,
    prompt_path: str | None,
    temperature: float,
    max_new_tokens: int,
    raw_output_dir: Path,
    goal_parser: MockGoalParser | QwenTextGoalParser | None = None,
    difficulty: str = "easy",
    prompt_variant: str = "full",
    alignment_mode: str = "full",
) -> dict[str, Any]:
    alignment_applicable = alignment_mode != "no_alignment"

    result: dict[str, Any] = {
        "sample_id": sample["id"],
        "task": task,
        "n_blocks": n_blocks,
        "category": sample.get("category", ""),
        "difficulty": sample.get("difficulty", difficulty),
        "instruction": sample["instruction"],
        "parser": parser_name,
        "model_name": sample.get("model_name", ""),
        "prompt_version": sample.get("prompt_version", ""),
        "prompt_variant": prompt_variant,
        "alignment_mode": alignment_mode,
        "alignment_applicable": alignment_applicable,
        "json_parse_success": False,
        "schema_validity": False,
        "alignment_success": None,
        "repaired": False,
        "repair_types": [],
        "planner_success": False,
        "execution_success": False,
        "goal_success": False,
        "error_type": None,
        "atomic_action_length": 0,
        "logical_plan_length": 0,
        "optimal_logical_length": 0,
        "plan_optimality_ratio": 0.0,
        "inference_time": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "raw_output_path": None,
    }

    if task == "hanoi":
        disks, pegs, objects, types, init = build_hanoi_problem(n_blocks)
        optimal_logical = hanoi_optimal_moves(n_blocks)
        pred_sig = hanoi_predicate_signature()
    else:
        cubes, objects, types, init = build_stacking_problem(n_blocks)
        optimal_logical = n_blocks - 1
        pred_sig = stacking_predicate_signature()

    goal_parser_: MockGoalParser | QwenTextGoalParser
    if goal_parser is not None:
        goal_parser_ = goal_parser
    elif parser_name == "mock":
        goal_parser_ = MockGoalParser()
    else:
        goal_parser_ = QwenTextGoalParser(
            model_path=model_path,
            prompt_path=prompt_path or str(PROJECT_ROOT / "prompts" / f"{task}_goal_parser.txt"),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    if isinstance(goal_parser_, MockGoalParser):
        parse_result = goal_parser_.parse(
            instruction=sample["instruction"],
            task_type=task,
            objects=objects,
            n_blocks=n_blocks,
        )
    else:
        if task == "hanoi":
            parse_result = goal_parser_.parse(
                instruction=sample["instruction"],
                objects=objects,
                valid_predicates=VALID_PREDICATES,
                type_table=types,
                task_type="hanoi",
                n_blocks=n_blocks,
                target_peg="peg3",
            )
        else:
            parse_result = goal_parser_.parse(
                instruction=sample["instruction"],
                objects=objects,
                valid_predicates=VALID_PREDICATES,
                type_table=types,
                task_type="stacking",
                n_blocks=n_blocks,
            )

    result["inference_time"] = parse_result.inference_time
    result["input_tokens"] = parse_result.input_tokens
    result["output_tokens"] = parse_result.output_tokens

    raw_output_path = raw_output_dir / f"{sample['id']}_raw.txt"
    raw_output_path.write_text(parse_result.raw_output or "", encoding="utf-8")
    result["raw_output_path"] = str(raw_output_path)

    result["json_parse_success"] = parse_result.success
    result["schema_validity"] = _check_schema_validity(parse_result.goal_json)

    if not parse_result.success:
        result["error_type"] = parse_result.error or "json_parse_error"
        return result

    if not result["schema_validity"]:
        result["error_type"] = "schema_invalid"
        return result

    # Alignment phase — mode dependent
    if alignment_mode == "no_alignment":
        # Skip alignment entirely, use raw predicates from parsed JSON
        goal_predicates = _predicates_from_goal_json(parse_result.goal_json)
        result["alignment_success"] = None
        result["alignment_applicable"] = False
        result["repaired"] = False
        result["repair_types"] = []
    else:
        # Configure AlignmentLayer per mode
        use_aliases = alignment_mode != "no_alias_repair" and alignment_mode != "no_repair"
        use_completeness = alignment_mode != "no_completeness_repair"

        aliases_arg = None if use_aliases else {}
        aligner = AlignmentLayer(aliases=aliases_arg)

        if task == "hanoi":
            alignment = aligner.align_and_validate(
                parse_result.goal_json,
                valid_objects=objects,
                valid_predicates=VALID_PREDICATES,
                type_table=types,
                predicate_signature=pred_sig,
                task_type="hanoi",
                target_peg="peg3" if use_completeness else "",
                n_blocks=n_blocks if use_completeness else None,
                allow_hanoi_autocomplete=use_completeness,
            )
        else:
            alignment = aligner.align_and_validate(
                parse_result.goal_json,
                valid_objects=objects,
                valid_predicates=VALID_PREDICATES,
                type_table=types,
                predicate_signature=pred_sig,
                task_type="stacking",
                n_blocks=n_blocks if use_completeness else None,
            )

        result["alignment_success"] = alignment.success
        result["repaired"] = alignment.repaired
        result["repair_types"] = alignment.repair_types if hasattr(alignment, "repair_types") else (
            ["predicate_alias"] if alignment.repaired else []
        )

        if not alignment.success:
            result["error_type"] = alignment.error or "alignment_error"
            return result

        goal_predicates = alignment.goal_predicates

    problem = PlanningProblem(
        domain_name=task,
        objects=objects,
        types=types,
        init_predicates=init,
        goal_predicates=goal_predicates,
        problem_name=f"{sample['id']}_problem",
    )
    PDDLGenerator().generate(problem)

    if task == "hanoi":
        plan_result = PlannerAdapter("internal_hanoi", source_peg="peg1", aux_peg="peg2", target_peg="peg3").solve(
            n_blocks=n_blocks,
        )
    else:
        plan_result = PlannerAdapter("bfs").solve(
            cubes=cubes,
            target_stack=list(reversed(cubes)),
        )

    result["planner_success"] = plan_result.success
    if not plan_result.success:
        result["error_type"] = plan_result.error or "planner_error"
        return result

    executor = SymbolicStateExecutor()
    with_smaller = PDDLGenerator()._with_smaller_predicates(problem, list(init))
    executor.reset([p.as_tuple() for p in with_smaller])
    execution = executor.execute_plan(plan_result.actions)
    final_state = execution.final_state if execution else executor.get_state()
    goal_ok = goal_satisfied(final_state, goal_predicates) if execution else False

    atomic_count = len(plan_result.actions)
    logical_count = atomic_count // 2 if task == "hanoi" else atomic_count
    ratio = logical_count / optimal_logical if optimal_logical > 0 else 0.0

    result["atomic_action_length"] = atomic_count
    result["logical_plan_length"] = logical_count
    result["optimal_logical_length"] = optimal_logical
    result["plan_optimality_ratio"] = ratio
    result["execution_success"] = execution.success if execution else False
    result["goal_success"] = goal_ok

    return result


def load_instructions(instruction_file: str) -> list[dict[str, Any]]:
    path = Path(instruction_file)
    if not path.exists():
        raise FileNotFoundError(f"instruction file not found: {instruction_file}")
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                samples.append(json.loads(stripped))
    return samples


def _resolve_instruction_file(task: str, difficulty: str, instruction_file: str | None) -> str:
    if instruction_file is not None:
        return instruction_file
    data_dir = PROJECT_ROOT / "data" / "language_variations"
    if difficulty == "easy":
        return str(data_dir / f"{task}_zh_en.jsonl")
    if difficulty == "hard":
        return str(data_dir / f"{task}_hard_zh_en.jsonl")
    # difficulty == "all": not used here; handled in main
    return str(data_dir / f"{task}_zh_en.jsonl")


def _resolve_prompt_path(task: str, prompt_variant: str, prompt_path: str | None) -> str | None:
    if prompt_path is not None:
        return prompt_path
    prompt_dir = PROJECT_ROOT / "prompts"
    if prompt_variant == "weak":
        return str(prompt_dir / f"{task}_goal_parser_weak.txt")
    if prompt_variant == "medium":
        return str(prompt_dir / f"{task}_goal_parser_medium.txt")
    return str(prompt_dir / f"{task}_goal_parser.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["hanoi", "stacking"])
    parser.add_argument("--n_blocks", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument("--parser", default="mock", choices=["mock", "qwen_text"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt_path", default=None)
    parser.add_argument("--instruction_file", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to first N samples (for smoke test)")
    parser.add_argument("--sample_ids", type=str, default=None,
                        help="Comma-separated list of sample IDs to process")
    parser.add_argument("--limit_categories", type=str, default=None,
                        help="Comma-separated list of categories to process")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name for output filenames")
    parser.add_argument("--prompt_version", type=str, default=None,
                        help="Prompt version tag recorded in results")
    parser.add_argument("--save_failures_only", action="store_true",
                        help="Only save results where goal_success is False")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "hard", "all"],
                        help="Dataset difficulty level")
    parser.add_argument("--prompt_variant", type=str, default="full",
                        choices=["full", "medium", "weak"],
                        help="Prompt variant to use")
    parser.add_argument("--alignment_mode", type=str, default="full",
                        choices=["full", "no_alignment", "no_repair",
                                 "no_alias_repair", "no_completeness_repair"],
                        help="Alignment layer mode")
    args = parser.parse_args()

    model_name = args.model_name or "mock"
    if args.parser == "qwen_text" and args.model_name is None:
        model_name = Path(args.model_path).name if args.model_path else "qwen_text"

    # Load samples
    if args.difficulty == "all":
        easy_file = _resolve_instruction_file(args.task, "easy", None)
        hard_file = _resolve_instruction_file(args.task, "hard", None)
        all_samples = load_instructions(easy_file) + load_instructions(hard_file)
        print(f"Loaded {len(all_samples)} samples (easy + hard)")
        # Tag difficulty
        for s in all_samples:
            if "difficulty" not in s:
                s["difficulty"] = "easy" if "hard" not in s.get("id", "") else "hard"
    else:
        instruction_file = _resolve_instruction_file(args.task, args.difficulty, args.instruction_file)
        all_samples = load_instructions(instruction_file)
        print(f"Loaded {len(all_samples)} samples from {instruction_file}")

    samples = list(all_samples)

    # Apply filters
    if args.limit_categories:
        allowed = {c.strip() for c in args.limit_categories.split(",")}
        samples = [s for s in samples if s.get("category", "") in allowed]
        print(f"Filtered by categories {allowed}: {len(samples)} samples remaining")

    if args.sample_ids:
        allowed = {s.strip() for s in args.sample_ids.split(",")}
        samples = [s for s in samples if s["id"] in allowed]
        print(f"Filtered by sample_ids: {len(samples)} samples remaining")

    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"Limited to first {args.max_samples} samples")

    # Tag samples with metadata
    for s in samples:
        s["model_name"] = model_name
        s["prompt_version"] = args.prompt_version or ""

    raw_output_dir = ensure_dir(PROJECT_ROOT / "generated" / "logs" / "qwen_outputs")
    results_dir = ensure_dir(args.output_dir or (PROJECT_ROOT / "generated" / "results"))

    # Resolve difficulty tag for output filename
    diff_tag = args.difficulty if args.difficulty != "all" else "all"

    # Pre-initialize QwenTextGoalParser once (not per sample) to avoid
    # repeated model loading and OOM from stale GPU allocations.
    shared_goal_parser: MockGoalParser | QwenTextGoalParser | None = None
    if args.parser == "qwen_text":
        prompt_path = _resolve_prompt_path(args.task, args.prompt_variant, args.prompt_path)
        shared_goal_parser = QwenTextGoalParser(
            model_path=args.model_path,
            prompt_path=prompt_path,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Model loaded with {args.prompt_variant} prompt, "
              f"processing {len(samples)} samples...")

    results: list[dict[str, Any]] = []
    for sample in samples:
        sample_difficulty = sample.get("difficulty", args.difficulty)
        result = run_single_sample(
            sample=sample,
            task=args.task,
            n_blocks=args.n_blocks,
            parser_name=args.parser,
            model_path=args.model_path,
            prompt_path=args.prompt_path,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            raw_output_dir=raw_output_dir,
            goal_parser=shared_goal_parser,
            difficulty=sample_difficulty,
            prompt_variant=args.prompt_variant,
            alignment_mode=args.alignment_mode,
        )
        results.append(result)
        status = "OK" if result["goal_success"] else "FAIL"
        err = f" [{result['error_type']}]" if result["error_type"] else ""
        print(f"  {status} {sample['id']}: parse={result['json_parse_success']} "
              f"align={result['alignment_success']} goal={result['goal_success']}{err}")

    # Filter to failures only if requested
    if args.save_failures_only:
        results = [r for r in results if not r["goal_success"]]
        print(f"Filtered to {len(results)} failure samples only")

    # Include prompt_variant and alignment_mode in filename to avoid overwrites
    pv_tag = args.prompt_variant if args.prompt_variant != "full" else "full"
    am_tag = args.alignment_mode if args.alignment_mode != "full" else "full"
    json_out = results_dir / f"language_variation_{args.task}_{diff_tag}_{model_name}_{pv_tag}_{am_tag}.json"
    json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults written to {json_out}")

    csv_out = results_dir / f"language_variation_{args.task}_{diff_tag}_{model_name}_{pv_tag}_{am_tag}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    print(f"CSV written to {csv_out}")

    total = len(results)
    json_ok = sum(1 for r in results if r["json_parse_success"])
    schema_ok = sum(1 for r in results if r["schema_validity"])
    align_applicable = sum(1 for r in results if r.get("alignment_applicable", True))
    align_ok = sum(1 for r in results if r.get("alignment_success") is True)
    planner_ok = sum(1 for r in results if r["planner_success"])
    exec_ok = sum(1 for r in results if r["execution_success"])
    goal_ok = sum(1 for r in results if r["goal_success"])
    repaired = sum(1 for r in results if r["repaired"])
    alignment_mode_label = args.alignment_mode
    align_rate_str = f"{align_ok}/{align_applicable} ({100*align_ok/max(align_applicable,1):.1f}%)" if align_applicable > 0 else "N/A"
    print(f"\n--- Summary for {args.task} ({model_name}, {diff_tag}, "
          f"{args.prompt_variant}, {alignment_mode_label}) ---")
    print(f"Total samples:       {total}")
    print(f"JSON parse rate:     {json_ok}/{total} ({100*json_ok/total:.1f}%)")
    print(f"Schema validity:     {schema_ok}/{total} ({100*schema_ok/total:.1f}%)")
    print(f"Alignment rate:      {align_rate_str}")
    print(f"Repair rate:         {repaired}/{total} ({100*repaired/total:.1f}%)")
    print(f"Planner rate:        {planner_ok}/{total} ({100*planner_ok/total:.1f}%)")
    print(f"Execution rate:      {exec_ok}/{total} ({100*exec_ok/total:.1f}%)")
    print(f"Goal success rate:   {goal_ok}/{total} ({100*goal_ok/total:.1f}%)")


if __name__ == "__main__":
    main()
