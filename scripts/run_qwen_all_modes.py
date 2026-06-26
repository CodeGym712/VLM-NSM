#!/usr/bin/env python3
"""Run all alignment modes for both tasks in a single model load.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_qwen_all_modes.py \
    --model_path /path/to/Qwen2.5-32B-Instruct \
    --model_name qwen25_32b --difficulty hard --prompt_variant full

Loads the model once, runs all 22+22 samples, processes through all 5
alignment modes, saves 10 result files.
"""

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from vlm_nsm.core.goal_parser import QwenTextGoalParser, MockGoalParser
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.pddl_generator import PDDLGenerator
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.data_types import PlanningProblem
from vlm_nsm.executors.symbolic_executor import SymbolicStateExecutor
from vlm_nsm.executors.predicate_monitor import goal_satisfied
from vlm_nsm.core.metrics import hanoi_optimal_moves
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.experiments.run_language_variation import (
    load_instructions,
    build_hanoi_problem,
    hanoi_predicate_signature,
    build_stacking_problem,
    stacking_predicate_signature,
    _resolve_prompt_path,
    _resolve_instruction_file,
    _predicates_from_goal_json,
    VALID_PREDICATES,
)

ALIGNMENT_MODES = [
    "full",
    "no_alignment",
    "no_repair",
    "no_alias_repair",
    "no_completeness_repair",
]


def process_one_sample(
    sample: dict,
    task: str,
    n_blocks: int,
    parse_result,
    raw_output_dir: Path,
    alignment_mode: str,
) -> dict:
    """Run alignment → planner → executor → monitor for one mode."""
    result = {
        "sample_id": sample["id"],
        "task": task,
        "n_blocks": n_blocks,
        "category": sample.get("category", ""),
        "difficulty": sample.get("difficulty", "hard"),
        "instruction": sample["instruction"],
        "parser": "qwen_text",
        "model_name": "qwen25_32b",
        "prompt_version": sample.get("prompt_version", ""),
        "prompt_variant": "full",
        "alignment_mode": alignment_mode,
        "json_parse_success": False,
        "alignment_success": False,
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
        "inference_time": parse_result.inference_time,
        "input_tokens": parse_result.input_tokens,
        "output_tokens": parse_result.output_tokens,
        "raw_output_path": "",
    }

    # Save raw output
    raw_path = raw_output_dir / f"{sample['id']}_raw.txt"
    raw_path.write_text(parse_result.raw_output or "", encoding="utf-8")
    result["raw_output_path"] = str(raw_path)

    result["json_parse_success"] = parse_result.success
    if not parse_result.success:
        result["error_type"] = parse_result.error or "json_parse_error"
        return result

    if task == "hanoi":
        disks, pegs, objects, types, init = build_hanoi_problem(n_blocks)
        optimal_logical = hanoi_optimal_moves(n_blocks)
        pred_sig = hanoi_predicate_signature()
    else:
        cubes, objects, types, init = build_stacking_problem(n_blocks)
        optimal_logical = n_blocks - 1
        pred_sig = stacking_predicate_signature()

    # Alignment
    if alignment_mode == "no_alignment":
        goal_predicates = _predicates_from_goal_json(parse_result.goal_json)
        result["alignment_success"] = True
        result["repaired"] = False
        result["repair_types"] = []
    else:
        use_aliases = alignment_mode not in ("no_alias_repair", "no_repair")
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
        if hasattr(alignment, "repaired"):
            result["repaired"] = alignment.repaired
        result["repair_types"] = (
            alignment.repair_types
            if hasattr(alignment, "repair_types") and alignment.repaired
            else (["predicate_alias"] if alignment.repaired else [])
        )

        if not alignment.success:
            result["error_type"] = alignment.error or "alignment_error"
            return result
        goal_predicates = alignment.goal_predicates

    # PDDL
    problem = PlanningProblem(
        domain_name=task,
        objects=objects,
        types=types,
        init_predicates=init,
        goal_predicates=goal_predicates,
        problem_name=f"{sample['id']}_problem",
    )
    PDDLGenerator().generate(problem)

    # Planner
    if task == "hanoi":
        plan_result = PlannerAdapter(
            "internal_hanoi", source_peg="peg1", aux_peg="peg2", target_peg="peg3"
        ).solve(n_blocks=n_blocks)
    else:
        plan_result = PlannerAdapter("internal_stacking").solve(
            cubes=cubes,
            target_stack=list(reversed(cubes)),
        )

    result["planner_success"] = plan_result.success
    if not plan_result.success:
        result["error_type"] = plan_result.error or "planner_error"
        return result

    # Executor
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


def save_results(results: list[dict], task: str, difficulty: str,
                 model_name: str, prompt_variant: str, alignment_mode: str,
                 results_dir: Path):
    """Save result JSON and CSV files."""
    filename = f"language_variation_{task}_{difficulty}_{model_name}_{prompt_variant}_{alignment_mode}"
    json_out = results_dir / f"{filename}.json"
    json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_out = results_dir / f"{filename}.csv"
    if results:
        import csv
        fieldnames = list(results[0].keys())
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  CSV: {csv_out}")

    # Print summary
    total = len(results)
    parse_ok = sum(1 for r in results if r["json_parse_success"])
    align_ok = sum(1 for r in results if r["alignment_success"])
    repaired = sum(1 for r in results if r["repaired"])
    planner_ok = sum(1 for r in results if r["planner_success"])
    exec_ok = sum(1 for r in results if r["execution_success"])
    goal_ok = sum(1 for r in results if r["goal_success"])

    print(f"\n--- Summary for {task} ({model_name}, {difficulty}, {prompt_variant}, {alignment_mode}) ---")
    print(f"Total samples:    {total}")
    print(f"JSON parse rate:  {parse_ok}/{total} ({100*parse_ok/total:.1f}%)" if total else "N/A")
    print(f"Alignment rate:   {align_ok}/{total} ({100*align_ok/total:.1f}%)")
    print(f"Repair rate:      {repaired}/{total} ({100*repaired/total:.1f}%)")
    print(f"Planner rate:     {planner_ok}/{total} ({100*planner_ok/total:.1f}%)")
    print(f"Execution rate:   {exec_ok}/{total} ({100*exec_ok/total:.1f}%)")
    print(f"Goal success rate:{goal_ok}/{total} ({100*goal_ok/total:.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name", default="qwen25_32b")
    p.add_argument("--difficulty", default="hard", choices=["easy", "hard"])
    p.add_argument("--prompt_variant", default="full", choices=["full", "weak"])
    p.add_argument("--n_blocks", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    raw_output_dir = ensure_dir(PROJECT_ROOT / "generated" / "logs" / "qwen_outputs")
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    samples_per_task = {"hanoi": [], "stacking": []}

    # Load samples for both tasks
    for task in ["hanoi", "stacking"]:
        instruction_file = _resolve_instruction_file(task, args.difficulty, None)
        samples = load_instructions(instruction_file)
        for s in samples:
            s["model_name"] = args.model_name
        if args.max_samples and args.max_samples < len(samples):
            samples = samples[:args.max_samples]
        samples_per_task[task] = samples
        print(f"Loaded {len(samples)} {task} samples from {instruction_file}")

    # Load model once
    print(f"\nLoading model from {args.model_path}...")
    first_task = "hanoi"
    prompt_path = _resolve_prompt_path(first_task, args.prompt_variant, None)
    parser = QwenTextGoalParser(
        model_path=args.model_path,
        prompt_path=prompt_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Model loaded with {args.prompt_variant} prompt")

    # Process both tasks with the shared model (swap prompt template between tasks)
    for task in ["hanoi", "stacking"]:
        samples = samples_per_task[task]
        if not samples:
            print(f"\nNo samples for {task}, skipping")
            continue

        # Update prompt template for this task
        prompt_path = _resolve_prompt_path(task, args.prompt_variant, None)
        parser.prompt_template = prompt_path.read_text(encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"Processing {len(samples)} {task} samples...")
        print(f"{'='*60}")

        # Run inference once per sample, accumulate results per mode
        mode_results: dict[str, list[dict]] = {m: [] for m in ALIGNMENT_MODES}

        for sample in samples:
            if task == "hanoi":
                disks, pegs, objects, types, init = build_hanoi_problem(args.n_blocks)
                parse_result = parser.parse(
                    instruction=sample["instruction"],
                    objects=objects,
                    valid_predicates=VALID_PREDICATES,
                    type_table=types,
                    task_type="hanoi",
                    n_blocks=args.n_blocks,
                    target_peg="peg3",
                )
            else:
                cubes, objects, types, init = build_stacking_problem(args.n_blocks)
                parse_result = parser.parse(
                    instruction=sample["instruction"],
                    objects=objects,
                    valid_predicates=VALID_PREDICATES,
                    type_table=types,
                    task_type="stacking",
                    n_blocks=args.n_blocks,
                )

            for mode in ALIGNMENT_MODES:
                result = process_one_sample(
                    sample, task, args.n_blocks, parse_result,
                    raw_output_dir, mode,
                )
                mode_results[mode].append(result)

            # Print per-sample status (from full alignment mode)
            r = mode_results["full"][-1]
            status = "OK" if r["goal_success"] else "FAIL"
            err = f" [{r['error_type']}]" if r["error_type"] else ""
            print(f"  {status} {sample['id']}: parse={r['json_parse_success']} "
                  f"align={r['alignment_success']} goal={r['goal_success']}{err}")

        # Save results for each mode
        for mode in ALIGNMENT_MODES:
            save_results(
                mode_results[mode], task, args.difficulty,
                args.model_name, args.prompt_variant, mode,
                results_dir,
            )

    print(f"\n{'='*60}")
    print("All done! Results saved.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
