from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import test_alignment
import test_action_grounder
import test_internal_stacking_planner
import test_language_variation_mock
import test_mock_embodied_executor
import test_pddl_generator
import test_planner_adapter
import test_physical_predicate_extractor
import test_qwen_output_json_extraction
import test_symbolic_executor
import test_run_stacking_symbolic
import test_alignment_ablation
import test_fixed_goal_baseline
import test_llm_direct_plan_baseline
import test_result_field_names
import test_prompt_files_exist
import test_language_variation_args
import test_analyze_qwen_failures
import test_export_report_tables
import test_hard_language_datasets
import test_prompt_variants
import test_alignment_modes
import test_language_variation_hard_mock
import test_analyze_alignment_repairs


def run_test(fn):
    kwargs = {}
    if "tmp_path" in inspect.signature(fn).parameters:
        with tempfile.TemporaryDirectory() as d:
            kwargs["tmp_path"] = Path(d)
            fn(**kwargs)
    else:
        fn()


def main():
    modules = [
        test_alignment,
        test_action_grounder,
        test_internal_stacking_planner,
        test_language_variation_mock,
        test_mock_embodied_executor,
        test_pddl_generator,
        test_physical_predicate_extractor,
        test_planner_adapter,
        test_qwen_output_json_extraction,
        test_symbolic_executor,
        test_run_stacking_symbolic,
        test_alignment_ablation,
        test_fixed_goal_baseline,
        test_llm_direct_plan_baseline,
        test_result_field_names,
        test_prompt_files_exist,
        test_language_variation_args,
        test_analyze_qwen_failures,
        test_export_report_tables,
        test_hard_language_datasets,
        test_prompt_variants,
        test_alignment_modes,
        test_language_variation_hard_mock,
        test_analyze_alignment_repairs,
    ]
    total = 0
    for module in modules:
        for name, fn in sorted(vars(module).items()):
            if name.startswith("test_") and callable(fn):
                run_test(fn)
                total += 1
                print(f"PASS {module.__name__}.{name}")
    print(f"{total} tests passed")


if __name__ == "__main__":
    main()
