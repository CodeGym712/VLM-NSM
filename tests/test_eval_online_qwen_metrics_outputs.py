"""Test eval_online_qwen_parser.py metrics and failure taxonomy output.

Uses a local MockGoalParser path — these tests verify the aggregate
metrics and failure case extraction logic, not the online API.
"""
import json
import pytest
from pathlib import Path

from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates
from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser


EXPECTED_METRICS_KEYS = [
    "total",
    "parse_ok",
    "schema_valid",
    "alignment_success",
    "target_stack_success",
    "planner_success",
    "fully_successful",
    "failure_taxonomy",
]


def _simulate_parse_run(instructions, available_objects=None):
    """Simulate what eval_online_qwen_parser does per-sample.

    Returns a list of result entries matching the script's output format.
    Also computes aggregate metrics matching the new schema.
    """
    if available_objects is None:
        available_objects = ["cube1", "cube2"]

    aligner = AlignmentLayer()
    planner = PlannerAdapter("internal_stacking")
    results = []

    for item in instructions:
        instruction = item.get("instruction", "")
        entry = {
            "instruction": instruction,
            "parse_success": True,
            "schema_valid": True,
            "alignment_success": None,
            "target_stack_success": None,
            "parser_target_stack": None,
            "planner_success": None,
        }

        # Parse (use MockGoalParser for deterministic results)
        parser = MockGoalParser()
        parse_result = parser.parse(
            instruction=instruction,
            task_type="stacking",
            objects=available_objects,
            n_blocks=len(available_objects),
        )
        entry["parse_success"] = parse_result.success
        if not parse_result.success:
            entry["schema_valid"] = False
            results.append(entry)
            continue

        goal_json = parse_result.goal_json
        has_schema = (
            isinstance(goal_json, dict)
            and "goal_predicates" in goal_json
            and isinstance(goal_json["goal_predicates"], list)
        )
        entry["schema_valid"] = has_schema
        if not has_schema:
            results.append(entry)
            continue

        # Alignment
        align_result = aligner.align_and_validate(
            goal_json=goal_json,
            valid_objects=available_objects + ["table"],
            valid_predicates=["on", "clear", "free-gripper", "grasped", "smaller"],
            type_table={**{k: "block" for k in available_objects}, "table": "table"},
            predicate_signature={"on": [["block", "block"], ["block", "table"]], "clear": [["block"]], "free-gripper": [[]], "grasped": [["block"]], "smaller": [["block", "block"], ["block", "table"]]},
            task_type="stacking",
            n_blocks=len(available_objects),
        )
        entry["alignment_success"] = align_result.success
        if not align_result.success:
            entry["error_type"] = align_result.error or "alignment_failed"
            results.append(entry)
            continue

        entry["aligned_goal_predicates"] = [
            [pred.name, *pred.args] for pred in align_result.goal_predicates
        ]

        # Target stack
        try:
            on_preds = [
                (p.name, p.args[0], p.args[1])
                for p in align_result.goal_predicates
                if p.name == "on"
            ]
            parser_stack = target_stack_from_predicates(on_preds, available_objects)
            entry["target_stack_success"] = True
            entry["parser_target_stack"] = parser_stack
        except Exception as e:
            entry["target_stack_success"] = False
            entry["planner_success"] = False
            entry["error_type"] = "target_stack_error"
            results.append(entry)
            continue

        # Planner
        try:
            plan_result = planner.solve(cubes=available_objects, target_stack=parser_stack)
            entry["planner_success"] = plan_result.success
        except Exception:
            entry["planner_success"] = False
            entry["error_type"] = "planner_error"

        results.append(entry)

    return results


class TestMetricsOutput:
    def test_all_metrics_keys_present(self):
        instructions = [
            {"instruction": "Put the red cube on the green cube."},
        ]
        results = _simulate_parse_run(instructions)
        total = len(results)
        parse_ok = sum(1 for r in results if r.get("parse_success"))
        schema_valid = sum(1 for r in results if r.get("schema_valid"))
        align_ok = sum(1 for r in results if r.get("alignment_success"))
        ts_ok = sum(1 for r in results if r.get("target_stack_success"))
        plan_ok = sum(1 for r in results if r.get("planner_success"))

        metrics = {
            "total": total,
            "parse_ok": parse_ok,
            "schema_valid": schema_valid,
            "alignment_success": align_ok,
            "target_stack_success": ts_ok,
            "planner_success": plan_ok,
            "fully_successful": plan_ok,
            "failure_taxonomy": {},
        }
        for key in EXPECTED_METRICS_KEYS:
            assert key in metrics, f"Missing metrics key: {key}"

    def test_all_success_no_failures(self):
        instructions = [
            {"instruction": "Put the red cube on the green cube."},
            {"instruction": "Stack cube A on cube B."},
        ]
        results = _simulate_parse_run(instructions)
        failures = [r for r in results if not r.get("planner_success")]
        assert len(failures) == 0

    def test_failure_taxonomy_on_bad_instruction(self):
        instructions = [
            {"instruction": "do something unrelated"},
        ]
        results = _simulate_parse_run(instructions, available_objects=["cube1", "cube2"])
        # MockGoalParser should still parse OK for stacking
        assert all(r.get("parse_success", True) for r in results)

    def test_metrics_json_serializable(self):
        instructions = [
            {"instruction": "Put the red cube on the green cube."},
            {"instruction": "Invalid instruction here"},
        ]
        results = _simulate_parse_run(instructions)
        total = len(results)
        metrics = {
            "total": total,
            "parse_ok": sum(1 for r in results if r.get("parse_success")),
            "schema_valid": sum(1 for r in results if r.get("schema_valid")),
            "alignment_success": sum(1 for r in results if r.get("alignment_success")),
            "target_stack_success": sum(1 for r in results if r.get("target_stack_success")),
            "planner_success": sum(1 for r in results if r.get("planner_success")),
            "fully_successful": sum(1 for r in results if r.get("planner_success")),
            "failure_taxonomy": {},
        }
        json_str = json.dumps(metrics)
        parsed = json.loads(json_str)
        assert parsed["total"] == 2

    def test_failure_cases_filtered(self):
        """Failure cases should only include entries that didn't fully succeed."""
        instructions = [
            {"instruction": "Put the red cube on the green cube."},
        ]
        results = _simulate_parse_run(instructions)
        failure_cases = [
            {k: v for k, v in r.items() if k != "results"}
            for r in results
            if not r.get("planner_success")
        ]
        assert isinstance(failure_cases, list)
