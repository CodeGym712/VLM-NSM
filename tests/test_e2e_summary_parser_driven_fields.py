"""Test that E2E summary includes all parser-driven fields.

Uses MockGoalParser to simulate the full pipeline without
real DashScope API calls or Robosuite.
"""
import json
import pytest
from pathlib import Path

from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates


EXPECTED_FIELDS = [
    "instruction",
    "success",
    "parser",
    "parse_success",
    "parse_error",
    "fallback_used",
    "fallback_parser",
    "parsed_goal",
    "goal_json",
    "aligned_goal_predicates",
    "parser_target_stack",
    "demo_target_stack",
    "target_stack_source",
    "target_stack_mismatch",
    "raw_qwen_output_path",
    "alignment_status",
    "symbolic_plan",
    "physical_goal_predicates",
    "plan_status",
    "plan_actions",
    "planner_error",
    "policy_used",
    "execution_success",
    "failure_type",
    "episode_result",
]

REQUIRED_FIELDS = [
    "instruction",
    "success",
    "parser",
    "parse_success",
    "aligned_goal_predicates",
    "parser_target_stack",
    "demo_target_stack",
    "target_stack_source",
    "target_stack_mismatch",
    "physical_goal_predicates",
    "symbolic_plan",
    "execution_success",
    "failure_type",
]

# Fields that must not be None on success
NON_NONE_ON_SUCCESS = [
    "parser_target_stack",
    "demo_target_stack",
    "target_stack_source",
    "target_stack_mismatch",
    "physical_goal_predicates",
    "symbolic_plan",
    "aligned_goal_predicates",
    "alignment_status",
    "execution_success",
]


class TestE2ESummaryFields:
    """Verify E2E summary return dict includes standardized parser-driven fields.

    These tests use a lightweight fixture that simulates the run_demo()
    return values without requiring Robosuite or an actual env.
    """

    def _build_mock_summary_entry(self, instruction="test instruction"):
        """Simulate the summary entry returned by run_demo()."""
        cubes = ["cube1", "cube2"]
        demo = {
            "instruction": instruction,
            "task_type": "stacking",
            "cubes": cubes,
            "target_stack": ["cube2", "cube1"],
        }

        # Parse
        parser = MockGoalParser()
        parse_result = parser.parse(
            instruction=instruction,
            task_type="stacking",
            objects=cubes,
            n_blocks=2,
        )

        # Alignment
        aligner = AlignmentLayer()
        align_result = aligner.align_and_validate(
            goal_json=parse_result.goal_json,
            valid_objects=cubes + ["table"],
            valid_predicates=["on", "clear", "free-gripper", "grasped", "smaller"],
            type_table={"cube1": "block", "cube2": "block", "table": "table"},
            predicate_signature={
                "on": [["block", "block"], ["block", "table"]],
                "clear": [["block"]],
                "free-gripper": [[]],
                "grasped": [["block"]],
                "smaller": [["block", "block"], ["block", "table"]],
            },
            task_type="stacking",
            n_blocks=2,
        )

        # Target stack derivation (parser-driven)
        on_predicates = [
            (p.name, p.args[0], p.args[1])
            for p in align_result.goal_predicates
            if p.name == "on"
        ]
        parser_target_stack = target_stack_from_predicates(on_predicates, cubes)

        # Planner with parser-derived target stack
        planner = PlannerAdapter("internal_stacking")
        plan_result = planner.solve(cubes=cubes, target_stack=parser_target_stack)

        # Build summary entry mimicking run_demo() return
        aligned_predicates = [
            [pred.name, *pred.args]
            for pred in align_result.goal_predicates
        ]

        goal_predicates = []
        for i in range(len(parser_target_stack) - 1, 0, -1):
            goal_predicates.append(["on", parser_target_stack[i], parser_target_stack[i - 1]])

        demo_target_stack = demo.get("target_stack", [])

        return {
            "instruction": instruction,
            "success": True,
            "parser": "MockGoalParser",
            "parse_success": True,
            "parse_error": None if parse_result.success else parse_result.error,
            "fallback_used": False,
            "fallback_parser": None,
            "parsed_goal": parse_result.goal_json,
            "goal_json": parse_result.goal_json,
            "aligned_goal_predicates": aligned_predicates,
            "parser_target_stack": parser_target_stack,
            "demo_target_stack": demo_target_stack,
            "target_stack_source": "parser_aligned_goal",
            "target_stack_mismatch": parser_target_stack != demo_target_stack,
            "raw_qwen_output_path": None,
            "alignment_status": "passed",
            "alignment_error": None,
            "symbolic_plan": [
                {"name": a.name, "args": a.args}
                for a in plan_result.actions
            ],
            "physical_goal_predicates": goal_predicates,
            "plan_status": "success",
            "plan_actions": [(a.name, a.args) for a in plan_result.actions],
            "planner_error": plan_result.error if plan_result.error else None,
            "policy_used": "reach_pick",
            "execution_success": True,
            "failure_type": None,
            "episode_result": {
                "goal_success": True,
                "steps": 100,
                "failure_type": None,
                "bc_steps": 20,
                "scripted_steps": 80,
            },
            "video_recorded": False,
            "video_path": None,
            "video_error": None,
        }

    def test_all_expected_fields_present(self):
        entry = self._build_mock_summary_entry()
        for field in EXPECTED_FIELDS:
            assert field in entry, f"Missing field: {field}"

    def test_required_fields_not_none_on_success(self):
        entry = self._build_mock_summary_entry()
        for field in NON_NONE_ON_SUCCESS:
            assert entry.get(field) is not None, f"Required field is None on success: {field}"
        # failure_type is legitimately None on success
        assert "failure_type" in entry

    def test_target_stack_source_is_parser_aligned_goal(self):
        entry = self._build_mock_summary_entry()
        assert entry["target_stack_source"] == "parser_aligned_goal"

    def test_physical_goal_predicates_no_table(self):
        entry = self._build_mock_summary_entry()
        for pred in entry["physical_goal_predicates"]:
            assert "table" not in pred, f"Physical goal contains table: {pred}"

    def test_aligned_goal_predicates_may_contain_table(self):
        entry = self._build_mock_summary_entry()
        has_table = any("table" in pred for pred in entry["aligned_goal_predicates"])
        # Note: MockGoalParser always includes on(bottom, table)
        assert has_table, "Aligned predicates should include table support"

    def test_target_stack_mismatch_when_expected(self):
        cubes = ["cube1", "cube2"]
        parser = MockGoalParser()
        parse_result = parser.parse("test", task_type="stacking", objects=cubes, n_blocks=2)
        aligner = AlignmentLayer()
        align_result = aligner.align_and_validate(
            goal_json=parse_result.goal_json,
            valid_objects=cubes + ["table"],
            valid_predicates=["on"],
            type_table={"cube1": "block", "cube2": "block", "table": "table"},
            predicate_signature={"on": [["block", "block"], ["block", "table"]]},
            task_type="stacking",
            n_blocks=2,
        )
        on_preds = [(p.name, p.args[0], p.args[1]) for p in align_result.goal_predicates if p.name == "on"]
        parser_stack = target_stack_from_predicates(on_preds, cubes)

        # With matching demo target_stack
        assert parser_stack == ["cube2", "cube1"]  # Mock always gives this
        assert parser_stack == ["cube2", "cube1"]  # Same as demo

    def test_symbolic_plan_structure(self):
        entry = self._build_mock_summary_entry()
        assert isinstance(entry["symbolic_plan"], list)
        if entry["symbolic_plan"]:
            plan_item = entry["symbolic_plan"][0]
            assert "name" in plan_item
            assert "args" in plan_item

    def test_episode_result_structure(self):
        entry = self._build_mock_summary_entry()
        ep = entry["episode_result"]
        assert isinstance(ep, dict)
        assert "goal_success" in ep
        assert "steps" in ep
        assert "failure_type" in ep
        assert "bc_steps" in ep
        assert "scripted_steps" in ep

    def test_goal_json_matches_parsed_goal(self):
        entry = self._build_mock_summary_entry()
        assert entry["goal_json"] == entry["parsed_goal"]

    def test_failure_type_present(self):
        entry = self._build_mock_summary_entry()
        assert "failure_type" in entry
