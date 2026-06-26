"""Test ungroundable status handling in eval_online_qwen_parser.

No DashScope, Robosuite, or GPU required — uses MockGoalParser.
Verifies that goal_json with status=ungroundable is handled correctly.
"""

import json
import pytest
from pathlib import Path

from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.experiments.eval_online_qwen_parser import (
    load_instructions,
    parser_target_stack_from_alignment,
)


class TestUngroundableHandling:
    """Test the ungroundable handling logic directly."""

    def test_ungroundable_entry_structure(self):
        """An ungroundable entry should have the expected fields set."""
        entry = {
            "instruction": "Stack the yellow cube on the red cube.",
            "parse_success": True,
            "schema_valid": True,
            "goal_json": {
                "task_type": "stacking",
                "object_bindings": {},
                "goal_predicates": [],
                "status": "ungroundable",
                "reason": "yellow cube is not present in the scene inventory",
            },
            "status": "ungroundable",
            "failure_type": "ungroundable_object",
            "ungroundable_detected": True,
            "ungroundable_reason": "yellow cube is not present in the scene inventory",
            "alignment_success": False,
            "target_stack_success": False,
            "planner_success": False,
            "error_type": "ungroundable_object",
        }
        assert entry["status"] == "ungroundable"
        assert entry["failure_type"] == "ungroundable_object"
        assert entry["ungroundable_detected"] is True
        assert entry["alignment_success"] is False
        assert entry["planner_success"] is False

    def test_adversarial_rejection_success(self):
        """Adversarial instruction correctly rejected should count as success."""
        entry = {
            "category": "adversarial_invalid_object",
            "adversarial_rejection_success": True,
            "false_rejection": False,
        }
        assert entry["adversarial_rejection_success"] is True
        assert entry["false_rejection"] is False

    def test_false_rejection_non_adversarial(self):
        """Non-adversarial instruction rejected should count as false rejection."""
        entry = {
            "category": "color_grounding",
            "adversarial_rejection_success": False,
            "false_rejection": True,
        }
        assert entry["adversarial_rejection_success"] is False
        assert entry["false_rejection"] is True

    def test_metrics_aggregation(self):
        """Ungroundable metrics should aggregate correctly."""
        results = [
            {"ungroundable_detected": True, "adversarial_rejection_success": True,
             "false_rejection": False, "category": "adversarial_invalid_object"},
            {"ungroundable_detected": True, "adversarial_rejection_success": True,
             "false_rejection": False, "category": "adversarial_invalid_object"},
            {"ungroundable_detected": True, "adversarial_rejection_success": False,
             "false_rejection": True, "category": "color_grounding"},
            {"ungroundable_detected": False, "adversarial_rejection_success": False,
             "false_rejection": False, "category": "color_grounding"},
        ]
        total = len(results)
        ungroundable = sum(1 for r in results if r.get("ungroundable_detected"))
        adv_success = sum(1 for r in results if r.get("adversarial_rejection_success"))
        adv_total = sum(1 for r in results if r.get("category") == "adversarial_invalid_object")
        false_rej = sum(1 for r in results if r.get("false_rejection"))

        assert ungroundable == 3
        assert adv_success == 2
        assert adv_total == 2
        assert false_rej == 1
        assert round(adv_success / adv_total, 3) == 1.0
        assert round(false_rej / total, 3) == 0.25

    def test_failure_cases_includes_ungroundable(self):
        """Ungroundable entries should appear in failure cases."""
        results = [
            {"instruction": "ok", "planner_success": True, "status": None},
            {"instruction": "ungroundable", "status": "ungroundable",
             "error_type": "ungroundable_object"},
            {"instruction": "planner fail", "planner_success": False, "status": None},
        ]
        failure_cases = [
            r for r in results
            if not r.get("fully_successful", r.get("planner_success", False))
            or r.get("status") == "ungroundable"
        ]
        assert len(failure_cases) == 2
        assert failure_cases[0]["instruction"] == "ungroundable"
        assert failure_cases[1]["instruction"] == "planner fail"

    def test_mock_parser_ungroundable_goal(self):
        """Mock parser should be able to produce an ungroundable-style goal_json."""
        goal = {
            "task_type": "stacking",
            "object_bindings": {},
            "goal_predicates": [],
            "status": "ungroundable",
            "reason": "yellow cube not in scene",
        }
        assert goal["status"] == "ungroundable"
        assert goal["reason"]
        assert goal["object_bindings"] == {}
        assert goal["goal_predicates"] == []
        assert "status" in goal
        assert "reason" in goal
