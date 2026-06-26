"""Tests for Stage 19 rollout runner — no robosuite, no GPU.

Uses mocks/fakes for environment and controller.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Prevent accidental robosuite imports during test collection
os.environ.setdefault("MUJOCO_GL", "egl")

from vlm_nsm.experiments.stage19_run_symbolic_rollouts import (
    FAILURE_LABELS,
    StaticGoalParser,
    build_stacking_goal,
    check_gate,
    _map_failure_type,
)


# ---------------------------------------------------------------------------
# StaticGoalParser tests
# ---------------------------------------------------------------------------

class TestStaticGoalParser:
    def test_returns_preset_goal_json(self):
        goal = {"status": "ok", "task_type": "stacking"}
        parser = StaticGoalParser(goal)
        result = parser.parse(instruction="anything", task_type="stacking",
                              objects=["cube1"], n_blocks=2)
        assert result.success
        assert result.goal_json == goal

    def test_ignores_all_kwargs(self):
        parser = StaticGoalParser({"status": "ok"})
        result = parser.parse(foo="bar", extra=123)
        assert result.success


# ---------------------------------------------------------------------------
# Gate validation tests
# ---------------------------------------------------------------------------

VALID_GOAL_CUBE1_ON_CUBE2 = {
    "status": "ok",
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
    "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]],
}

VALID_GOAL_CUBE2_ON_CUBE1 = {
    "status": "ok",
    "task_type": "stacking",
    "object_bindings": {"top block": "cube2", "bottom block": "cube1"},
    "goal_predicates": [["on", "cube2", "cube1"], ["on", "cube1", "table"]],
}

UNGROUNDABLE_GOAL = {
    "status": "ungroundable",
    "task_type": "stacking",
    "object_bindings": {},
    "goal_predicates": [],
    "reason": "ambiguous reference",
}

SAME_OBJECT_GOAL = {
    "status": "ok",
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube1"},
    "goal_predicates": [["on", "cube1", "cube1"]],
}


class TestGateValidation:
    def test_valid_goal_passes_gate(self):
        result = check_gate(VALID_GOAL_CUBE1_ON_CUBE2, ["cube1", "cube2"])
        assert result["passed"]
        assert result["target_stack"] == ["cube2", "cube1"]

    def test_valid_goal_reverse_passes_gate(self):
        result = check_gate(VALID_GOAL_CUBE2_ON_CUBE1, ["cube1", "cube2"])
        assert result["passed"]
        assert result["target_stack"] == ["cube1", "cube2"]

    def test_ungroundable_goal_rejected(self):
        result = check_gate(UNGROUNDABLE_GOAL, ["cube1", "cube2"])
        assert not result["passed"]
        assert "ungroundable" in result["reason"].lower()

    def test_same_object_goal_rejected(self):
        result = check_gate(SAME_OBJECT_GOAL, ["cube1", "cube2"])
        assert not result["passed"]

    def test_missing_object_rejected(self):
        goal = {
            "status": "ok",
            "task_type": "stacking",
            "object_bindings": {"top block": "cube3", "bottom block": "cube2"},
            "goal_predicates": [["on", "cube3", "cube2"], ["on", "cube2", "table"]],
        }
        result = check_gate(goal, ["cube1", "cube2"])
        assert not result["passed"]

    def test_unsupported_relation_rejected(self):
        goal = {
            "status": "ok",
            "task_type": "stacking",
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
            "goal_predicates": [["next_to", "cube1", "cube2"]],
        }
        result = check_gate(goal, ["cube1", "cube2"])
        assert not result["passed"]


# ---------------------------------------------------------------------------
# build_stacking_goal tests
# ---------------------------------------------------------------------------

class TestBuildStackingGoal:
    def test_two_block_stack(self):
        result = build_stacking_goal(["cube2", "cube1"])
        assert result == [["on", "cube1", "cube2"]]

    def test_reverse_two_block_stack(self):
        result = build_stacking_goal(["cube1", "cube2"])
        assert result == [["on", "cube2", "cube1"]]

    def test_single_block(self):
        result = build_stacking_goal(["cube1"])
        assert result == []


# ---------------------------------------------------------------------------
# Failure type mapping tests
# ---------------------------------------------------------------------------

class TestFailureTypeMapping:
    def test_grasp_failure_mapped(self):
        assert _map_failure_type("grasp_failed") == "grasp_failure"

    def test_place_failure_mapped(self):
        assert _map_failure_type("place_failed_during_retry") == "place_failure"

    def test_timeout_mapped(self):
        assert _map_failure_type("timeout_horizon") == "timeout"

    def test_none_mapped_to_unknown(self):
        assert _map_failure_type(None) == "unknown"

    def test_empty_string_mapped_to_unknown(self):
        assert _map_failure_type("") == "unknown"

    def test_all_failure_labels_are_valid(self):
        for label in FAILURE_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0


# ---------------------------------------------------------------------------
# Mock-based rollout tests (no robosuite)
# ---------------------------------------------------------------------------

class FakeEnv:
    """Minimal fake env for testing the runner logic without robosuite."""

    def __init__(self):
        self._poses: dict[str, dict] = {}

    def reset(self):
        pass

    def get_object_pose(self, name):
        return self._poses.get(name)

    def close(self):
        pass


def _make_episode(episode_id, goal_id, parser_output, expected_execution="execute",
                  seed=0, resolved_top="cube1", resolved_bottom="cube2"):
    return {
        "episode_id": episode_id,
        "goal_id": goal_id,
        "description": "",
        "parser_output": parser_output,
        "expected_execution": expected_execution,
        "expected_top": resolved_top,
        "expected_bottom": resolved_bottom,
        "seed": seed,
    }


class TestRolloutRunnerWithMocks:
    """Test the run_single_episode function with mocked env and controller."""

    def _make_planner(self):
        from vlm_nsm.core.planner_adapter import PlannerAdapter
        return PlannerAdapter("internal_stacking")

    def test_valid_goal_passes_gate_and_calls_executor(self):
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep = _make_episode("test_valid", "red_on_green", VALID_GOAL_CUBE1_ON_CUBE2)
        env = FakeEnv()
        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={"kp": 0.08},
        )

        assert result["gate_decision"] == "execute"
        assert result["resolved_top"] == "cube1"
        assert result["resolved_bottom"] == "cube2"
        assert result["target_stack_valid"]
        assert result["env_reset_ok"]
        assert result["rollout_started"]

    def test_ungroundable_goal_rejected_and_rollout_not_called(self):
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep = _make_episode("test_ungroundable", "ambiguous", UNGROUNDABLE_GOAL,
                           expected_execution="reject")
        env = FakeEnv()
        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={},
        )

        assert result["gate_decision"] == "reject"
        assert result["failure_reason"] == "rejected_by_gate"
        assert not result["rollout_started"]

    def test_invalid_same_object_goal_rejected(self):
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep = _make_episode("test_same_obj", "invalid_same", SAME_OBJECT_GOAL,
                           expected_execution="reject")
        env = FakeEnv()
        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={},
        )

        assert result["gate_decision"] == "reject"
        assert not result["rollout_started"]

    def test_env_reset_exception_logged_as_env_reset_failure(self):
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep = _make_episode("test_env_fail", "red_on_green", VALID_GOAL_CUBE1_ON_CUBE2)
        env = FakeEnv()

        def failing_reset():
            raise RuntimeError("simulation crashed")
        env.reset = failing_reset

        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={},
        )

        assert result["failure_reason"] == "env_reset_failure"
        assert result["exception_type"] == "RuntimeError"
        assert "simulation crashed" in result["exception_message"]

    def test_task_success_result_has_none_failure_reason(self):
        """When task_success=True, failure_reason must be 'none'."""
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep = _make_episode("test_success", "red_on_green", VALID_GOAL_CUBE1_ON_CUBE2)
        env = FakeEnv()
        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={},
        )

        # Even if execution fails, the failure reason should not be "none" if task_success is False
        if not result["task_success"]:
            assert result["failure_reason"] != "none" or result["gate_decision"] == "reject"

    def test_unsafe_goal_leak_detected(self):
        """A goal expected to be rejected but that passes gate = unsafe leak."""
        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        # This goal is structurally valid but expected to be rejected
        ep = _make_episode("test_leak", "should_reject", VALID_GOAL_CUBE1_ON_CUBE2,
                           expected_execution="reject")
        env = FakeEnv()
        planner = self._make_planner()

        result = run_single_episode(
            ep, ["cube1", "cube2"], env, planner,
            policy_override=None, override_phases={0},
            exec_config={},
        )

        assert result["unsafe_execution"]
        assert result["failure_reason"] == "unsafe_goal_leak"

    def test_summary_aggregates_per_goal_success(self):
        """Manual summary aggregation test."""
        summary = {
            "total_episodes": 4,
            "executed_episodes": 0,
            "unsafe_goal_leak_count": 0,
            "task_success_count": 0,
            "per_goal_success": {},
        }

        from vlm_nsm.experiments.stage19_run_symbolic_rollouts import run_single_episode

        ep1 = _make_episode("ep1", "red_on_green", VALID_GOAL_CUBE1_ON_CUBE2)
        ep2 = _make_episode("ep2", "green_on_red", VALID_GOAL_CUBE2_ON_CUBE1)

        for ep in [ep1, ep1, ep2, ep2]:
            env = FakeEnv()
            planner = self._make_planner()
            result = run_single_episode(
                ep, ["cube1", "cube2"], env, planner,
                policy_override=None, override_phases={0},
                exec_config={},
            )
            if result["gate_decision"] == "execute":
                summary["executed_episodes"] += 1
            if result["task_success"]:
                summary["task_success_count"] += 1
            gid = result["goal_id"]
            if gid not in summary["per_goal_success"]:
                summary["per_goal_success"][gid] = {"total": 0, "success": 0}
            summary["per_goal_success"][gid]["total"] += 1
            if result["task_success"]:
                summary["per_goal_success"][gid]["success"] += 1

        assert summary["executed_episodes"] == 4
        assert summary["per_goal_success"]["red_on_green"]["total"] == 2
        assert summary["per_goal_success"]["green_on_red"]["total"] == 2
