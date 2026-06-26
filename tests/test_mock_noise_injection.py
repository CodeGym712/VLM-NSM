"""Tests for mock noise injection in EmbodiedSkillExecutor."""

import tempfile
from pathlib import Path

import numpy as np

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.embodied.controllers.scripted_controller import ScriptedController
from vlm_nsm.executors.embodied_skill_executor import EmbodiedSkillExecutor
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper


def _build_goal(target_stack):
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def test_noise_default_does_not_affect_success():
    """Default noise=0 should keep success_rate=1.0."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    assert plan_result.success

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(
        env_wrapper=env, cubes=cubes, controller=controller,
        pose_noise_std=0.0, grasp_fail_prob=0.0, place_noise_std=0.0,
    )

    result = executor.execute_plan(plan_result.actions, goal)
    assert result["episode_success"] is True
    assert result["step_success_rate"] == 1.0
    assert result["failure_stage_distribution"] == {}


def test_grasp_fail_prob_one_produces_failure():
    """grasp_fail_prob=1.0 should always produce grasp_failed."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(
        env_wrapper=env, cubes=cubes, controller=controller,
        grasp_fail_prob=1.0, seed=42,
    )

    result = executor.execute_plan(plan_result.actions, goal)
    assert result["episode_success"] is False
    # Should have grasp_failed in failure distribution
    fd = result["failure_stage_distribution"]
    assert "grasp_failed" in fd
    assert fd["grasp_failed"] >= 1


def test_place_noise_std_affects_goal():
    """Sufficient place noise should reduce success rate by shifting placement."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(
        env_wrapper=env, cubes=cubes, controller=controller,
        place_noise_std=0.5, seed=42,
    )

    result = executor.execute_plan(plan_result.actions, goal)
    # With high place noise, objects land in wrong positions → reduced match
    assert result["predicate_match_rate"] < 1.0


def test_reproducible_seed():
    """Same seed should produce same failure distribution."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    def run_with_seed(s):
        env = MockEnvWrapper(cubes=cubes)
        ctrl = ScriptedController(action_dim=4)
        ex = EmbodiedSkillExecutor(
            env_wrapper=env, cubes=cubes, controller=ctrl,
            grasp_fail_prob=0.5, seed=s,
        )
        return ex.execute_plan(plan_result.actions, goal)

    r1 = run_with_seed(42)
    r2 = run_with_seed(42)
    r3 = run_with_seed(99)

    assert r1["failure_stage_distribution"] == r2["failure_stage_distribution"]
    # Different seed should eventually differ (not guaranteed but very likely)
    # At minimum, both should be valid results
    assert "episode_success" in r3
