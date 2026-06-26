"""Tests for EmbodiedSkillExecutor with MockEnvWrapper."""

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.embodied.controllers.scripted_controller import ScriptedController
from vlm_nsm.executors.embodied_skill_executor import EmbodiedSkillExecutor
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper


def _build_goal(target_stack):
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def test_executor_executes_stacking_3():
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    assert plan_result.success

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(env_wrapper=env, cubes=cubes, controller=controller)

    result = executor.execute_plan(plan_result.actions, goal)

    assert result["goal_success"] is True
    assert result["execution_success"] is True
    assert result["step_success_rate"] == 1.0


def test_executor_step_success_rate():
    """All steps should succeed in mock mode with scripted controller."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(env_wrapper=env, cubes=cubes, controller=controller)

    result = executor.execute_plan(plan_result.actions, goal)

    assert result["step_success_rate"] == 1.0
    assert result["skill_success_rate"] == 1.0
    # Stacking 3 has 2 moves, each with 6 steps = 12 total
    assert result["num_steps"] == 12


def test_executor_predicate_match():
    """Predicates from physical state should match goal."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(env_wrapper=env, cubes=cubes, controller=controller)

    result = executor.execute_plan(plan_result.actions, goal)

    assert result["predicate_match_rate"] >= 0.9


def test_executor_episode_success():
    """Episode success should be True when all steps and goal succeed."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=4)
    executor = EmbodiedSkillExecutor(env_wrapper=env, cubes=cubes, controller=controller)

    result = executor.execute_plan(plan_result.actions, goal)

    assert result["episode_success"] is True


def test_executor_no_controller():
    """Executor should work even without a controller (zero actions)."""
    cubes = ["cube1", "cube2", "cube3"]
    target_stack = ["cube3", "cube2", "cube1"]
    goal = _build_goal(target_stack)
    planner = PlannerAdapter("internal_stacking")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    executor = EmbodiedSkillExecutor(env_wrapper=env, cubes=cubes, controller=None)

    result = executor.execute_plan(plan_result.actions, goal)

    assert result["goal_success"] is True
