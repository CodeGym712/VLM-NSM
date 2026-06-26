from vlm_nsm.experiments.run_embodied_stacking_demo import build_stacking_goal, load_config, run_demo
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper
from vlm_nsm.executors.robosuite_scripted_executor import RobosuiteScriptedExecutor
from vlm_nsm.planners.internal_stacking_planner import InternalStackingPlanner
from vlm_nsm.core.utils import PROJECT_ROOT


def test_mock_env_wrapper_support_queries_update_after_move():
    env = MockEnvWrapper()
    assert env.simulate_pick_and_place("cube2", "cube3")["success"]
    assert env.get_objects_on("cube3") == ["cube2"]


def test_mock_scripted_executor_completes_stacking_goal():
    env = MockEnvWrapper()
    executor = RobosuiteScriptedExecutor(
        env,
        task_name="stacking",
        thresholds={"xy_threshold": 0.03, "z_min": 0.02, "z_max": 0.06},
        block_height=0.04,
    )
    planner_result = InternalStackingPlanner().solve(
        cubes=["cube1", "cube2", "cube3"],
        target_stack=["cube3", "cube2", "cube1"],
    )
    goal_predicates = build_stacking_goal(["cube3", "cube2", "cube1"])
    executor.reset(None)
    executor.set_goal(goal_predicates)
    result = executor.execute_plan(planner_result.actions)
    assert result["success"], result["error"]
    assert executor.check_success(goal_predicates)


def test_mock_embodied_demo_runs_from_config():
    config = load_config(PROJECT_ROOT / "configs" / "embodied_stacking_3.yaml")
    result = run_demo(config, env_backend="mock", render_mode="none")
    assert result["planner_success"]
    assert result["physical_execution_success"]
    assert result["physical_goal_success"]
