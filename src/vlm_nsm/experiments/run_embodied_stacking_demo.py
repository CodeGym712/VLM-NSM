from __future__ import annotations

import argparse
import json

import yaml

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.predicate_monitor import compare_symbolic_physical
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.executors.robosuite_scripted_executor import RobosuiteScriptedExecutor


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def make_env(env_backend: str, cubes: list[str], render_mode: str):
    if env_backend == "mock":
        return MockEnvWrapper(cubes=cubes)
    return RobosuiteEnvWrapper(render_mode=render_mode)


def run_demo(config: dict, env_backend: str | None = None, render_mode: str | None = None):
    task_name = config["task_name"]
    n_blocks = config["n_blocks"]
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = config["target_stack"]
    planner_name = config.get("planner", "internal_stacking")
    env_backend = env_backend or config["env"]["backend"]
    render_mode = render_mode or config["env"]["render_mode"]

    plan_result = PlannerAdapter(planner_name).solve(cubes=cubes, target_stack=target_stack)
    env_wrapper = make_env(env_backend, cubes, render_mode)
    executor = RobosuiteScriptedExecutor(
        env_wrapper,
        task_name=task_name,
        thresholds=config["thresholds"],
        block_height=config["block_height"],
    )
    executor.reset(None)
    goal_predicates = build_stacking_goal(target_stack)
    executor.set_goal(goal_predicates)
    execution = executor.execute_plan(plan_result.actions if plan_result.success else [])
    physical_state = execution["physical_state"] if plan_result.success else executor.get_state()
    comparison = compare_symbolic_physical(goal_predicates, physical_state)
    goal_success = executor.check_success(goal_predicates) if plan_result.success and execution["success"] else False

    result = {
        "task_name": task_name,
        "planner": planner_name,
        "planner_success": plan_result.success,
        "actions": [{"name": action.name, "args": action.args} for action in plan_result.actions],
        "physical_execution_success": execution["success"] if plan_result.success else False,
        "physical_goal_success": goal_success,
        "failure_stage": None if goal_success else _failure_stage(plan_result, execution, goal_success),
        "mismatch_rate": comparison["mismatch_rate"],
        "missing_in_physical": comparison["missing_in_physical"],
        "extra_in_physical": comparison["extra_in_physical"],
        "env_backend": env_backend,
        "render_mode": render_mode,
    }

    out_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    out_path = out_dir / f"embodied_stacking_{n_blocks}_{env_backend}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result_path"] = str(out_path)
    return result


def _failure_stage(plan_result, execution, goal_success: bool) -> str:
    if not plan_result.success:
        return "planner"
    if not execution["success"]:
        return "execution"
    if not goal_success:
        return "goal_check"
    return "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "embodied_stacking_3.yaml"))
    parser.add_argument("--env_backend", default=None, choices=["mock", "robosuite"])
    parser.add_argument("--render_mode", default=None, choices=["none", "human", "offscreen"])
    args = parser.parse_args()

    result = run_demo(load_config(args.config), env_backend=args.env_backend, render_mode=args.render_mode)
    print(json.dumps(result, indent=2))
    print(f"result_path={result['result_path']}")


if __name__ == "__main__":
    main()
