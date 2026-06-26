"""3-Block Stacking with Monitor+Replan for collapse recovery.

Uses the original run_embodied_stacking_robosuite runner for each
individual stacking step, with intermediate predicate checks and
replanning when the intermediate stack collapses.

Based on P1-5 experiment: 14/15 (93%) recovery rate.

Usage:
    python -m vlm_nsm.experiments.run_3block_replan \
        --num_episodes 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlm_nsm.core.data_types import Action, Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.physical_predicate_extractor import (
    physical_state_to_predicates,
)
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
    _run_robosuite_episode,
    build_stacking_goal,
)

_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
_MAX_REPLANS = 3


def _check_predicates(env, predicates: list[Predicate]) -> bool:
    """Check if predicates hold in current env state."""
    try:
        object_poses = env.get_all_object_poses()
        targets = dict(object_poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(
            object_poses, targets, _PREDICATE_THRESHOLDS)
        return bool(goal_satisfied(physical, predicates, mode="symbolic"))
    except Exception:
        return False


def run_3block_episode(env, cubes, max_replans=_MAX_REPLANS, **episode_kwargs) -> dict:
    """Run one 3-block episode with Monitor+Replan.

    Strategy:
      Step 1: cube1 on cube2
      Step 2: cube3 on cube1
      If Step 2 causes collapse → replan: restack cube1, retry cube3
    """
    target_stack = ["cube3", "cube2", "cube1"]
    goal_predicates = build_stacking_goal(target_stack)
    planner = PlannerAdapter("bfs")

    total_steps = 0
    replans_used = 0
    success = False
    failures: dict[str, int] = {}
    steps_per_action: list[int] = []

    plan = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan.success:
        return {"success": False, "error": plan.error, "replans": 0}

    actions = list(plan.actions)  # [MoveOn(cube1,cube2), MoveOn(cube2,cube3)]

    while replans_used <= max_replans:
        step_ok = True
        steps_per_action = []
        env.reset()  # fresh start each replan attempt

        for idx, action in enumerate(actions):
            result = _run_robosuite_episode(
                env, [action], cubes, goal_predicates,
                skip_reset=(idx > 0),  # only skip reset between steps
                early_stop_on_goal=False,
                **episode_kwargs,
            )
            steps_per_action.append(result.get("num_steps", 0))
            total_steps += result.get("num_steps", 0)

            if not result.get("goal_success", result.get("episode_success", False)):
                ft = result.get("failure_type", "step_failed")
                failures[ft] = failures.get(ft, 0) + 1
                step_ok = False

                # Step 2 failure → check if step 1's result collapsed
                if idx > 0 and replans_used < max_replans:
                    first_action = actions[0]
                    mid_preds = build_stacking_goal(
                        [first_action.args[2], first_action.args[0]])
                    if not _check_predicates(env, mid_preds):
                        replans_used += 1
                        break  # restart while loop (env.reset at top)
                break  # no replan possible → fail

        if step_ok:
            success = _check_predicates(env, build_stacking_goal(target_stack))
            break  # done

    # prevent infinite loop
    if replans_used > max_replans:
        success = False

    return {
        "success": success,
        "total_steps": total_steps,
        "replans_used": replans_used,
        "failure_stage_distribution": failures,
        "steps_per_action": steps_per_action,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="generated/results/3block_replan")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Monkey-patch Stack for 3 cubes
    import vlm_nsm.executors.stack_three_cube_env as _  # noqa: F401

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    cubes = ["cube1", "cube2", "cube3"]

    render_mode = "human" if args.render else "none"
    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda",
        cube_names=cubes, render_mode=render_mode,
    )
    env.env.horizon = 10000  # FIX: 3-block with replan needs many steps

    episode_kwargs = dict(
        horizon=1500, use_place_state_machine=True,
        horizon_aware=True, place_retry=True,
        holding_check_enabled=True, lift_early_exit_enabled=True,
        use_pid_controller=True,  # PID is default
    )

    successes = 0
    all_steps = []
    all_replans = 0
    all_failures: dict[str, int] = {}

    for ep in range(args.num_episodes):
        result = run_3block_episode(env, cubes, **episode_kwargs)
        if result["success"]:
            successes += 1
        all_steps.append(result["total_steps"])
        all_replans += result["replans_used"]
        for k, v in result.get("failure_stage_distribution", {}).items():
            all_failures[k] = all_failures.get(k, 0) + v

        if (ep + 1) % 5 == 0:
            rate = successes / (ep + 1)
            print(f"  [{ep+1}/{args.num_episodes}] success={rate:.0%} "
                  f"replans={all_replans}", flush=True)

    summary = {
        "task": "3block_stacking_with_replan",
        "num_episodes": args.num_episodes,
        "success_rate": successes / args.num_episodes,
        "mean_steps": float(np.mean(all_steps)),
        "total_replans": all_replans,
        "failure_stage_distribution": all_failures,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n3-block+Replan: {successes}/{args.num_episodes} "
          f"({summary['success_rate']:.0%}), "
          f"replans={all_replans}, steps={summary['mean_steps']:.0f}")
    print(f"Failures: {all_failures}")
    print(f"Saved to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
