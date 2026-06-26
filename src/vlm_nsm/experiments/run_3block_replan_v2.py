"""3-Block Stacking with Monitor+Replan — Skill-Level Implementation.

Unlike v1, this does NOT use _run_robosuite_episode.
Instead, it directly executes PID-controlled skills with explicit
predicate checking between steps, solving the env reset dead loop.

Architecture:
  env.reset()  ← ONCE at start
  for action in plan:
    for skill in [reach, grasp, lift, reach_place, release_place]:
      execute skill with PID (NO env reset)
    check intermediate predicates
    if collapse → replan from current env state (NO env reset)

Usage:
    python -m vlm_nsm.experiments.run_3block_replan_v2 \
        --num_episodes 20 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vlm_nsm.core.data_types import Action, Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.physical_predicate_extractor import (
    physical_state_to_predicates,
)
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig

_OSC_POSE_MAX_DELTA = 0.05
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0
_REACH_THRESHOLD = 0.02
_STEP_LIMIT = 800
_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
_MAX_REPLANS = 3


def _dist(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


def _check_predicates(env, predicates):
    try:
        poses = env.get_all_object_poses()
        targets = dict(poses)
        targets["table"] = env.get_object_pose("table")
        physical = physical_state_to_predicates(poses, targets, _PREDICATE_THRESHOLDS)
        return bool(goal_satisfied(physical, predicates, mode="symbolic"))
    except Exception:
        return False


def _reach(env, pid, target_obj, phase="reach_pick", gripper=_GRIPPER_OPEN) -> tuple[bool, int]:
    """Move EEF above target object using PID."""
    obj = env.get_object_pose(target_obj)
    target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}
    pid.set_phase(phase)
    pid.set_target(pos=target)
    steps = 0
    for _ in range(_STEP_LIMIT):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            return False, steps
        pid.set_current(eef)
        action = pid.act(phase, None)
        action[6] = gripper
        try:
            env.step(action)
        except ValueError:
            return False, steps
        steps += 1
        if _dist(eef, target) < _REACH_THRESHOLD:
            return True, steps
    return False, steps


def _lift(env, pid, target_obj) -> tuple[bool, int]:
    """Lift object using PID."""
    obj = env.get_object_pose(target_obj)
    target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.15}
    pid.set_phase("lift")
    pid.set_target(pos=target)
    steps = 0
    for _ in range(_STEP_LIMIT):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            return False, steps
        pid.set_current(eef)
        action = pid.act("lift", None)
        action[6] = _GRIPPER_CLOSE
        try:
            env.step(action)
        except ValueError:
            return False, steps
        steps += 1
        if _dist(eef, target) < _REACH_THRESHOLD:
            return True, steps
    return False, steps


def _retreat(env, pid) -> tuple[bool, int]:
    """Move arm up."""
    eef = env.get_eef_pose()
    target = {"x": eef["x"], "y": eef["y"], "z": eef["z"] + 0.15}
    pid.set_phase("retreat")
    pid.set_target(pos=target)
    steps = 0
    for _ in range(_STEP_LIMIT):
        try:
            eef_cur = env.get_eef_pose()
        except ValueError:
            return False, steps
        pid.set_current(eef_cur)
        action = pid.act("retreat", None)
        action[6] = _GRIPPER_OPEN
        try:
            env.step(action)
        except ValueError:
            return False, steps
        steps += 1
        if _dist(eef_cur, target) < _REACH_THRESHOLD:
            return True, steps
    return False, steps


def _execute_skill_sequence(env, pid, pick_obj, place_obj) -> dict:
    """Execute a full pick+place sequence at the skill level.

    Corresponds to one MoveOn(pick, place) action.
    Returns {"success": bool, "steps": int, "failure": str|None}
    """
    steps = 0

    # 1. reach_pick
    ok, s = _reach(env, pid, pick_obj, "reach_pick", _GRIPPER_OPEN)
    steps += s
    if not ok:
        return {"success": False, "steps": steps, "failure": "reach_pick"}

    # 2. grasp
    cfg = GraspConfig(kp=0.05, max_delta=0.01, close_steps=70,
                      max_retries=2, holding_check_enabled=True,
                      lift_early_exit_enabled=True)
    result = execute_grasp(env, pick_obj, config=cfg, skip_approach=True, skip_lift=True)
    steps += result.get("total_step_count", 0)
    if not result["success"]:
        return {"success": False, "steps": steps, "failure": "grasp"}

    # 3. lift
    ok, s = _lift(env, pid, pick_obj)
    steps += s
    if not ok:
        return {"success": False, "steps": steps, "failure": "lift"}

    # 4. reach_place
    tgt_pose = env.get_target_pose(place_obj)
    place_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"] + 0.15}
    pid.set_phase("reach_place")
    pid.set_target(pos=place_target)
    for _ in range(2 * _STEP_LIMIT):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            return {"success": False, "steps": steps, "failure": "reach_place"}
        pid.set_current(eef)
        action = pid.act("reach_place", None)
        action[6] = _GRIPPER_CLOSE
        try:
            env.step(action)
        except ValueError:
            return {"success": False, "steps": steps, "failure": "reach_place"}
        steps += 1
        if _dist(eef, place_target) < _REACH_THRESHOLD:
            break

    # 5. release + retreat (place_state_machine)
    p_cfg = PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2, mode="simple")
    p_result = execute_place(env, pick_obj, place_obj, config=p_cfg)
    steps += p_result.get("step_count", 0)
    if not p_result.get("success", False):
        return {"success": False, "steps": steps, "failure": "place"}

    return {"success": True, "steps": steps, "failure": None}


def run_3block_with_replan(env, pid, cubes, max_replans=_MAX_REPLANS) -> dict:
    """Run 3-block stacking with Monitor+Replan at skill level."""
    target_stack = ["cube3", "cube2", "cube1"]
    planner = PlannerAdapter("bfs")

    env.reset()
    plan = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan.success:
        return {"success": False, "replans": 0, "steps": 0, "error": plan.error}

    actions = list(plan.actions)  # [move(cube2,cube3), move(cube1,cube2)]
    total_steps = 0
    replans_used = 0
    failures: dict[str, int] = {}

    while replans_used <= max_replans:
        # Check env termination — reset if horizon exhausted
        try:
            env.get_eef_pose()
        except ValueError:
            env.reset()
            replans_used += 1  # count as forced replan

        all_ok = True

        for idx, action in enumerate(actions):
            pick = action.args[0]
            place = action.args[2]
            result = _execute_skill_sequence(env, pid, pick, place)
            total_steps += result["steps"]

            if not result["success"]:
                ft = result["failure"] or "unknown"
                failures[ft] = failures.get(ft, 0) + 1
                all_ok = False

                # Check collapse: did step N's failure violate step N-1's result?
                if idx > 0 and replans_used < max_replans:
                    prev = actions[0]
                    mid_preds = [
                        Predicate("on", [prev.args[0], prev.args[2]])
                    ]
                    if not _check_predicates(env, mid_preds):
                        # Collapse detected — replan from CURRENT env state
                        replans_used += 1
                        # env is NOT reset — continue from current physical state
                        break  # restart while loop from first action
                break  # no replan possible

        if all_ok:
            # Final goal check
            goal_preds = [
                Predicate("on", [actions[0].args[0], actions[0].args[2]]),
                Predicate("on", [actions[1].args[0], actions[1].args[2]]),
            ]
            success = _check_predicates(env, goal_preds)
            return {
                "success": success,
                "replans": replans_used,
                "steps": total_steps,
                "failures": failures,
            }

        if replans_used > max_replans:
            break

    return {
        "success": False,
        "replans": replans_used,
        "steps": total_steps,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="generated/results/3block_replan_v2")
    args = parser.parse_args()

    import vlm_nsm.executors.stack_three_cube_env as _  # noqa: F401
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    cubes = ["cube1", "cube2", "cube3"]
    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda",
        cube_names=cubes, render_mode="none",
    )
    env.env.horizon = 10000

    pid = OperationalSpacePID()

    successes = 0
    all_steps = []
    all_replans = 0
    all_failures: dict[str, int] = {}

    for ep in range(args.num_episodes):
        r = run_3block_with_replan(env, pid, cubes)
        if r["success"]:
            successes += 1
        all_steps.append(r["steps"])
        all_replans += r["replans"]
        for k, v in r.get("failures", {}).items():
            all_failures[k] = all_failures.get(k, 0) + v

        if (ep + 1) % 3 == 0:
            rate = successes / (ep + 1)
            print(f"  [{ep+1}/{args.num_episodes}] success={rate:.0%} "
                  f"replans={all_replans} steps={np.mean(all_steps):.0f}", flush=True)

    summary = {
        "task": "3block_replan_v2",
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
    print(f"\n3-block Replan v2: {successes}/{args.num_episodes} "
          f"({summary['success_rate']:.0%}), "
          f"replans={all_replans}, steps={summary['mean_steps']:.0f}")
    print(f"Failures: {all_failures}")


if __name__ == "__main__":
    main()
