"""Hanoi Tower evaluation with PID controller.

Uses InternalHanoiPlanner + StackThreeCube environment.
Each Hanoi move = pick disk + place disk, using reach/grasp/lift + reach/place/release.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from vlm_nsm.core.planner_adapter import PlannerAdapter

_OSC_POSE_MAX_DELTA = 0.05
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0
_REACH_THRESHOLD = 0.01  # tighter for cross-table Hanoi moves
_STEP_LIMIT = 500


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


def _norm_action(dx, dy, dz, gripper):
    return np.array([np.clip(dx/_OSC_POSE_MAX_DELTA,-1,1),
                     np.clip(dy/_OSC_POSE_MAX_DELTA,-1,1),
                     np.clip(dz/_OSC_POSE_MAX_DELTA,-1,1),
                     0,0,0,gripper], dtype=np.float32)


def _safe_step(env, action):
    """env.step with termination guard."""
    try:
        env.step(action)
        return True
    except ValueError:
        return False


# Peg positions (from stack_three_cube_env.py)
_PEG_MAP = {"peg1": (-0.12, -0.08), "peg2": (0.0, -0.08), "peg3": (0.12, -0.08)}


def _do_reach(env, pid_ctrl, disk, total_steps):
    """Move EEF above the object."""
    obj = env.get_object_pose(disk)
    target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}
    pid_ctrl.set_phase("reach_pick")
    pid_ctrl.set_target(pos=target)
    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        pid_ctrl.set_current(eef)
        a = pid_ctrl.act("reach_pick", None)
        a[6] = _GRIPPER_OPEN
        if not _safe_step(env, a):
            return False, total_steps
        total_steps += 1
        if _dist(eef, target) < _REACH_THRESHOLD:
            break
    return True, total_steps


def _do_grasp(env, pid_ctrl, disk, total_steps):
    """Descend and close gripper on object."""
    obj = env.get_object_pose(disk)
    mid_z = obj["z"] - 0.010
    pid_ctrl.set_phase("grasp")
    pid_ctrl.set_target(pos={"x": obj["x"], "y": obj["y"], "z": mid_z})
    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        pid_ctrl.set_current(eef)
        a = pid_ctrl.act("grasp", None)
        a[6] = _GRIPPER_OPEN
        if not _safe_step(env, a):
            return False, total_steps
        total_steps += 1
        if eef["z"] <= obj["z"] + 0.005:
            break
    # Close
    for _ in range(100):
        a = _norm_action(0, 0, 0, _GRIPPER_CLOSE)
        if not _safe_step(env, a):
            break
        total_steps += 1
    gs = env.get_gripper_state()
    grasped = (gs.get("open", -1) > 0.003 and gs.get("open", -1) < 0.035)
    return grasped, total_steps


def _do_lift(env, pid_ctrl, disk, total_steps):
    """Lift grasped object."""
    obj = env.get_object_pose(disk)
    lt = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.15}
    pid_ctrl.set_phase("lift")
    pid_ctrl.set_target(pos=lt)
    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        pid_ctrl.set_current(eef)
        a = pid_ctrl.act("lift", None)
        a[6] = _GRIPPER_CLOSE
        if not _safe_step(env, a):
            return total_steps
        total_steps += 1
        if _dist(eef, lt) < _REACH_THRESHOLD:
            break
    return total_steps


def _do_place(env, pid_ctrl, to_peg, total_steps):
    """Move to peg and release."""
    px, py = _PEG_MAP.get(to_peg, (0.0, -0.08))
    place_target = {"x": px, "y": py, "z": 0.95}
    pid_ctrl.set_phase("reach_place")
    pid_ctrl.set_target(pos=place_target)
    for _ in range(2 * _STEP_LIMIT):
        eef = env.get_eef_pose()
        pid_ctrl.set_current(eef)
        a = pid_ctrl.act("reach_place", None)
        a[6] = _GRIPPER_CLOSE
        if not _safe_step(env, a):
            return False, total_steps
        total_steps += 1
        if _dist(eef, place_target) < _REACH_THRESHOLD:
            break
    # Release + retreat
    for _ in range(30):
        if not _safe_step(env, _norm_action(0, 0, 0, _GRIPPER_OPEN)):
            return False, total_steps
        total_steps += 1
    for _ in range(30):
        if not _safe_step(env, _norm_action(0, 0, 0.1, _GRIPPER_OPEN)):
            return False, total_steps
        total_steps += 1
    return True, total_steps


def run_hanoi_episode(env, pid_ctrl, n_disks=3) -> dict:
    """Run one Hanoi episode using pick/place action format."""
    from vlm_nsm.planners.internal_hanoi_planner import InternalHanoiPlanner

    planner = InternalHanoiPlanner()
    plan = planner.solve(n_disks)
    if not plan.success:
        return {"success": False, "error": plan.error}

    env.reset()
    env.env.horizon = max(5000, (2**n_disks - 1) * 2 * 500)
    # Place cubes at peg1 position for consistent Hanoi initial state
    _init_x, _init_y = -0.12, -0.08  # peg1
    _cube_joints = {3: "cubeC_joint0", 2: "cubeB_joint0", 1: "cubeA_joint0"}
    for i in range(n_disks, 0, -1):
        jname = _cube_joints.get(i, f"cube{i}_joint0")
        z = 0.82 + (n_disks - i) * 0.04
        env.env.sim.data.set_joint_qpos(jname, np.array([_init_x, _init_y, z, 1,0,0,0]))
    env.env.sim.forward()  # sync physics
    total_steps = 0
    failures: dict[str, int] = {}

    for action in plan.actions:
        name = action.name  # "pick" or "place"
        disk = action.args[0]

        if name == "pick":
            ok, total_steps = _do_reach(env, pid_ctrl, disk, total_steps)
            if not ok:
                return {"success": False, "num_steps": total_steps,
                        "failure_stage_distribution": {"horizon": 1}}
            grasped, total_steps = _do_grasp(env, pid_ctrl, disk, total_steps)
            if not grasped:
                failures["grasp"] = failures.get("grasp", 0) + 1
                return {"success": False, "num_steps": total_steps,
                        "failure_stage_distribution": failures}
            total_steps = _do_lift(env, pid_ctrl, disk, total_steps)

        elif name == "place":
            to_peg = action.args[1]  # "peg1", "peg2", or "peg3"
            ok, total_steps = _do_place(env, pid_ctrl, to_peg, total_steps)
            if not ok:
                return {"success": False, "num_steps": total_steps,
                        "failure_stage_distribution": {"horizon": 1}}

    return {"success": True, "num_steps": total_steps,
            "failure_stage_distribution": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_disks", type=int, default=3)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dir", default="generated/results/hanoi_pid")
    args = parser.parse_args()

    # Patch Stack for 3-cube + pegs
    import vlm_nsm.executors.stack_three_cube_env as _  # noqa: F401

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID

    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=[f"cube{i}" for i in range(1, args.n_disks + 1)],
        render_mode="none",
    )
    # Hanoi needs more steps: 2^n - 1 moves, each ~200-300 steps
    env.env.horizon = max(5000, (2**args.n_disks - 1) * 2 * 500)  # ~24,500 for 3-disk

    pid = OperationalSpacePID()

    successes = 0
    all_steps = []
    all_failures: dict[str, int] = {}

    for ep in range(args.num_episodes):
        result = run_hanoi_episode(env, pid, args.n_disks)
        if result["success"]:
            successes += 1
        all_steps.append(result["num_steps"])
        for k, v in result.get("failure_stage_distribution", {}).items():
            all_failures[k] = all_failures.get(k, 0) + v
        pid.reset()

    summary = {
        "task": f"hanoi_{args.n_disks}disk",
        "num_episodes": args.num_episodes,
        "episode_success_rate": successes / args.num_episodes,
        "mean_steps": float(np.mean(all_steps)),
        "failure_stage_distribution": all_failures,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Hanoi {args.n_disks}-disk: {successes}/{args.num_episodes} "
          f"({summary['episode_success_rate']:.0%}), "
          f"steps={summary['mean_steps']:.0f}, "
          f"failures={all_failures}")
    print(f"Saved to {out / 'summary.json'}")


if __name__ == "__main__":
    main()
