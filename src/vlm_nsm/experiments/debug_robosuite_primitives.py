"""Debug script for individual robosuite primitives (reach, grasp, place).

Uses robosuite 1.5.2 normalized action space [-1, 1].

Usage:
    python -m vlm_nsm.experiments.debug_robosuite_primitives \\
        --primitive reach --object cube1

    python -m vlm_nsm.experiments.debug_robosuite_primitives \\
        --primitive grasp --object cube1

    python -m vlm_nsm.experiments.debug_robosuite_primitives \\
        --primitive place --object cube1 --target cube2
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir

_STEP_LIMIT = 500
_GRIPPER_CLOSE = 1.0
_GRIPPER_OPEN = -1.0
_REACH_THRESHOLD = 0.02
_LIFT_HEIGHT = 0.05

# OSC_POSE max position delta per step (for action normalization)
_OSC_POSE_MAX_DELTA = 0.05


def _dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _norm_action(dx, dy, dz, gripper):
    """Convert physical deltas to normalized actions."""
    return np.array([
        np.clip(dx / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dy / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dz / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        0.0, 0.0, 0.0,
        gripper,
    ], dtype=np.float32)


def run_reach(object_name: str) -> dict:
    """Move EEF to above the target object."""
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    except ImportError as e:
        return {"error": str(e)}

    env = RobosuiteEnvWrapper(render_mode="human" if "--render" in sys.argv else "none")
    env.reset()
    obj_pose = env.get_object_pose(object_name)
    target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + 0.10}
    initial_eef = env.get_eef_pose()

    kp = 0.05
    max_delta = 0.02

    for step in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
        dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
        dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)

        action = _norm_action(dx, dy, dz, _GRIPPER_OPEN)
        env.step(action)

        if _dist(eef, target) < _REACH_THRESHOLD:
            final_eef = env.get_eef_pose()
            return {
                "primitive": "reach",
                "object": object_name,
                "success": True,
                "steps": step + 1,
                "initial_eef": initial_eef,
                "target": target,
                "final_eef": final_eef,
                "final_distance": _dist(final_eef, target),
            }

    final_eef = env.get_eef_pose()
    return {
        "primitive": "reach",
        "object": object_name,
        "success": False,
        "steps": _STEP_LIMIT,
        "initial_eef": initial_eef,
        "target": target,
        "final_eef": final_eef,
        "final_distance": _dist(final_eef, target),
        "failure_type": "timeout",
    }


def run_grasp(object_name: str) -> dict:
    """Closed-loop grasp with retry and detailed diagnostics."""
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    except ImportError as e:
        return {"error": str(e)}

    env = RobosuiteEnvWrapper(render_mode="human" if "--render" in sys.argv else "none")
    env.reset()

    from vlm_nsm.embodied.grasp_state_machine import execute_grasp

    result = execute_grasp(env, object_name)
    result["primitive"] = "grasp"
    result["object"] = object_name
    return result


def run_place(object_name: str, target_name: str) -> dict:
    """Pick object_name and place onto target_name."""
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    except ImportError as e:
        return {"error": str(e)}

    env = RobosuiteEnvWrapper(render_mode="human" if "--render" in sys.argv else "none")
    env.reset()

    kp = 0.05
    max_delta = 0.02

    # Phase 1: Reach and grasp the object
    obj_pose = env.get_object_pose(object_name)
    reach_target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + 0.08}

    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        dx = np.clip(kp * (reach_target["x"] - eef["x"]), -max_delta, max_delta)
        dy = np.clip(kp * (reach_target["y"] - eef["y"]), -max_delta, max_delta)
        dz = np.clip(kp * (reach_target["z"] - eef["z"]), -max_delta, max_delta)
        action = _norm_action(dx, dy, dz, _GRIPPER_OPEN)
        env.step(action)
        if _dist(eef, reach_target) < _REACH_THRESHOLD:
            break

    obj_z_before = env.get_object_pose(object_name)["z"]

    for _ in range(50):
        action = _norm_action(0.0, 0.0, 0.0, _GRIPPER_CLOSE)
        env.step(action)

    obj_z_after_grasp = env.get_object_pose(object_name)["z"]
    grasp_ok = (obj_z_after_grasp - obj_z_before) > 0.005

    # Phase 2: Lift
    lift_target = {"x": reach_target["x"], "y": reach_target["y"], "z": reach_target["z"] + _LIFT_HEIGHT}
    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        dx = np.clip(kp * (lift_target["x"] - eef["x"]), -max_delta, max_delta)
        dy = np.clip(kp * (lift_target["y"] - eef["y"]), -max_delta, max_delta)
        dz = np.clip(kp * (lift_target["z"] - eef["z"]), -max_delta, max_delta)
        action = _norm_action(dx, dy, dz, _GRIPPER_CLOSE)
        env.step(action)
        if _dist(eef, lift_target) < _REACH_THRESHOLD:
            break

    # Phase 3: Move to target
    tgt_pose = env.get_target_pose(target_name)
    place_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"] + 0.10}

    for _ in range(2 * _STEP_LIMIT):
        eef = env.get_eef_pose()
        dx = np.clip(kp * (place_target["x"] - eef["x"]), -max_delta, max_delta)
        dy = np.clip(kp * (place_target["y"] - eef["y"]), -max_delta, max_delta)
        dz = np.clip(kp * (place_target["z"] - eef["z"]), -max_delta, max_delta)
        action = _norm_action(dx, dy, dz, _GRIPPER_CLOSE)
        env.step(action)
        if _dist(eef, place_target) < _REACH_THRESHOLD:
            break

    # Phase 4: Release
    for _ in range(50):
        action = _norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN)
        env.step(action)

    # Phase 5: Retreat
    retreat_target = {"x": place_target["x"], "y": place_target["y"], "z": place_target["z"] + 0.15}
    for _ in range(_STEP_LIMIT):
        eef = env.get_eef_pose()
        dx = np.clip(kp * (retreat_target["x"] - eef["x"]), -max_delta, max_delta)
        dy = np.clip(kp * (retreat_target["y"] - eef["y"]), -max_delta, max_delta)
        dz = np.clip(kp * (retreat_target["z"] - eef["z"]), -max_delta, max_delta)
        action = _norm_action(dx, dy, dz, _GRIPPER_OPEN)
        env.step(action)
        if _dist(eef, retreat_target) < _REACH_THRESHOLD:
            break

    # Check final state
    final_obj_pose = env.get_object_pose(object_name)
    final_tgt_pose = env.get_target_pose(target_name)
    place_dx = final_obj_pose["x"] - final_tgt_pose["x"]
    place_dy = final_obj_pose["y"] - final_tgt_pose["y"]
    place_dist = math.hypot(place_dx, place_dy)
    place_success = place_dist < 0.05 and grasp_ok

    # Extract predicates
    from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
    poses = env.get_all_object_poses()
    targets = {}
    for cn in env.cube_names:
        try:
            targets[cn] = env.get_object_pose(cn)
        except ValueError:
            targets[cn] = {"x": 0.0, "y": 0.0, "z": 0.0}
    targets["table"] = env.get_object_pose("table")
    thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
    predicates = physical_state_to_predicates(poses, targets, thresholds)

    return {
        "primitive": "place",
        "object": object_name,
        "target": target_name,
        "success": place_success,
        "grasp_success": grasp_ok,
        "place_success": place_success,
        "final_object_pose": final_obj_pose,
        "place_distance": place_dist,
        "extracted_predicates": [str(p) for p in predicates],
        "failure_type": None if place_success else ("grasp_failed" if not grasp_ok else "place_failed"),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug robosuite primitives")
    parser.add_argument("--primitive", required=True, choices=["reach", "grasp", "place"])
    parser.add_argument("--object", default="cube1")
    parser.add_argument("--target", default="cube2")
    args = parser.parse_args()

    if args.primitive == "reach":
        result = run_reach(args.object)
    elif args.primitive == "grasp":
        result = run_grasp(args.object)
    elif args.primitive == "place":
        result = run_place(args.object, args.target)
    else:
        result = {"error": f"Unknown primitive: {args.primitive}"}

    print(json.dumps(result, indent=2))

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / f"robosuite_primitive_{args.primitive}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResult saved to {output_path}")


if __name__ == "__main__":
    main()
