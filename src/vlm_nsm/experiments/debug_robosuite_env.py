"""Debug script to inspect robosuite Stack environment.

Prints action_spec, observable keys, object poses, and EEF pose.

Usage:
    python -m vlm_nsm.experiments.debug_robosuite_env
"""

from __future__ import annotations

import json
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper


def main():
    try:
        env = RobosuiteEnvWrapper(
            env_name="Stack",
            robots="Panda",
            render_mode="none",
        )
    except ImportError as e:
        print(f"ERROR: {e}")
        return

    obs = env.reset()

    debug_info = {
        "env_name": "Stack",
        "robot": "Panda",
        "render_mode": "none",
        "action_spec": {},
        "observable_keys": [],
        "available_bodies": env.get_available_bodies(),
        "cube_body_mapping": {},
        "cube_poses": {},
        "table_pose": {},
        "eef_pose": {},
        "gripper_state": {},
    }

    # Action spec
    if hasattr(env.env, "action_spec"):
        lo, hi = env.env.action_spec
        debug_info["action_spec"] = {
            "low": lo.tolist() if hasattr(lo, "tolist") else str(lo),
            "high": hi.tolist() if hasattr(hi, "tolist") else str(hi),
            "dim": len(lo) if hasattr(lo, "__len__") else 7,
        }

    # Observable keys
    if hasattr(env.env, "_observables"):
        debug_info["observable_keys"] = list(env.env._observables.keys())

    # Cube body mapping
    for cn in env.cube_names:
        body_name = env._cube_to_body.get(cn, "?")
        debug_info["cube_body_mapping"][cn] = body_name

    # Cube poses
    for cn in env.cube_names:
        try:
            pose = env.get_object_pose(cn)
            debug_info["cube_poses"][cn] = pose
        except ValueError as e:
            debug_info["cube_poses"][cn] = {"error": str(e)}

    # Table
    debug_info["table_pose"] = env.get_object_pose("table")

    # EEF
    try:
        debug_info["eef_pose"] = env.get_eef_pose()
    except ValueError as e:
        debug_info["eef_pose"] = {"error": str(e)}

    # Gripper
    try:
        debug_info["gripper_state"] = {
            k: str(v) if not isinstance(v, (int, float, bool, str, dict)) else v
            for k, v in env.get_gripper_state().items()
        }
    except ValueError as e:
        debug_info["gripper_state"] = {"error": str(e)}

    print(json.dumps(debug_info, indent=2))

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "robosuite_env_debug.json"
    output_path.write_text(json.dumps(debug_info, indent=2), encoding="utf-8")
    print(f"\nDebug info saved to {output_path}")


if __name__ == "__main__":
    main()
