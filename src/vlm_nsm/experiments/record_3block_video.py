"""Record 3-block stacking with replan to MP4 video."""

import os, sys, json
from pathlib import Path
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["MUJOCO_GL"] = "egl"

import vlm_nsm.executors.stack_three_cube_env as _
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import OperationalSpacePID
from vlm_nsm.experiments.run_3block_replan_v2 import _execute_skill_sequence, _check_predicates
import vlm_nsm.experiments.run_3block_replan_v2 as _v2
_v2._STEP_LIMIT = 800  # increase from default
from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter

env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
    cube_names=["cube1","cube2","cube3"], render_mode="none")
env.env.horizon = 10000
pid = OperationalSpacePID()
cubes = ["cube1","cube2","cube3"]

print("Recording 3-block replan episode...")
frames = []

# Override env.step to capture frames
_orig_step = env.step
def _step_with_capture(action):
    result = _orig_step(action)
    try:
        img = env.env.sim.render(camera_name="agentview", height=512, width=512)
        frames.append(img[::2,::2].copy())  # downsample to 256x256
    except Exception:
        pass
    return result
env.step = _step_with_capture

# Run episode
target_stack = ["cube3","cube2","cube1"]
planner = PlannerAdapter("bfs")
plan = planner.solve(cubes=cubes, target_stack=target_stack)
actions = list(plan.actions)

total_steps = 0
replans = 0
max_replans = 3

while replans <= max_replans:
    all_ok = True
    for idx, action in enumerate(actions):
        pick = action.args[0]
        place = action.args[2]
        result = _execute_skill_sequence(env, pid, pick, place)
        total_steps += result["steps"]
        print(f"  Action {idx+1}: {pick}→{place}, ok={result['success']}, steps={result['steps']}")

        if not result["success"]:
            all_ok = False
            if idx > 0 and replans < max_replans:
                mid_preds = [Predicate("on", [actions[0].args[0], actions[0].args[2]])]
                if not _check_predicates(env, mid_preds):
                    replans += 1
                    print(f"  COLLAPSE! Replan #{replans}")
                    break
            break

    if all_ok:
        goal_preds = [Predicate("on", [a.args[0],a.args[2]]) for a in actions]
        success = _check_predicates(env, goal_preds)
        print(f"Success: {success}, steps: {total_steps}, replans: {replans}")
        break
    if replans > max_replans:
        print(f"FAILED after {replans} replans")
        break

# Write video
out = Path("generated/videos/3block_replan.mp4")
out.parent.mkdir(parents=True, exist_ok=True)

try:
    import cv2
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved: {out} ({len(frames)} frames, {len(frames)/30:.1f}s)")
except ImportError:
    print("opencv not available, saving frames as npz")
    np.savez_compressed(str(out.with_suffix(".npz")), frames=np.stack(frames))
    print(f"Frames saved: {out.with_suffix('.npz')}")
