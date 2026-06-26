"""Record one 4-block reverse stacking episode with BFS planner."""

import os, sys, time
from pathlib import Path
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["MUJOCO_GL"] = "egl"

import vlm_nsm.executors.stack_four_cube_env as _
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID, GainSchedule,
)
from vlm_nsm.core.task_spec import TaskSpec
from vlm_nsm.embodied.task_runner import run_task

def _init_stack(env, task):
    for jn, x, y, z in [
        ('cubeA_joint0', -0.12, -0.08, 0.82),
        ('cubeB_joint0', -0.12, -0.08, 0.85),
        ('cubeC_joint0', -0.12, -0.08, 0.88),
        ('cubeD_joint0', -0.12, -0.08, 0.91),
    ]:
        env.env.sim.data.set_joint_qpos(jn, np.array([x, y, z, 1, 0, 0, 0]))
    env.env.sim.forward()

task = TaskSpec(
    name='stack_4block_reverse', env_name='Stack',
    env_kwargs={'robots': 'Panda'}, planner='bfs', n_objects=4,
    skills=['reach', 'grasp', 'lift', 'reach_place', 'release_place', 'retreat'],
    goal_predicates=[['on', 'cube4', 'cube3'], ['on', 'cube3', 'cube2'],
                     ['on', 'cube2', 'cube1'], ['on', 'cube1', 'table']],
    target_stack=['cube1', 'cube2', 'cube3', 'cube4'], horizon=80000,
    init_hook=_init_stack,
)

env = RobosuiteEnvWrapper(
    env_name='Stack', robots='Panda',
    cube_names=['cube1', 'cube2', 'cube3', 'cube4'],
    render_mode='offscreen',
)
pid = OperationalSpacePID(gain_schedule=GainSchedule())

# Increase step limit for 4-block
import vlm_nsm.embodied.task_runner as _tr
_tr._STEP_LIMIT = 2000

# Frame capture
env.reset()
task.init_hook(env, task)
env.env.horizon = 80000

frames = []
_orig_step = env.step

def _capture_step(action):
    result = _orig_step(action)
    try:
        img = env.env.sim.render(camera_name="agentview", height=512, width=512)
        frames.append(img[::2, ::2].copy())
    except Exception:
        pass
    return result

env.step = _capture_step

# Run one episode with BFS
print("Recording 4-block BFS reverse stacking...")
r = run_task(env, task, pid)
print(f"Result: success={r.get('goal_success', False)} "
      f"steps={r.get('num_steps', '?')} "
      f"failure={r.get('failure_type', '?')}")
print(f"Phase trace: {r.get('phase_trace', {})}")
print(f"Plan: {len(r.get('plan_actions', []))} actions")

# Save video
timestamp = time.strftime("%Y%m%d_%H%M%S")
output = f"generated/videos/4block_reverse_bfs_{timestamp}.mp4"
Path(output).parent.mkdir(parents=True, exist_ok=True)
h, w = frames[0].shape[:2]
writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
print(f"Saved {len(frames)} frames to {output}")
env.env.close() if hasattr(env.env, 'close') else None
