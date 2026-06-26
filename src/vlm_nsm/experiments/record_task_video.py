"""Record task execution to MP4 video.

Generic recorder — works for any task via function injection.
Captures agentview renders at each env.step and saves as MP4.

Usage:
    # 3-block stacking
    CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl nsm/bin/python \\
      -m vlm_nsm.experiments.record_task_video --task stacking_3block

    # Hanoi
    CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl nsm/bin/python \\
      -m vlm_nsm.experiments.record_task_video --task hanoi_3disk

Output: generated/videos/{task}_{timestamp}.mp4
"""

from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

import numpy as np


def _save_video(frames: list, output_path: str, fps: int = 20):
    """Save frames to MP4 via OpenCV (available on this system)."""
    import cv2
    h, w = frames[0].shape[:2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _make_env(cube_names: list[str]):
    """Create robosuite env with frame capture monkey-patch."""
    import vlm_nsm.executors.stack_three_cube_env as _  # noqa
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda",
        cube_names=cube_names, render_mode="offscreen",
    )
    env.env.horizon = 50000

    frames: list[np.ndarray] = []
    _orig_step = env.step

    def _capture_step(action):
        result = _orig_step(action)
        try:
            img = env.env.sim.render(
                camera_name="agentview", height=512, width=512)
            frames.append(img[::2, ::2].copy())  # 256x256
        except Exception:
            pass
        return result

    env.step = _capture_step
    return env, frames


def _run_stacking_3block_direct(env, pid, cubes):
    """Execute 3-block stacking via direct PID loop (avoids run_3block_replan bug)."""
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig
    from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig

    env.reset()
    env.env.horizon = 20000
    planner = PlannerAdapter("bfs")
    plan = planner.solve(cubes=cubes, target_stack=["cube3", "cube2", "cube1"])
    steps = 0

    for action in plan.actions:
        pick, place = action.args[0], action.args[2]
        # reach_pick
        obj = env.get_object_pose(pick)
        t = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}
        pid.set_phase("reach_pick"); pid.set_target(pos=t)
        for _ in range(500):
            e = env.get_eef_pose(); pid.set_current(e)
            a = pid.act("reach_pick", None); a[6] = -1.0
            try: env.step(a)
            except ValueError: break
            steps += 1
            if abs(e["x"]-t["x"])+abs(e["y"]-t["y"])+abs(e["z"]-t["z"]) < 0.03: break
        # grasp
        g = execute_grasp(env, pick, config=GraspConfig(
            kp=0.05, max_delta=0.01, max_retries=2, close_mode="adaptive"),
            skip_approach=True, skip_lift=True)
        steps += g.get("total_step_count", 0)
        # lift
        obj = env.get_object_pose(pick)
        lt = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.15}
        pid.set_phase("lift"); pid.set_target(pos=lt)
        for _ in range(500):
            e = env.get_eef_pose(); pid.set_current(e)
            a = pid.act("lift", None); a[6] = 1.0
            try: env.step(a)
            except ValueError: break
            steps += 1
            if abs(e["x"]-lt["x"])+abs(e["y"]-lt["y"])+abs(e["z"]-lt["z"]) < 0.03: break
        # reach_place
        tg = env.get_target_pose(place)
        pt = {"x": tg["x"], "y": tg["y"], "z": tg["z"] + 0.12}
        pid.set_phase("reach_place"); pid.set_target(pos=pt)
        for _ in range(1000):
            e = env.get_eef_pose(); pid.set_current(e)
            a = pid.act("reach_place", None); a[6] = 1.0
            try: env.step(a)
            except ValueError: break
            steps += 1
            if abs(e["x"]-pt["x"])+abs(e["y"]-pt["y"])+abs(e["z"]-pt["z"]) < 0.03: break
        # release
        p = execute_place(env, pick, place,
                          config=PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2,
                                             mode="simple"))
        steps += p.get("step_count", 0)
    return steps


def record_stacking_3block(output_path: str):
    """Record one 3-block stacking episode."""
    from vlm_nsm.embodied.controllers.operational_space_pid import (
        OperationalSpacePID, GainSchedule,
    )
    env, frames = _make_env(["cube1", "cube2", "cube3"])
    pid = OperationalSpacePID(gain_schedule=GainSchedule())

    steps = _run_stacking_3block_direct(env, pid, ["cube1", "cube2", "cube3"])
    print(f"3-block done. steps={steps}")

    print(f"Saving {len(frames)} frames to {output_path}...")
    _save_video(frames, output_path)
    print(f"Saved: {output_path}")


def record_hanoi(output_path: str):
    """Record one Hanoi 3-disk episode."""
    from vlm_nsm.embodied.controllers.operational_space_pid import (
        OperationalSpacePID, GainSchedule,
    )
    from vlm_nsm.experiments.run_hanoi_eval import run_hanoi_episode

    env, frames = _make_env(["cube1", "cube2", "cube3"])
    pid = OperationalSpacePID(gain_schedule=GainSchedule())

    result = run_hanoi_episode(env, pid, n_disks=3)
    print(f"Hanoi: success={result['success']} steps={result['num_steps']}")

    print(f"Saving {len(frames)} frames to {output_path}...")
    _save_video(frames, output_path)
    print(f"Saved: {output_path}")
    env.env.close() if hasattr(env.env, 'close') else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["stacking_3block", "hanoi_3disk"])
    parser.add_argument("--output_dir", default="generated/videos")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/{args.task}_{timestamp}.mp4"

    if args.task == "stacking_3block":
        record_stacking_3block(output_path)
    elif args.task == "hanoi_3disk":
        record_hanoi(output_path)


if __name__ == "__main__":
    main()
