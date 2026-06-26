"""Record execution videos for both paths.

Path A (Teacher): best scripted config — regrasp + safe place, 70%
Path B (VLA): S1 reach_pick XYZ override — M0 D1-small

Output: generated/videos/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def record_episode(env, output_path, fps=20):
    """Record RGB frames during episode execution and save as MP4."""
    frames = []
    _orig_step = env.step

    def recording_step(action):
        img = env.render_rgb("agentview", 512, 512)
        frames.append(img)
        return _orig_step(action)

    env.step = recording_step
    return frames, _orig_step


def save_video(frames, path, fps=20):
    """Save frames as MP4 video."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return True


def record_teacher_episode(seed: int, output_path: str, enhanced: bool = True):
    """Record a single scripted teacher episode."""
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode
    from vlm_nsm.core.data_types import Action

    cubes = ["cube1", "cube2"]
    # Alternate tasks for variety
    if seed % 2 == 0:
        obj, tgt = "cube1", "cube2"
    else:
        obj, tgt = "cube2", "cube1"

    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda", cube_names=cubes,
        render_mode="offscreen",
    )
    env.reset()

    # Record
    frames, _orig_step = record_episode(env, output_path)

    action = Action("move", [obj, "table", tgt])
    goal = [["on", obj, tgt]]

    place_kwargs = {}
    if enhanced and obj == "cube2" and tgt == "cube1":
        place_kwargs = {"place_mode": "safe"}

    try:
        result = run_episode(
            env=env, actions=[action], cubes=cubes, goal_predicates=goal,
            horizon=1000, early_stop_on_goal=True,
            **place_kwargs,
        )
        success = bool(result.get("goal_success", False))
    except Exception as e:
        success = False
        print(f"  Error: {e}")
    finally:
        env.step = _orig_step

    save_video(frames, output_path)
    return success, len(frames)


def record_vla_episode(seed: int, output_path: str):
    """Record a VLA S1 reach_pick override episode."""
    import torch
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode
    from vlm_nsm.core.data_types import Action
    from vlm_nsm.vla.models.vision_encoder import VLABCModel
    from vlm_nsm.neuro_symbolic.safety_gate import GateConfig, NeuroSymbolicSafetyGate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load M0
    ckpt = torch.load("generated/results/vla/vla_m0_d1_small/best.pt",
                      map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = VLABCModel(
        backbone="resnet18", pretrained=False,
        vision_output_dim=cfg.get("vision_dim", 256),
        symbolic_dim=cfg.get("symbolic_dim", 42),
        hidden_dims=tuple(cfg.get("hidden_dims", [512, 256])),
        action_dim=4,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gate = NeuroSymbolicSafetyGate(GateConfig(
        use_confidence=False, max_xyz_deviation=0.025,
    ))

    stats = json.loads(
        Path("generated/datasets/vla_demos/d1_small_200ep/stats.json").read_text())
    action_mean = np.array(stats["action_mean"], dtype=np.float32)
    action_std = np.array(stats["action_std"], dtype=np.float32)

    cubes = ["cube1", "cube2"]
    if seed % 2 == 0:
        obj, tgt = "cube1", "cube2"
    else:
        obj, tgt = "cube2", "cube1"

    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda", cube_names=cubes,
        render_mode="offscreen",
    )
    env.reset()
    env._vla_phase = ""

    frames = []
    _orig_step = env.step

    def vla_step(raw_action):
        # Record frame
        img = env.render_rgb("agentview", 512, 512)
        frames.append(img)

        # VLA override for reach_pick
        phase_name = getattr(env, '_vla_phase', '') or ''
        raw = np.array(raw_action, dtype=np.float32)

        if phase_name == "reach_pick":
            img_a = env.render_rgb("agentview", 128, 128)
            img_f = env.render_rgb("frontview", 128, 128)
            eef = env.get_eef_pose()
            grip = env.get_gripper_state()

            av = torch.from_numpy(img_a.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
            fv = torch.from_numpy(img_f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
            prop = torch.tensor([[eef["x"],eef["y"],eef["z"],grip.get("open",0.),0.,0.,0.]], device=device)
            sym = torch.zeros(1, 42, device=device)

            with torch.no_grad():
                vla_4d = model(av, fv, prop, sym).cpu().numpy().ravel()
            vla_dn = vla_4d * action_std + action_mean

            vla_7d = np.zeros(7, dtype=np.float32)
            vla_7d[:3] = vla_dn[:3]
            vla_7d[6] = raw[6]
            gr = gate.validate(vla_7d, raw, 0.8,
                               np.array([eef["x"],eef["y"],eef["z"]]), phase_name)
            if gr.approved:
                return _orig_step(vla_7d)
        return _orig_step(raw)

    env.step = vla_step

    action = Action("move", [obj, "table", tgt])
    goal = [["on", obj, tgt]]

    try:
        result = run_episode(
            env=env, actions=[action], cubes=cubes, goal_predicates=goal,
            horizon=1000, early_stop_on_goal=True,
            policy_override=lambda obs, pid: np.zeros(7),
            override_phases=set(),
        )
        success = bool(result.get("goal_success", False))
    except Exception as e:
        success = False
        print(f"  Error: {e}")
    finally:
        env.step = _orig_step

    save_video(frames, output_path)
    return success, len(frames)


def main():
    parser = argparse.ArgumentParser(description="Record demo videos")
    parser.add_argument("--output_dir", default="generated/videos")
    parser.add_argument("--num_episodes", type=int, default=2,
                        help="Videos per route")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=== Path A: Teacher (best config: regrasp + safe place) ===")
    for i in range(args.num_episodes):
        seed = args.seed + i
        path = output / f"teacher_demo_{i+1}.mp4"
        print(f"  [{i+1}/{args.num_episodes}] seed={seed} ...")
        t0 = time.time()
        success, frames = record_teacher_episode(seed, path)
        elapsed = time.time() - t0
        print(f"    success={success} frames={frames} {elapsed:.1f}s -> {path}")

    print("\n=== Path B: VLA S1 (M0 D1-small, reach_pick override) ===")
    for i in range(args.num_episodes):
        seed = args.seed + 100 + i
        path = output / f"vla_s1_demo_{i+1}.mp4"
        print(f"  [{i+1}/{args.num_episodes}] seed={seed} ...")
        t0 = time.time()
        try:
            success, frames = record_vla_episode(seed, path)
            elapsed = time.time() - t0
            print(f"    success={success} frames={frames} {elapsed:.1f}s -> {path}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nDone. Videos in {output}/")


if __name__ == "__main__":
    main()
