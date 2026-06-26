"""S0 Live Shadow — VLA predicts alongside scripted execution without controlling.

Runs scripted episodes while VLA predicts actions in parallel at EVERY step.
Measures per-phase action MSE, gripper sign accuracy, simulated gate acceptance,
and direction cosine.

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python \\
      -m vlm_nsm.experiments.eval_vla_live_shadow \\
      --checkpoint generated/results/vla/vla_m0_d1_small/best.pt \\
      --num_episodes 50 --device cuda:0 \\
      --output_dir generated/results/vla/s0_live_shadow
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlm_nsm.neuro_symbolic.safety_gate import GateConfig, NeuroSymbolicSafetyGate
from vlm_nsm.neuro_symbolic.symbolic_condition import SymbolicConditionEncoder
from vlm_nsm.skills.skill_types import SkillCommand
from vlm_nsm.vla.models.vision_encoder import VLABCModel

_PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def load_m0(checkpoint_path: str, device: torch.device) -> VLABCModel:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = VLABCModel(
        backbone=cfg.get("backbone", "resnet18"), pretrained=False,
        vision_output_dim=cfg.get("vision_dim", 256),
        symbolic_dim=cfg.get("symbolic_dim", 42),
        hidden_dims=tuple(cfg.get("hidden_dims", [512, 256])),
        action_dim=4,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_live_shadow(
    checkpoint_path: str,
    num_episodes: int = 50,
    seed: int = 9000,
    device: str = "cuda:0",
    output_dir: str | None = None,
    image_size: int = 128,
    max_steps: int = 1000,
) -> dict[str, Any]:
    """Run live shadow evaluation."""
    device_obj = torch.device(device)
    model = load_m0(checkpoint_path, device_obj)
    gate = NeuroSymbolicSafetyGate(GateConfig(use_confidence=False))

    # Late imports
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode
    from vlm_nsm.core.data_types import Action
    from vlm_nsm.vla.data.vla_dataset import VLADatasetManifest

    # Load action stats for denormalization
    stats_path = Path("generated/datasets/vla_demos/d1_small_200ep/stats.json")
    stats = json.loads(stats_path.read_text())
    action_mean = np.array(stats["action_mean"], dtype=np.float32)
    action_std = np.array(stats["action_std"], dtype=np.float32)

    cubes = ["cube1", "cube2"]
    tasks = [
        {"object_id": "cube1", "target_object_id": "cube2"},
        {"object_id": "cube2", "target_object_id": "cube1"},
    ]

    # Per-phase accumulators
    per_phase: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    gate_stats = {"total": 0, "accepted": 0, "reasons": defaultdict(int)}
    overall_mse: list[float] = []
    gripper_correct: list[float] = []
    direction_cosines: list[float] = []

    # Fix C6: Initialize symbolic condition encoder for S0 shadow evaluation
    sym_encoder = SymbolicConditionEncoder(objects=["cube1", "cube2"])

    total_start = time.time()
    success_count = 0

    for ep_idx in range(num_episodes):
        cmd = tasks[ep_idx % 2]
        ep_seed = seed + ep_idx

        # Collect observations during execution
        observations: list[dict[str, Any]] = []
        scripted_actions: list[np.ndarray] = []
        phase_ids: list[int] = []
        phase_names_list: list[str] = []

        # Phase tracking via mutable container
        _current_phase = [{"name": "", "id": 0}]

        # Create env
        env = RobosuiteEnvWrapper(
            env_name="Stack", robots="Panda", cube_names=cubes,
            render_mode="offscreen",
        )
        env.reset()

        # Monkey-patch step to capture RGB + action
        _orig_step = env.step

        def recording_step(raw_action):
            # Record RGB
            img_a = env.render_rgb("agentview", image_size, image_size)
            img_f = env.render_rgb("frontview", image_size, image_size)
            eef = env.get_eef_pose()
            grip = env.get_gripper_state()

            observations.append({
                "agentview": img_a.astype(np.float32) / 255.0,
                "frontview": img_f.astype(np.float32) / 255.0,
                "eef_pos": np.array([eef["x"], eef["y"], eef["z"]], dtype=np.float32),
                "gripper": np.array([grip.get("open", 0.0)], dtype=np.float32),
            })
            scripted_actions.append(np.array(raw_action[:4], dtype=np.float32))
            phase_ids.append(_current_phase[0]["id"])
            phase_names_list.append(_current_phase[0]["name"])
            return _orig_step(raw_action)

        env.step = recording_step

        # Read phase from wrapper after episode (via result)
        action = Action("move", [cmd["object_id"], "table", cmd["target_object_id"]])
        goal = [["on", cmd["object_id"], cmd["target_object_id"]]]

        try:
            result = run_episode(
                env=env, actions=[action], cubes=cubes, goal_predicates=goal,
                horizon=max_steps, early_stop_on_goal=True,
            )
            # Backfill phases from result
            phase_trace = result.get("phase_trace", {})
            phase_order = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
            phase_seq = []
            for pname in phase_order:
                count = phase_trace.get(pname, 0)
                pid = {"reach_pick": 0, "grasp": 1, "lift": 2,
                       "reach_place": 3, "release": 4, "retreat": 5}.get(pname, 0)
                phase_seq.extend([(pname, pid)] * count)

            for i, (pname, pid) in enumerate(phase_seq):
                if i < len(phase_ids):
                    phase_ids[i] = pid
                    phase_names_list[i] = pname

            success = bool(result.get("goal_success", False))
            if success:
                success_count += 1
        except Exception as e:
            print(f"  [{ep_idx+1}/{num_episodes}] ERROR: {e}")
            continue
        finally:
            env.step = _orig_step

        # Run VLA inference on all captured observations
        n_steps = len(observations)
        if n_steps == 0:
            continue

        for i in range(n_steps):
            obs = observations[i]
            av = torch.from_numpy(obs["agentview"]).permute(2, 0, 1).unsqueeze(0).to(device_obj)
            fv = torch.from_numpy(obs["frontview"]).permute(2, 0, 1).unsqueeze(0).to(device_obj)
            prop = np.zeros(7, dtype=np.float32)
            prop[:3] = obs["eef_pos"]
            prop[3] = obs["gripper"][0]
            prop_t = torch.from_numpy(prop).unsqueeze(0).to(device_obj)
            # Phase 2.0: real symbolic condition
            phase_name = phase_names_list[i] if i < len(phase_names_list) else ""
            _sc = SkillCommand("pick_place", cmd["object_id"],
                              target_object_id=cmd["target_object_id"], relation="on")
            _sv = sym_encoder.encode(_sc, phase_name=phase_name)
            _raw = _sv.as_vector()
            _padded = np.zeros(42, dtype=np.float32)
            _padded[:len(_raw)] = _raw
            sym = torch.from_numpy(_padded).unsqueeze(0).to(device_obj)

            with torch.no_grad():
                vla_pred = model(av, fv, prop_t, sym).cpu().numpy().ravel()

            # Denormalize
            vla_denorm = vla_pred * action_std + action_mean
            scripted = scripted_actions[i]
            scripted_denorm = scripted * action_std + action_mean

            # Metrics
            mse = float(np.mean((vla_denorm - scripted_denorm) ** 2))
            overall_mse.append(mse)

            # Gripper sign
            pred_sign = 1 if vla_denorm[3] > 0 else -1
            true_sign = 1 if scripted_denorm[3] > 0 else -1
            gripper_correct.append(1.0 if pred_sign == true_sign else 0.0)

            # Direction cosine (XYZ)
            v_xyz = vla_denorm[:3]
            s_xyz = scripted_denorm[:3]
            v_norm = np.linalg.norm(v_xyz)
            s_norm = np.linalg.norm(s_xyz)
            if v_norm > 1e-8 and s_norm > 1e-8:
                cos_sim = np.dot(v_xyz, s_xyz) / (v_norm * s_norm)
                direction_cosines.append(float(cos_sim))

            # Simulate gate
            phase_name = phase_names_list[i] if i < len(phase_names_list) else ""
            vla_7d = np.zeros(7, dtype=np.float32)
            vla_7d[:3] = vla_denorm[:3]
            vla_7d[6] = vla_denorm[3]
            scripted_7d = np.zeros(7, dtype=np.float32)
            scripted_7d[:3] = scripted_denorm[:3]
            scripted_7d[6] = scripted_denorm[3]

            gate_result = gate.validate(
                vla_action=vla_7d, scripted_action=scripted_7d,
                vla_confidence=0.8,
                eef_pos=obs["eef_pos"], phase=phase_name,
            )
            gate_stats["total"] += 1
            if gate_result.approved:
                gate_stats["accepted"] += 1
            else:
                gate_stats["reasons"][gate_result.reason] += 1

            # Per-phase
            pname = phase_name if phase_name else "unknown"
            per_phase[pname]["mse"].append(mse)
            per_phase[pname]["cos_sim"].append(
                direction_cosines[-1] if direction_cosines else 0.0)
            per_phase[pname]["gripper_acc"].append(gripper_correct[-1])

        print(f"  [{ep_idx+1}/{num_episodes}] success={success} steps={n_steps}")

    total_elapsed = time.time() - total_start

    # Summarize
    phase_summary = {}
    for phase, metrics in sorted(per_phase.items()):
        phase_summary[phase] = {
            "count": len(metrics["mse"]),
            "mse": float(np.mean(metrics["mse"])),
            "direction_cosine": float(np.mean(metrics["cos_sim"])),
            "gripper_accuracy": float(np.mean(metrics["gripper_acc"])),
        }

    result = {
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_count / max(num_episodes, 1),
        "total_steps": len(overall_mse),
        "overall_mse": float(np.mean(overall_mse)),
        "gripper_sign_accuracy": float(np.mean(gripper_correct)),
        "direction_cosine": float(np.mean(direction_cosines)),
        "gate_acceptance": gate_stats["accepted"] / max(gate_stats["total"], 1),
        "gate_reasons": dict(gate_stats["reasons"]),
        "per_phase": phase_summary,
        "device": str(device_obj),
        "elapsed_s": total_elapsed,
    }

    print(f"\nS0 Live Shadow Complete:")
    print(f"  Success: {success_count}/{num_episodes} ({result['success_rate']:.1%})")
    print(f"  Overall MSE: {result['overall_mse']:.4f}")
    print(f"  Gripper acc: {result['gripper_sign_accuracy']:.1%}")
    print(f"  Direction cos: {result['direction_cosine']:.3f}")
    print(f"  Gate acceptance: {result['gate_acceptance']:.1%}")
    for phase, s in phase_summary.items():
        if s["count"] > 50:
            print(f"  {phase}: MSE={s['mse']:.4f} cos={s['direction_cosine']:.3f} grip={s['gripper_accuracy']:.1%}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved to {out / 'report.json'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="S0 Live Shadow Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_live_shadow(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
