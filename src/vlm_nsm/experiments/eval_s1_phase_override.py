"""S1 Phase Override — VLA controls reach_pick XYZ, scripted controls gripper+rest.

Gate A: Can VLA control the simplest phase (reach_pick) in closed-loop?

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl nsm/bin/python \\
      -m vlm_nsm.experiments.eval_s1_phase_override \\
      --checkpoint generated/results/vla/vla_m0_d1_small/best.pt \\
      --num_episodes 50 --device cuda:0 \\
      --output_dir generated/results/vla/s1_reach_pick
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

_OVERRIDE_PHASES = {"reach_pick"}  # S1: reach_pick only


def load_model(checkpoint_path: str, device: torch.device):
    """Auto-detect model type (M0 MLP or M1 Diffusion) from checkpoint config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_type = cfg.get("model", "")

    if "Diffusion" in model_type or "diffusion" in str(checkpoint_path):
        from vlm_nsm.vla.models.diffusion_policy import DiffusionPolicy
        model = DiffusionPolicy(
            backbone=cfg.get("backbone", "resnet18"), pretrained=False,
            vision_output_dim=cfg.get("vision_dim", 256),
            symbolic_dim=cfg.get("symbolic_dim", 42),
            action_dim=4, horizon=cfg.get("horizon", 16),
            unet_dims=tuple(cfg.get("unet_dims", [256, 512, 1024])),
            num_diffusion_steps=cfg.get("diffusion_steps", 100),
            num_inference_steps=cfg.get("inference_steps", 16),
        ).to(device)
    else:
        from vlm_nsm.vla.models.vision_encoder import VLABCModel
        model = VLABCModel(
            backbone=cfg.get("backbone", "resnet18"), pretrained=False,
            vision_output_dim=cfg.get("vision_dim", 256),
            symbolic_dim=cfg.get("symbolic_dim", 42),
            hidden_dims=tuple(cfg.get("hidden_dims", [512, 256])),
            action_dim=4,
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_type


def run_s1(
    checkpoint_path: str,
    num_episodes: int = 50,
    seed: int = 9000,
    device: str = "cuda:0",
    output_dir: str | None = None,
    dataset_stats: str = "generated/datasets/vla_demos/d1_small_200ep/stats.json",
    no_symbolic: bool = False,
    gate_config_path: str | None = None,
    max_xyz_deviation: float = 0.025,
    save_trace: bool = False,
    random_symbolic: bool = False,
    random_vector_path: str = "generated/results/vla/p0_fixes/b4_5_random_fixed_vector.npy",
) -> dict[str, Any]:
    device_obj = torch.device(device)
    model, model_type = load_model(checkpoint_path, device_obj)
    print(f"Model: {model_type or 'M0 (VLABCModel)'}")
    is_diffusion = "Diffusion" in str(model_type or "")

    # Load per-phase gate calibration if provided
    per_phase_gate: dict[str, float] = {}
    if gate_config_path:
        gc = json.loads(Path(gate_config_path).read_text())
        per_phase_gate = gc.get("per_phase_max_xyz_deviation", {})
        print(f"Loaded per-phase gate: {per_phase_gate}")

    gate = NeuroSymbolicSafetyGate(GateConfig(
        use_confidence=False,
        max_xyz_deviation=max_xyz_deviation,
        max_gripper_deviation=0.5,
    ))

    # Phase 2.0: real symbolic condition encoder
    sym_encoder = SymbolicConditionEncoder(objects=["cube1", "cube2"])

    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode
    from vlm_nsm.core.data_types import Action

    stats = json.loads(Path(dataset_stats).read_text())
    action_mean = np.array(stats["action_mean"], dtype=np.float32)
    action_std = np.array(stats["action_std"], dtype=np.float32)

    cubes = ["cube1", "cube2"]
    tasks = [
        {"object_id": "cube1", "target_object_id": "cube2"},
        {"object_id": "cube2", "target_object_id": "cube1"},
    ]

    phase_success: dict[str, list[bool]] = defaultdict(list)
    gate_accepted: list[int] = []
    gate_total: list[int] = []
    vla_caused_failures: list[dict] = []
    overall_success: list[bool] = []
    ep_steps: list[int] = []

    total_start = time.time()

    for ep_idx in range(num_episodes):
        cmd = tasks[ep_idx % 2]
        ep_seed = seed + ep_idx

        env = RobosuiteEnvWrapper(
            env_name="Stack", robots="Panda", cube_names=cubes,
            render_mode="offscreen",
        )
        env.reset()

        # Phase tracking
        _orig_step = env.step
        _gate_ok = 0
        _gate_all = 0
        _phase_start_xy_err: float | None = None
        _phase_end_xy_err: float | None = None
        _episode_trace: list[dict[str, Any]] = []  # per-step trace

        # Initialize phase tracking attribute for wrapper to write to
        env._vla_phase = ""

        def s1_step(raw_action):
            nonlocal _gate_ok, _gate_all, _phase_start_xy_err, _phase_end_xy_err
            # Phase written by _RobosuiteRecordingWrapper.step() via _vla_phase
            phase_name = getattr(env, '_vla_phase', '') or ''
            raw = np.array(raw_action, dtype=np.float32)

            if phase_name in _OVERRIDE_PHASES:
                # VLA prediction
                img_a = env.render_rgb("agentview", 128, 128)
                img_f = env.render_rgb("frontview", 128, 128)
                eef = env.get_eef_pose()
                grip = env.get_gripper_state()

                av = torch.from_numpy(img_a.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device_obj)
                fv = torch.from_numpy(img_f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device_obj)
                prop = torch.tensor([[eef["x"],eef["y"],eef["z"],grip.get("open",0.),0.,0.,0.]], device=device_obj)
                _model_sym_dim = 42
                if no_symbolic:
                    sym = torch.zeros(1, _model_sym_dim, device=device_obj)
                elif random_symbolic:
                    sym = torch.from_numpy(B45_RANDOM_VEC).unsqueeze(0).to(device_obj)
                else:
                    _sc = SkillCommand(
                        skill_name="pick_place",
                        object_id=cmd["object_id"],
                        target_object_id=cmd["target_object_id"],
                        relation="on",
                    )
                    _sym_vec = sym_encoder.encode(_sc, phase_name=phase_name)
                    _raw = _sym_vec.as_vector()
                    _padded = np.zeros(_model_sym_dim, dtype=np.float32)
                    _padded[:len(_raw)] = _raw
                    sym = torch.from_numpy(_padded).unsqueeze(0).to(device_obj)

                with torch.no_grad():
                    if is_diffusion:
                        chunk = model.sample(av, fv, prop, sym)
                        vla_4d = chunk[0, 0, :].cpu().numpy().ravel()
                    else:
                        vla_4d = model(av, fv, prop, sym).cpu().numpy().ravel()
                vla_dn = vla_4d * action_std + action_mean

                # Safety gate
                vla_7d = np.zeros(7, dtype=np.float32)
                vla_7d[:3] = vla_dn[:3]
                vla_7d[6] = raw[6]
                scripted_7d = raw.copy()
                eef_arr = np.array([eef["x"], eef["y"], eef["z"]])
                gr = gate.validate(vla_7d, scripted_7d, 0.8, eef_arr, phase_name)
                _gate_all += 1
                _approved = bool(gr.approved)
                # Per-phase override: apply per-phase threshold IN ADDITION TO
                # global gate checks (not INSTEAD OF). The global gate handles
                # NaN, workspace, and phase semantic checks.
                if per_phase_gate and _approved:
                    _phase_dev = float(np.linalg.norm(vla_7d[:3] - scripted_7d[:3]))
                    _phase_thresh = per_phase_gate.get(phase_name, max_xyz_deviation)
                    _approved = _phase_dev <= _phase_thresh
                if _approved:
                    _gate_ok += 1
                    action_out = vla_7d
                    _controller = "vla"
                else:
                    action_out = scripted_7d
                    _controller = "scripted"

                # Compute direction cosine for trace
                vn = float(np.linalg.norm(vla_dn[:3]))
                sn = float(np.linalg.norm(scripted_7d[:3]))
                _cos = 0.0
                if vn > 1e-8 and sn > 1e-8:
                    _cos = float(np.dot(vla_dn[:3], scripted_7d[:3]) / (vn * sn))

                # Record per-step trace
                if save_trace:
                    _episode_trace.append({
                        "phase": phase_name,
                        "gate_approved": _approved,
                        "controller": _controller,
                        "vla_xyz": vla_dn[:3].tolist(),
                        "scripted_xyz": scripted_7d[:3].tolist(),
                        "cos_sim": _cos,
                    })

                # Track phase-level reach_pick error
                if phase_name == "reach_pick":
                    obj = env.get_object_pose(cmd["object_id"])
                    obj_xy = np.array([obj["x"], obj["y"]])
                    eef_xy = np.array([eef["x"], eef["y"]])
                    err = float(np.linalg.norm(eef_xy - obj_xy))
                    if _phase_start_xy_err is None:
                        _phase_start_xy_err = err
                    _phase_end_xy_err = err

                return _orig_step(action_out)
            else:
                # Non-override phase: record trace too
                if save_trace:
                    raw_7d = np.array(raw, dtype=np.float32)
                    rn = float(np.linalg.norm(raw_7d[:3]))
                    _episode_trace.append({
                        "phase": phase_name,
                        "gate_approved": True,
                        "controller": "scripted",
                        "vla_xyz": [0.0, 0.0, 0.0],
                        "scripted_xyz": raw_7d[:3].tolist(),
                        "cos_sim": 1.0,
                    })
                return _orig_step(raw)

        env.step = s1_step

        action = Action("move", [cmd["object_id"], "table", cmd["target_object_id"]])
        goal = [["on", cmd["object_id"], cmd["target_object_id"]]]

        try:
            # Force _RobosuiteRecordingWrapper creation via a dummy policy_override
            # (never actually used because override_phases=set())
            # This is needed so wrapper.step() writes _vla_phase to our env
            result = run_episode(
                env=env, actions=[action], cubes=cubes, goal_predicates=goal,
                horizon=1000, early_stop_on_goal=True,
                policy_override=lambda obs, pid: np.zeros(7),
                override_phases=set(),  # empty → never overrides
            )
            success = bool(result.get("goal_success", False))
        except Exception as e:
            print(f"  [{ep_idx+1}/{num_episodes}] ERROR: {e}")
            success = False
        finally:
            env.step = _orig_step

        overall_success.append(success)
        ep_steps.append(result.get("num_steps", 0) if 'result' in dir() else 0)
        gate_accepted.append(_gate_ok)
        gate_total.append(_gate_all)

        # Phase-level reach_pick success
        if _phase_end_xy_err is not None:
            phase_success["reach_pick"].append(_phase_end_xy_err < 0.02)

        print(f"  [{ep_idx+1}/{num_episodes}] success={success} gate={_gate_ok}/{_gate_all}")

        # Track VLA-caused failures
        if not success and _gate_all > 0 and _gate_ok / max(_gate_all, 1) > 0.5:
            vla_caused_failures.append({
                "episode": ep_idx,
                "failure_type": result.get("failure_type") if 'result' in dir() else "unknown",
            })

        # Save per-episode trace
        if save_trace and output_dir:
            trace_out = Path(output_dir) / "traces"
            trace_out.mkdir(parents=True, exist_ok=True)
            (trace_out / f"ep_{ep_idx:04d}.jsonl").write_text(
                "\n".join(json.dumps(s, ensure_ascii=False) for s in _episode_trace))

    total_elapsed = time.time() - total_start

    result = {
        "num_episodes": num_episodes,
        "overall_success": float(np.mean(overall_success)),
        "success_count": int(sum(overall_success)),
        "mean_steps": float(np.mean(ep_steps)),
        "gate_acceptance": sum(gate_accepted) / max(sum(gate_total), 1),
        "gate_total": sum(gate_total),
        "gate_accepted_count": sum(gate_accepted),
        "reach_pick_phase_success": float(np.mean(phase_success.get("reach_pick", [False]))),
        "vla_caused_failures": len(vla_caused_failures),
        "elapsed_s": total_elapsed,
    }

    print(f"\nS1 Reach Pick Override Complete:")
    print(f"  Overall: {result['overall_success']:.1%} ({result['success_count']}/{num_episodes})")
    print(f"  reach_pick phase success: {result['reach_pick_phase_success']:.1%}")
    print(f"  Gate acceptance: {result['gate_acceptance']:.1%}")
    print(f"  VLA-caused failures: {result['vla_caused_failures']}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    return result


def main():
    parser = argparse.ArgumentParser(description="S1 Phase Override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset_stats", default="generated/datasets/vla_demos/d1_small_200ep/stats.json")
    parser.add_argument("--no_symbolic", action="store_true", help="B4: use zeros for symbolic condition")
    parser.add_argument("--random_symbolic", action="store_true", help="B4.5: use fixed random vector for symbolic condition")
    parser.add_argument("--random_vector_path", default="generated/results/vla/p0_fixes/b4_5_random_fixed_vector.npy")
    parser.add_argument("--gate_config", default=None, help="Path to per-phase gate calibration JSON")
    parser.add_argument("--max_xyz_deviation", type=float, default=0.025, help="Global max XYZ deviation for gate")
    parser.add_argument("--save_trace", action="store_true", help="Save per-step trace to JSONL")
    parser.add_argument("--override_phases", default="reach_pick",
                       help="Comma-separated phases to override, e.g. 'reach_pick' (S1) or 'reach_pick,reach_place' (S2)")
    args = parser.parse_args()
    # Set override phases
    import sys
    import numpy as np
    global _OVERRIDE_PHASES, B45_RANDOM_VEC
    B45_RANDOM_VEC = None
    if args.random_symbolic:
        B45_RANDOM_VEC = np.load(args.random_vector_path).astype(np.float32)
        print(f"B4.5: random vector loaded ({B45_RANDOM_VEC.shape}), range=[{B45_RANDOM_VEC.min():.3f},{B45_RANDOM_VEC.max():.3f}]")
    _OVERRIDE_PHASES = set(args.override_phases.split(","))
    run_s1(checkpoint_path=args.checkpoint, num_episodes=args.num_episodes,
           seed=args.seed, device=args.device, output_dir=args.output_dir,
           dataset_stats=args.dataset_stats, no_symbolic=args.no_symbolic, random_symbolic=args.random_symbolic, random_vector_path=args.random_vector_path,
           gate_config_path=args.gate_config, max_xyz_deviation=args.max_xyz_deviation,
           save_trace=args.save_trace)


if __name__ == "__main__":
    main()
