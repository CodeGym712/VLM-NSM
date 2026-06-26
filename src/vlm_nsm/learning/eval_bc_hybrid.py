"""Evaluate a BC or residual BC model in hybrid reach-only mode.

Uses the shared episode runner (run_episode) for proper state machines,
termination logic, and recovery behaviors.

Usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python3 -m vlm_nsm.learning.eval_bc_hybrid \
        --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
        --num_episodes 20 --device cuda --seed 0 \
        --bc_phases reach_pick reach_place

    # Scripted-only (no BC override):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python3 -m vlm_nsm.learning.eval_bc_hybrid \
        --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
        --num_episodes 20 --seed 0 \
        --bc_phases none
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.train_bc import MLPBCPolicy

PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
N_PHASES = 6


def _get_bc_action(
    model, obs, device, action_mean, action_std, fixed_zero_dims=None,
) -> np.ndarray:
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    action_t = model(obs_t).squeeze(0).cpu()
    action = action_t.detach().numpy()
    if action_mean is not None and action_std is not None:
        action = action * action_std + action_mean
    action = np.clip(action, -1.0, 1.0)
    if fixed_zero_dims:
        for d in fixed_zero_dims:
            if 0 <= d < len(action):
                action[d] = 0.0
    return action


def main():
    parser = argparse.ArgumentParser(description="Eval BC in hybrid reach-only mode")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="Environment seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--condition_on_phase", action="store_true", default=False)
    parser.add_argument("--bc_phases", type=str, nargs="+", default=["reach_pick", "reach_place"],
                        help="BC-controlled phases, or 'none' for scripted-only")
    parser.add_argument("--fixed_zero_action_dims", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (overrides --output_dir default name)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (results use a derived name)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Resolve bc_phases: "none" → empty list (scripted-only)
    bc_phase_list: list[str] = []
    if args.bc_phases and args.bc_phases != ["none"] and args.bc_phases != ["None"]:
        bc_phase_list = [p for p in args.bc_phases if p in PHASE_NAMES]

    # Load checkpoint (only needed if BC phases are active)
    model = None
    condition_on_phase = args.condition_on_phase
    obs_dim = 11
    action_mean = None
    action_std = None

    if bc_phase_list:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            obs_dim = checkpoint.get("obs_dim", 11)
            action_dim = checkpoint.get("action_dim", 7)
            hidden_dims = checkpoint.get("hidden_dims", [256, 256])
            ckpt_cond = checkpoint.get("condition_on_phase", False)
        else:
            state_dict = checkpoint
            obs_dim = state_dict["net.0.weight"].shape[1]
            action_dim = state_dict["net.4.weight"].shape[0]
            hidden_dims = [state_dict["net.0.weight"].shape[0]]
            if "net.2.weight" in state_dict:
                hidden_dims.append(state_dict["net.2.weight"].shape[0])
            ckpt_cond = False

        condition_on_phase = args.condition_on_phase or ckpt_cond
        if not condition_on_phase and obs_dim > 11:
            condition_on_phase = True

        model = MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded checkpoint from {args.checkpoint}")

        # Load action norm stats
        ckpt_dir = Path(args.checkpoint).parent
        norm_path = ckpt_dir.parent / "action_norm_stats.json"
        if not norm_path.exists():
            norm_path = Path(PROJECT_ROOT) / "generated/results/stage_7B-1/action_norm_stats.json"
        if norm_path.exists():
            cfg = json.loads(norm_path.read_text())
            action_mean = np.array(cfg.get("action_mean", [0.0]*7), dtype=np.float32)
            action_std = np.array(cfg.get("action_std", [1.0]*7), dtype=np.float32)
            print(f"Loaded action norm from {norm_path}")

    # Load execution config
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = json.loads(config_path.read_text())
            print(f"Loaded config from {config_path}")

    bc_phase_ids = {PHASE_NAMES.index(p) for p in bc_phase_list} if bc_phase_list else set()
    print(f"BC phases: {bc_phase_list} (ids={bc_phase_ids})")

    # Setup
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import build_stacking_goal
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode

    cubes = ["cube1", "cube2"]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        raise RuntimeError(f"Planner failed: {plan_result.error}")

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=cubes, render_mode="none")
    horizon = env.env.horizon if hasattr(env.env, "horizon") else 1000

    # Note: robosuite 1.5.2 does not support env.seed() as callable.
    # The seed arg is recorded for metadata; env randomization is inherent.
    if args.seed is not None:
        print(f"Seed arg recorded (env seeding not available in robosuite 1.5.2)")

    ep_kwargs = dict(horizon=horizon, early_stop_on_goal=True)
    if config:
        for k in ["kp", "max_delta", "use_place_state_machine", "horizon_aware",
                   "place_retry", "grasp_close_steps", "grasp_max_retries",
                   "place_max_retries", "place_mode", "skip_retreat_if_goal_satisfied",
                   "holding_check_enabled", "lift_early_exit_enabled",
                   "close_steps_min", "close_steps_max", "close_stabilize_steps",
                   "release_steps_min", "release_steps_max", "post_release_stable_steps",
                   "close_mode", "close_steps_fast", "fallback_extend_close_if_not_holding"]:
            if k in config:
                ep_kwargs[k] = config[k]

    # Eval
    results = []
    t_start = time.time()

    for ep in range(args.num_episodes):
        if model is not None and bc_phase_ids:
            def policy_fn(obs, phase_id):
                obs_in = obs
                if condition_on_phase:
                    one_hot = np.zeros(N_PHASES, dtype=np.float32)
                    one_hot[phase_id] = 1.0
                    obs_in = np.concatenate([obs, one_hot])
                return _get_bc_action(model, obs_in, device, action_mean, action_std,
                                      args.fixed_zero_action_dims)
            override = policy_fn
        else:
            override = None

        result = run_episode(
            env, plan_result.actions, cubes, goal_predicates,
            policy_override=override,
            override_phases=bc_phase_ids if override else None,
            **ep_kwargs,
        )
        results.append(result)
        success = result.get("goal_success", False)
        steps = result.get("num_steps", 0)
        bc_st = result.get("bc_steps", 0)
        print(f"  Episode {ep+1}/{args.num_episodes}: success={success}, steps={steps}, bc_steps={bc_st}")

    env.env.close()
    elapsed = time.time() - t_start

    # Summary
    successes = sum(1 for r in results if r.get("goal_success", False))
    all_steps = [r.get("num_steps", 0) for r in results]
    all_bc = [r.get("bc_steps", 0) for r in results]
    all_scripted = [r.get("scripted_steps", 0) for r in results]
    all_failure_types = [r.get("failure_type") for r in results if r.get("failure_type")]

    # In scripted-only mode, wrapper is not created → bc_steps/scripted_steps=0.
    # Infer scripted steps from total_steps in that case.
    is_scripted_only = not bc_phase_list
    mean_bc = float(np.mean(all_bc)) if not is_scripted_only else 0.0
    mean_scripted = float(np.mean(all_scripted))
    if is_scripted_only:
        mean_scripted = float(np.mean(all_steps))

    summary = {
        "model": str(args.checkpoint),
        "seed": args.seed,
        "bc_phases": bc_phase_list,
        "num_episodes": args.num_episodes,
        "success_rate": successes / max(args.num_episodes, 1),
        "success_count": successes,
        "mean_steps": float(np.mean(all_steps)),
        "std_steps": float(np.std(all_steps)),
        "p95_steps": float(np.percentile(all_steps, 95)) if all_steps else 0.0,
        "mean_bc_steps": mean_bc,
        "mean_scripted_steps": mean_scripted,
        "elapsed_seconds": elapsed,
        "failure_type_counts": {ft: all_failure_types.count(ft) for ft in sorted(set(all_failure_types))},
        "episode_results": [
            {"success": r.get("goal_success", False), "steps": r.get("num_steps", 0),
             "bc_steps": r.get("bc_steps", 0), "failure_type": r.get("failure_type")}
            for r in results
        ],
    }

    print(f"\nResults: {successes}/{args.num_episodes} successful ({summary['success_rate']*100:.1f}%)")
    print(f"  Mean steps: {summary['mean_steps']:.1f} ± {summary['std_steps']:.1f}")
    print(f"  p95 steps: {summary['p95_steps']:.1f}")
    print(f"  Mean BC steps: {mean_bc:.1f}, Scripted: {mean_scripted:.1f}")
    print(f"  Failure types: {summary['failure_type_counts']}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Results saved to {output_path}")
    elif args.output_dir:
        # Derive name from bc_phases and seed
        phase_str = "_".join(bc_phase_list) if bc_phase_list else "scripted"
        seed_str = f"_seed{args.seed}" if args.seed is not None else ""
        out_name = f"{phase_str}{seed_str}_{args.num_episodes}ep.json"
        output_dir = ensure_dir(Path(args.output_dir))
        out_path = output_dir / out_name
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Results saved to {out_path}")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
