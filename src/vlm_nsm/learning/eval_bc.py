"""Evaluate MLP BC policy on robosuite stacking task.

Supports full BC control and hybrid (BC for selected phases, scripted for others).

Usage:
    MUJOCO_GL=egl python3 -m vlm_nsm.learning.eval_bc \\
        --checkpoint generated/models/stage_7B-2/bc_phase_conditioned/best.pt \\
        --num_episodes 20

    # Hybrid: BC only for reach phases
    MUJOCO_GL=egl python3 -m vlm_nsm.learning.eval_bc \\
        --checkpoint ... --num_episodes 20 \\
        --control_mode hybrid --bc_phases reach_pick reach_place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.robosuite_schema import OBS_DIM, ACTION_DIM
from vlm_nsm.learning.train_bc import MLPBCPolicy


PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def _get_bc_action(
    model: MLPBCPolicy,
    obs: np.ndarray,
    device: str,
    action_mean: np.ndarray | None,
    action_std: np.ndarray | None,
    fixed_zero_dims: list[int] | None = None,
) -> np.ndarray:
    """Get denormalized BC action."""
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


@torch.no_grad()
def evaluate_policy(
    model: MLPBCPolicy,
    num_episodes: int = 20,
    device: str = "cpu",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    condition_on_phase: bool = False,
    control_mode: str = "full",
    bc_phases: list[str] | None = None,
    fixed_zero_action_dims: list[int] | None = None,
    bc_blend_alpha: float = 1.0,
) -> dict:
    """Evaluate BC policy in robosuite environment.

    Args:
        model: Trained MLPBCPolicy.
        num_episodes: Number of episodes.
        device: Torch device.
        action_mean: Action normalization mean.
        action_std: Action normalization std.
        condition_on_phase: Whether model expects phase one-hot in obs.
        control_mode: "full" (BC all phases) or "hybrid" (BC selected phases only).
        bc_phases: Phases where BC controls (hybrid mode). Others use scripted.
        fixed_zero_action_dims: Dims to zero out in BC output.

    Returns:
        Summary dict with evaluation metrics.
    """
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import build_stacking_goal
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode

    is_hybrid = control_mode == "hybrid"
    bc_phase_ids = {PHASE_NAMES.index(p) for p in (bc_phases or [])} if is_hybrid else set(range(6))
    override_phases = bc_phase_ids
    if is_hybrid:
        scripted_phase_ids = set(range(6)) - bc_phase_ids
        print(f"Hybrid mode: BC phases={bc_phases} (ids={bc_phase_ids}), scripted phases (ids={scripted_phase_ids})")
    else:
        print(f"Full BC mode: BC controls all phases")

    cubes = ["cube1", "cube2"]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        return {"error": f"Planner failed: {plan_result.error}"}

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda",
                              cube_names=cubes, render_mode="none")
    horizon = env.env.horizon if hasattr(env.env, "horizon") else 1000
    model.eval()

    def policy_fn(obs, phase_id):
        """BC policy closure: adds phase one-hot if needed, denormalizes action."""
        if condition_on_phase:
            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[phase_id] = 1.0
            obs = np.concatenate([obs, one_hot])
        return _get_bc_action(model, obs, device, action_mean, action_std, fixed_zero_action_dims)

    episode_results = []
    for ep in range(num_episodes):
        result = run_episode(
            env, plan_result.actions, cubes, goal_predicates,
            policy_override=policy_fn,
            override_phases=override_phases,
            horizon=horizon,
            early_stop_on_goal=True,
            bc_blend_alpha=bc_blend_alpha,
        )
        episode_results.append(result)

    env.env.close()

    n_ep = max(num_episodes, 1)
    episode_successes = [r.get("goal_success", False) for r in episode_results]
    goal_successes = [r.get("semantic_goal_success", False) for r in episode_results]
    step_counts = [r.get("num_steps", 0) for r in episode_results]
    bc_steps_list = [r.get("bc_steps", 0) for r in episode_results]
    scripted_steps_list = [r.get("scripted_steps", 0) for r in episode_results]

    summary = {
        "num_episodes": num_episodes,
        "episode_success_rate": sum(1 for s in episode_successes if s) / n_ep,
        "goal_success_rate": sum(1 for s in goal_successes if s) / n_ep,
        "mean_steps": float(np.mean(step_counts)) if step_counts else 0,
        "p95_steps": float(np.percentile(step_counts, 95)) if step_counts else 0,
        "step_counts": step_counts,
        "control_mode": control_mode,
        "bc_phases": bc_phases,
        "condition_on_phase": condition_on_phase,
        "mean_bc_steps": float(np.mean(bc_steps_list)) if bc_steps_list else 0,
        "mean_scripted_steps": float(np.mean(scripted_steps_list)) if scripted_steps_list else 0,
    }
    return summary


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[MLPBCPolicy, dict]:
    """Load model checkpoint and return (model, metadata)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = {}

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata["obs_dim"] = checkpoint.get("obs_dim", OBS_DIM)
        metadata["action_dim"] = checkpoint.get("action_dim", ACTION_DIM)
        metadata["hidden_dims"] = checkpoint.get("hidden_dims", [256, 256])
        metadata["condition_on_phase"] = checkpoint.get("condition_on_phase", False)
        metadata["phase_dim"] = checkpoint.get("phase_dim", 0)
    else:
        state_dict = checkpoint
        # Infer dims from state_dict weights
        metadata["obs_dim"] = state_dict["net.0.weight"].shape[1]
        metadata["action_dim"] = state_dict["net.4.weight"].shape[0]
        hidden = [state_dict["net.0.weight"].shape[0]]
        if "net.2.weight" in state_dict:
            hidden.append(state_dict["net.2.weight"].shape[0])
        metadata["hidden_dims"] = hidden
        metadata["condition_on_phase"] = False
        metadata["phase_dim"] = 0

    model = MLPBCPolicy(
        obs_dim=metadata["obs_dim"],
        action_dim=metadata["action_dim"],
        hidden_dims=metadata["hidden_dims"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC model checkpoint .pt")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config (action_norm_stats.json)")
    # Hybrid mode
    parser.add_argument("--control_mode", type=str, default="full",
                        choices=["full", "hybrid"],
                        help="full=BC all phases, hybrid=BC selected phases")
    parser.add_argument("--bc_phases", type=str, nargs="+", default=None,
                        choices=PHASE_NAMES,
                        help="Phases for BC control in hybrid mode")
    parser.add_argument("--condition_on_phase", action="store_true", default=False,
                        help="Model expects phase one-hot in obs")
    parser.add_argument("--fixed_zero_action_dims", type=int, nargs="*", default=None,
                        help="Action dims to fix at zero")
    # Stage 7B-3: BC blend alpha
    parser.add_argument("--bc_blend_alpha", type=float, default=1.0,
                        help="Blend factor for BC: 1.0=pure BC, 0.0=pure scripted (default: 1.0)")
    # Legacy normalization args
    parser.add_argument("--action-mean", type=float, nargs="+", default=None)
    parser.add_argument("--action-std", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load model
    model, meta = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  obs_dim={meta['obs_dim']}, action_dim={meta['action_dim']}, "
          f"hidden={meta['hidden_dims']}, condition_on_phase={meta['condition_on_phase']}")

    # Resolve conditioning: CLI arg takes precedence over checkpoint metadata
    condition_on_phase = args.condition_on_phase or meta.get("condition_on_phase", False)
    if condition_on_phase:
        print(f"Phase conditioning enabled (phase_dim={meta.get('phase_dim', 6)})")

    # Load config for action normalization
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            if "action_mean" in cfg:
                action_mean = np.array(cfg["action_mean"], dtype=np.float32)
            if "action_std" in cfg:
                action_std = np.array(cfg["action_std"], dtype=np.float32)
            print(f"Loaded config from {config_path}")

    # Legacy CLI args override
    if args.action_mean is not None:
        action_mean = np.array(args.action_mean, dtype=np.float32)
    if args.action_std is not None:
        action_std = np.array(args.action_std, dtype=np.float32)

    if action_mean is not None:
        print(f"  action_mean: {action_mean}")
    if action_std is not None:
        print(f"  action_std:  {action_std}")

    # Hybrid mode validation
    if args.control_mode == "hybrid" and not args.bc_phases:
        parser.error("--bc_phases required when --control_mode=hybrid")

    summary = evaluate_policy(
        model, num_episodes=args.num_episodes, device=device,
        action_mean=action_mean, action_std=action_std,
        condition_on_phase=condition_on_phase,
        control_mode=args.control_mode,
        bc_phases=args.bc_phases,
        fixed_zero_action_dims=args.fixed_zero_action_dims,
        bc_blend_alpha=args.bc_blend_alpha,
    )

    print(json.dumps(summary, indent=2))

    if "error" not in summary:
        output_path = args.output or str(
            ensure_dir(PROJECT_ROOT / "generated" / "results" / "stage_7B-2") /
            f"bc_{args.control_mode}_eval.json"
        )
        Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
