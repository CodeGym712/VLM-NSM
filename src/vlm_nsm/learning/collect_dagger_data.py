"""Collect DAgger data: on-policy BC rollout with oracle (scripted) action targets.

For each BC-controlled step during episode rollout, records:
  - filtered_obs: observation at the BC-visited state
  - oracle_scripted_action: what the proportional controller would output
  - phase_id, episode_id, success

The recorded data pairs (state, oracle_action) can be used to train a
DAgger reach BC model that addresses covariate shift.

Usage:
    PYTHONUNBUFFERED=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \\
    python3 -m vlm_nsm.learning.collect_dagger_data \\
        --checkpoint generated/results/stage_7B-3/bc_main/best.pt \\
        --config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \\
        --num_episodes 30 --seed 0 --condition_on_phase \\
        --bc_phases reach_pick reach_place --fixed_zero_action_dims 3 4 5 \\
        --output_dir generated/demos/stage_7B-4/dagger_round1_reach
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
ACTION_DIM_NAMES = ["dx", "dy", "dz", "dOri0", "dOri1", "dOri2", "gripper"]
N_PHASES = 6

# Proportional controller defaults
KP_DEFAULT = 0.05
MAX_DELTA_DEFAULT = 0.02
OSC_POSE_MAX_DELTA = 0.05
LIFT_HEIGHT = 0.05
REACH_CLEARANCE = 0.10


def _norm_action_scripted(dx: float, dy: float, dz: float, gripper: float) -> np.ndarray:
    """Simulate _norm_action from run_embodied_stacking_robosuite.py."""
    return np.array([
        np.clip(dx / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dy / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dz / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        0.0, 0.0, 0.0,
        gripper,
    ], dtype=np.float32)


def _estimate_oracle_action(obs: np.ndarray, phase_id: int, kp: float, max_delta: float) -> np.ndarray:
    """Estimate proportional controller action for given obs and phase.

    Args:
        obs: 11-dim observation [eef_xyz, gripper, has_object, cube1_xyz, cube2_xyz]
        phase_id: 0-5 phase index
        kp: Proportional gain
        max_delta: Max position delta per step (physical)

    Returns:
        7-dim normalized action vector.
    """
    eef = obs[0:3]
    cube1 = obs[5:8]
    cube2 = obs[8:11]
    GRIPPER_OPEN = -1.0
    GRIPPER_CLOSE = 1.0

    if phase_id == 0:  # reach_pick
        target = cube1 + np.array([0.0, 0.0, REACH_CLEARANCE])
        gripper = GRIPPER_OPEN
    elif phase_id == 1:  # grasp
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE], dtype=np.float32)
    elif phase_id == 2:  # lift
        target = cube1 + np.array([0.0, 0.0, REACH_CLEARANCE + LIFT_HEIGHT])
        gripper = GRIPPER_CLOSE
    elif phase_id == 3:  # reach_place
        target = cube2 + np.array([0.0, 0.0, REACH_CLEARANCE])
        gripper = GRIPPER_CLOSE
    elif phase_id == 4:  # release
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN], dtype=np.float32)
    elif phase_id == 5:  # retreat
        target = eef + np.array([0.0, 0.0, 0.15])
        gripper = GRIPPER_OPEN
    else:
        return np.zeros(7, dtype=np.float32)

    dx = np.clip(kp * (target[0] - eef[0]), -max_delta, max_delta)
    dy = np.clip(kp * (target[1] - eef[1]), -max_delta, max_delta)
    dz = np.clip(kp * (target[2] - eef[2]), -max_delta, max_delta)
    return _norm_action_scripted(dx, dy, dz, gripper)


def _get_bc_action(
    model: MLPBCPolicy,
    obs: np.ndarray,
    device: str,
    action_mean: np.ndarray | None,
    action_std: np.ndarray | None,
    fixed_zero_dims: list[int] | None = None,
) -> np.ndarray:
    """Get denormalized BC action (same as eval_bc._get_bc_action)."""
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
def collect_dagger_episodes(
    model: MLPBCPolicy,
    num_episodes: int = 20,
    device: str = "cpu",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    condition_on_phase: bool = False,
    bc_phases: list[str] | None = None,
    fixed_zero_action_dims: list[int] | None = None,
    kp: float = KP_DEFAULT,
    max_delta: float = MAX_DELTA_DEFAULT,
    config: dict | None = None,
) -> tuple[list[dict], dict]:
    """Collect DAgger data by running BC policy and recording oracle actions.

    For each BC-controlled step, records (obs, oracle_scripted_action, phase_id).

    Args:
        model: Trained MLPBCPolicy.
        num_episodes: Number of episodes to collect.
        device: Torch device.
        action_mean: Action normalization mean.
        action_std: Action normalization std.
        condition_on_phase: Whether model expects phase one-hot in obs.
        bc_phases: List of phase names for BC control.
        fixed_zero_action_dims: Dims to zero out in BC output.
        kp: Proportional gain for oracle action estimation.
        max_delta: Max position delta for oracle action estimation.
        config: Execution config overrides (kp, max_delta, etc.).

    Returns:
        Tuple of (dagger_data_list, eval_summary).
    """
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import build_stacking_goal
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode

    bc_phase_ids = {PHASE_NAMES.index(p) for p in (bc_phases or [])}
    model.eval()

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

    # Build kwargs from config
    ep_kwargs = dict(
        horizon=horizon,
        early_stop_on_goal=True,
    )
    if config:
        config_keys = [
            "kp", "max_delta", "reach_threshold", "lift_height",
            "grasp_close_steps", "release_open_steps",
            "use_place_state_machine", "horizon_aware", "place_retry",
            "skip_retreat_if_goal_satisfied", "terminate_after_safe_release",
            "grasp_max_retries", "place_max_retries", "place_mode",
            "holding_check_enabled", "close_steps_min", "close_steps_max",
            "close_stabilize_steps", "lift_early_exit_enabled",
            "release_steps_min", "release_steps_max", "post_release_stable_steps",
            "close_mode", "close_steps_fast", "fallback_extend_close_if_not_holding",
        ]
        for k in config_keys:
            if k in config:
                ep_kwargs[k] = config[k]

    dagger_data: list[dict] = []
    episode_results = []

    def policy_fn(obs, phase_id):
        """BC policy closure: records oracle + BC actions, returns BC action."""
        # Estimate oracle (scripted) action at this BC-visited state
        oracle_action = _estimate_oracle_action(obs, phase_id, kp, max_delta)
        # Compute BC action (for residual computation later)
        if condition_on_phase:
            one_hot = np.zeros(N_PHASES, dtype=np.float32)
            one_hot[phase_id] = 1.0
            obs_with_phase = np.concatenate([obs, one_hot])
        else:
            obs_with_phase = obs
        bc_action = _get_bc_action(model, obs_with_phase, device, action_mean, action_std, fixed_zero_action_dims)

        dagger_data.append({
            "filtered_obs": obs.astype(np.float32).copy(),
            "oracle_scripted_action": oracle_action.astype(np.float32).copy(),
            "bc_action": bc_action.astype(np.float32).copy(),
            "phase_id": np.int64(phase_id),
        })
        return bc_action

    for ep in range(num_episodes):
        result = run_episode(
            env, plan_result.actions, cubes, goal_predicates,
            policy_override=policy_fn,
            override_phases=bc_phase_ids,
            **ep_kwargs,
        )
        episode_results.append(result)

        # Tag recorded data with episode_id and ep_success
        ep_success = result.get("goal_success", False) or result.get("semantic_goal_success", False)
        n_new = sum(1 for d in dagger_data if "episode_id" not in d)
        for i in range(n_new):
            dagger_data[-(n_new - i)]["episode_id"] = ep
            dagger_data[-(n_new - i)]["episode_success"] = ep_success

        bc_steps = result.get("bc_steps", 0)
        goal_ok = result.get("goal_success", False)
        print(f"  Episode {ep + 1}/{num_episodes}: success={goal_ok}, "
              f"steps={result.get('num_steps', 0)}, bc_steps={bc_steps}, "
              f"dagger_samples={n_new}")

    env.env.close()

    n_ep = max(len(episode_results), 1)
    eval_summary = {
        "num_episodes": num_episodes,
        "total_dagger_samples": len(dagger_data),
        "episode_successes": sum(1 for r in episode_results if r.get("goal_success", False)),
        "mean_steps": float(np.mean([r.get("num_steps", 0) for r in episode_results])),
        "mean_bc_steps": float(np.mean([r.get("bc_steps", 0) for r in episode_results])),
    }
    print(f"\nCollection complete: {len(dagger_data)} DAgger samples across {num_episodes} episodes")
    print(f"  Episode success rate: {eval_summary['episode_successes']}/{num_episodes}")

    return dagger_data, eval_summary


def save_dagger_data(dagger_data: list[dict], output_path: Path):
    """Save DAgger data as npz arrays compatible with RobosuiteDemoDataset.

    Produces a single npz file with keys:
      - filtered_obs: (T, 11) float32
      - oracle_scripted_action: (T, 7) float32
      - bc_action: (T, 7) float32 — BC model's action at this state
      - phase_id: (T,) int64
      - episode_id: (T,) int32
      - episode_success: (T,) bool
    """
    if not dagger_data:
        print("Warning: No DAgger data to save.")
        return

    T = len(dagger_data)
    obs_arr = np.stack([d["filtered_obs"] for d in dagger_data]).astype(np.float32)
    oracle_arr = np.stack([d["oracle_scripted_action"] for d in dagger_data]).astype(np.float32)
    bc_arr = np.stack([d["bc_action"] for d in dagger_data]).astype(np.float32) if "bc_action" in dagger_data[0] else np.zeros((T, 7), dtype=np.float32)
    phase_arr = np.array([d["phase_id"] for d in dagger_data], dtype=np.int64)
    ep_id_arr = np.array([d.get("episode_id", 0) for d in dagger_data], dtype=np.int32)
    success_arr = np.array([d.get("episode_success", False) for d in dagger_data], dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        filtered_obs=obs_arr,
        oracle_scripted_action=oracle_arr,
        bc_action=bc_arr,
        phase_id=phase_arr,
        episode_id=ep_id_arr,
        episode_success=success_arr,
    )
    print(f"DAgger data saved to {output_path} ({T} samples, {len(set(ep_id_arr))} episodes)")

    # Quick stats per phase
    phase_counts = np.bincount(phase_arr, minlength=N_PHASES)
    print("  Per-phase counts:")
    for pid in range(N_PHASES):
        if phase_counts[pid] > 0:
            print(f"    {PHASE_NAMES[pid]}: {phase_counts[pid]}")


def main():
    parser = argparse.ArgumentParser(description="Collect DAgger data from BC rollout")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC model checkpoint .pt")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to execution config JSON")
    # BC params
    parser.add_argument("--condition_on_phase", action="store_true", default=False)
    parser.add_argument("--bc_phases", type=str, nargs="+", default=["reach_pick", "reach_place"],
                        choices=PHASE_NAMES,
                        help="Phases for BC control (default: reach_pick reach_place)")
    parser.add_argument("--fixed_zero_action_dims", type=int, nargs="*", default=None)
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for DAgger npz file")
    # Oracle controller params
    parser.add_argument("--kp", type=float, default=KP_DEFAULT)
    parser.add_argument("--max_delta", type=float, default=MAX_DELTA_DEFAULT)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load checkpoint (handle state_dict-only or full checkpoint)
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

    # Resolve conditioning
    condition_on_phase = args.condition_on_phase or ckpt_cond
    if not condition_on_phase and obs_dim > 11:
        condition_on_phase = True
        print(f"  Inferred condition_on_phase=True from obs_dim={obs_dim}")

    model = MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, hidden={hidden_dims}, "
          f"condition_on_phase={condition_on_phase}")

    # Load action norm stats from checkpoint directory
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None
    ckpt_dir = Path(args.checkpoint).parent
    norm_path = ckpt_dir.parent / "action_norm_stats.json"  # stage results dir
    if not norm_path.exists():
        norm_path = Path(PROJECT_ROOT) / "generated/results/stage_7B-1/action_norm_stats.json"
    if norm_path.exists():
        cfg = json.loads(norm_path.read_text())
        if "action_mean" in cfg:
            action_mean = np.array(cfg["action_mean"], dtype=np.float32)
        if "action_std" in cfg:
            action_std = np.array(cfg["action_std"], dtype=np.float32)
        print(f"Loaded action norm from {norm_path}")
        print(f"  action_mean: {action_mean}")
        print(f"  action_std:  {action_std}")

    # Load execution config for oracle action estimation
    config: dict | None = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            config = json.loads(config_path.read_text())
            # Use config kp/max_delta if provided (CLI overrides config)
            if args.kp == KP_DEFAULT and "kp" in config:
                args.kp = config["kp"]
            if args.max_delta == MAX_DELTA_DEFAULT and "max_delta" in config:
                args.max_delta = config["max_delta"]
            print(f"Loaded execution config from {config_path}")

    print(f"Oracle controller: kp={args.kp}, max_delta={args.max_delta}")
    print(f"BC phases: {args.bc_phases}")

    # Collect DAgger data
    t_start = time.time()
    dagger_data, summary = collect_dagger_episodes(
        model, num_episodes=args.num_episodes, device=device,
        action_mean=action_mean, action_std=action_std,
        condition_on_phase=condition_on_phase,
        bc_phases=args.bc_phases,
        fixed_zero_action_dims=args.fixed_zero_action_dims,
        kp=args.kp, max_delta=args.max_delta,
        config=config,
    )
    elapsed = time.time() - t_start
    print(f"Collection took {elapsed:.1f}s ({elapsed / max(args.num_episodes, 1):.1f}s/ep)")

    # Save
    output_dir = ensure_dir(Path(args.output_dir))
    output_path = output_dir / "trajectories_dagger_round1_reach.npz"
    save_dagger_data(dagger_data, output_path)

    # Save summary
    summary["elapsed_seconds"] = elapsed
    summary_path = output_dir / "dagger_collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
