"""Collect stacking demonstrations using ScriptedController + EmbodiedSkillExecutor.

Usage:
    python -m vlm_nsm.experiments.collect_stacking_demos \\
        --n_blocks 3 --num_episodes 20 --env mock \\
        --controller scripted \\
        --output generated/demos/stacking_3_scripted_lowdim.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
from vlm_nsm.embodied.controllers.scripted_controller import ScriptedController
from vlm_nsm.executors.embodied_skill_executor import EmbodiedSkillExecutor
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper


_STEP_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    """Build stacking goal predicates from a target stack list (top-to-bottom)."""
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def collect_demos(
    n_blocks: int = 3,
    num_episodes: int = 20,
    env_type: str = "mock",
    controller_type: str = "scripted",
    output_path: str | None = None,
    obs_dim: int = 16,
    action_dim: int = 4,
    save_full_obs: bool = False,
    pose_noise_std: float = 0.0,
    grasp_fail_prob: float = 0.0,
    place_noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    """Collect demonstrations and return summary.

    Args:
        n_blocks: Number of blocks (3, 4, or 5).
        num_episodes: Number of episodes to collect.
        env_type: Environment type ("mock").
        controller_type: Controller type ("scripted" or "replay").
        output_path: Path for output npz file.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        save_full_obs: Whether to save full (unfiltered) observations.
        pose_noise_std: Std of Gaussian noise on poses.
        grasp_fail_prob: Probability of grasp failure.
        place_noise_std: Std of Gaussian noise on place positions.
        seed: Random seed.

    Returns:
        Summary dict with collection stats.
    """
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = cubes[::-1]  # bottom-to-top: [cubeN, ..., cube1]
    goal_predicates = build_stacking_goal(target_stack)

    # Planner
    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        raise RuntimeError(f"Planner failed: {plan_result.error}")

    # Env + controller + recorder
    env = MockEnvWrapper(cubes=cubes)

    noise_config = {
        "pose_noise_std": pose_noise_std,
        "grasp_fail_prob": grasp_fail_prob,
        "place_noise_std": place_noise_std,
    }

    # Full obs dim: n_cubes * 7 + table * 7 + gripper * 7 + 1 scalar
    full_obs_dim = n_blocks * 7 + 7 + 7 + 1

    if output_path:
        output_dir = str(Path(output_path).parent)
        recorder = DemonstrationRecorder(
            output_dir=output_dir,
            obs_dim=obs_dim,
            action_dim=action_dim,
            full_obs_dim=full_obs_dim if save_full_obs else None,
        )
    else:
        recorder = None

    executor = EmbodiedSkillExecutor(
        env_wrapper=env,
        cubes=cubes,
        controller=ScriptedController(action_dim=action_dim),
        recorder=recorder,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pose_noise_std=pose_noise_std,
        grasp_fail_prob=grasp_fail_prob,
        place_noise_std=place_noise_std,
        seed=seed,
    )

    episode_successes = []
    total_steps_list = []
    failure_stages: dict[str, int] = {}

    for ep in range(num_episodes):
        result = executor.execute_plan(plan_result.actions, goal_predicates)
        episode_successes.append(result.get("episode_success", False))
        total_steps_list.append(result.get("num_steps", 0))
        fd = result.get("failure_stage_distribution", {})
        for stage, count in fd.items():
            failure_stages[stage] = failure_stages.get(stage, 0) + count

        if recorder:
            recorder.end_episode()

    # Save npz and metadata
    npz_path = None
    meta_path = None
    if recorder and output_path:
        step_names = _STEP_NAMES
        npz_path, meta_path = recorder.save(
            filename=Path(output_path).name,
            task="stacking",
            n_blocks=n_blocks,
            env_type=env_type,
            controller_type=controller_type,
            step_names=step_names,
            observation_type="symbol_conditioned",
            noise_config=noise_config,
        )

    # Summary
    num_success = sum(1 for s in episode_successes if s)
    avg_steps = sum(total_steps_list) / max(len(total_steps_list), 1)

    summary = {
        "num_episodes": num_episodes,
        "num_success": num_success,
        "episode_success_rate": num_success / max(num_episodes, 1),
        "avg_episode_length": float(avg_steps),
        "avg_steps_per_skill": float(avg_steps) / max(n_blocks - 1, 1) if n_blocks > 1 else 0.0,
        "failure_stage_distribution": failure_stages,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "npz_path": str(npz_path) if npz_path else None,
        "meta_path": str(meta_path) if meta_path else None,
        "noise_config": noise_config,
        "full_obs_saved": save_full_obs,
        "full_obs_dim": full_obs_dim if save_full_obs else None,
        "filtered_obs_dim": obs_dim,
        "reduction_ratio": 1.0 - (obs_dim / full_obs_dim) if save_full_obs and full_obs_dim > 0 else None,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Collect stacking demonstrations")
    parser.add_argument("--n_blocks", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--env", default="mock", choices=["mock"])
    parser.add_argument("--controller", default="scripted", choices=["scripted", "replay"])
    parser.add_argument("--output", default=None, help="Output npz path")
    parser.add_argument("--obs_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--save_full_obs", action="store_true", default=False)
    parser.add_argument("--pose_noise_std", type=float, default=0.0)
    parser.add_argument("--grasp_fail_prob", type=float, default=0.0)
    parser.add_argument("--place_noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = collect_demos(
        n_blocks=args.n_blocks,
        num_episodes=args.num_episodes,
        env_type=args.env,
        controller_type=args.controller,
        output_path=args.output,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        save_full_obs=args.save_full_obs,
        pose_noise_std=args.pose_noise_std,
        grasp_fail_prob=args.grasp_fail_prob,
        place_noise_std=args.place_noise_std,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))

    # Save summary
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    summary_path = results_dir / "collect_stacking_demos_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
