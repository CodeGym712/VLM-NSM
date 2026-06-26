"""Run mock embodied stacking evaluation.

Executes the full embodied pipeline with ScriptedController + MockEnvWrapper
and reports Experiment C metrics including Reality Gap.

Does NOT save demonstrations — purely evaluation.

Usage:
    python -m vlm_nsm.experiments.run_embodied_stacking_mock \\
        --n_blocks 3 --num_episodes 20 --controller scripted

    # With noise injection:
    python -m vlm_nsm.experiments.run_embodied_stacking_mock \\
        --n_blocks 3 --num_episodes 20 --controller scripted \\
        --pose_noise_std 0.01 --grasp_fail_prob 0.1 --place_noise_std 0.01 --seed 0
"""

from __future__ import annotations

import argparse
import json

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.metrics import reality_gap
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.embodied.controllers.scripted_controller import ScriptedController
from vlm_nsm.executors.embodied_skill_executor import EmbodiedSkillExecutor
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def run_mock_eval(
    n_blocks: int = 3,
    num_episodes: int = 20,
    controller_type: str = "scripted",
    obs_dim: int = 16,
    action_dim: int = 4,
    pose_noise_std: float = 0.0,
    grasp_fail_prob: float = 0.0,
    place_noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run mock embodied stacking evaluation.

    Args:
        n_blocks: Number of blocks (3).
        num_episodes: Number of episodes.
        controller_type: Controller type ("scripted").
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        pose_noise_std: Std of Gaussian noise on poses.
        grasp_fail_prob: Probability of grasp failure.
        place_noise_std: Std of Gaussian noise on place positions.
        seed: Random seed.

    Returns:
        Dict with Experiment C metrics.
    """
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    # Planner
    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        raise RuntimeError(f"Planner failed: {plan_result.error}")

    # Env + controller
    env = MockEnvWrapper(cubes=cubes)
    controller = ScriptedController(action_dim=action_dim)

    executor = EmbodiedSkillExecutor(
        env_wrapper=env,
        cubes=cubes,
        controller=controller,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pose_noise_std=pose_noise_std,
        grasp_fail_prob=grasp_fail_prob,
        place_noise_std=place_noise_std,
        seed=seed,
    )

    # Aggregate metrics across episodes
    step_successes: list[float] = []
    skill_successes: list[float] = []
    episode_successes: list[bool] = []
    goal_successes: list[bool] = []
    failure_stages: dict[str, int] = {}
    all_predicate_rates: list[float] = []
    symbolic_success_count = 0
    physical_success_count = 0

    for _ in range(num_episodes):
        result = executor.execute_plan(plan_result.actions, goal_predicates)
        step_successes.append(result.get("step_success_rate", 0.0))
        skill_successes.append(result.get("skill_success_rate", 0.0))
        episode_successes.append(result.get("episode_success", False))
        goal_successes.append(result.get("goal_success", False))
        all_predicate_rates.append(result.get("predicate_match_rate", 0.0))

        fd = result.get("failure_stage_distribution", {})
        for stage, count in fd.items():
            failure_stages[stage] = failure_stages.get(stage, 0) + count

        if result.get("goal_success"):
            symbolic_success_count += 1
            physical_success_count += 1

    # Aggregate
    num_ep = max(num_episodes, 1)
    episode_success_count = sum(1 for s in episode_successes if s)

    rg = reality_gap(symbolic_success_count, physical_success_count)

    summary = {
        "task": "stacking",
        "n_blocks": n_blocks,
        "num_episodes": num_episodes,
        "controller_type": controller_type,
        "noise_config": {
            "pose_noise_std": pose_noise_std,
            "grasp_fail_prob": grasp_fail_prob,
            "place_noise_std": place_noise_std,
        },
        "step_success_rate": sum(step_successes) / num_ep if step_successes else 0.0,
        "skill_success_rate": sum(skill_successes) / num_ep if skill_successes else 0.0,
        "episode_success_rate": episode_success_count / num_ep,
        "goal_success_rate": sum(1 for s in goal_successes if s) / num_ep if goal_successes else 0.0,
        "failure_stage_distribution": failure_stages,
        "predicate_match_rate": sum(all_predicate_rates) / num_ep if all_predicate_rates else 0.0,
        "symbolic_success_count": symbolic_success_count,
        "physical_success_count": physical_success_count,
        "symbolic_success_rate": symbolic_success_count / num_ep,
        "physical_success_rate": physical_success_count / num_ep,
        "reality_gap": rg,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run mock embodied stacking evaluation")
    parser.add_argument("--n_blocks", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--controller", default="scripted", choices=["scripted", "replay"])
    parser.add_argument("--obs_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=4)
    parser.add_argument("--pose_noise_std", type=float, default=0.0)
    parser.add_argument("--grasp_fail_prob", type=float, default=0.0)
    parser.add_argument("--place_noise_std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_mock_eval(
        n_blocks=args.n_blocks,
        num_episodes=args.num_episodes,
        controller_type=args.controller,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        pose_noise_std=args.pose_noise_std,
        grasp_fail_prob=args.grasp_fail_prob,
        place_noise_std=args.place_noise_std,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))

    # Determine output filename based on noise
    has_noise = any([args.pose_noise_std > 0, args.grasp_fail_prob > 0, args.place_noise_std > 0])
    filename = "embodied_stacking_mock_summary_noise.json" if has_noise else "embodied_stacking_mock_summary.json"

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    summary_path = results_dir / filename
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
