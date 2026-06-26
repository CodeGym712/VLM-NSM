"""Analyze robosuite failure patterns across multiple episodes.

Categorizes failure types including timeout_before_goal vs timeout_after_goal,
place_failed, and grasp_failed. Tracks strict vs semantic success.

Usage:
    python -m vlm_nsm.experiments.analyze_robosuite_failures
    python -m vlm_nsm.experiments.analyze_robosuite_failures --num_episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def _try_import_robosuite():
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper  # noqa: F401
        return True
    except ImportError:
        return False


def analyze_failures(num_episodes: int = 10, early_stop_on_goal: bool = True) -> dict:
    """Run stacking evaluation and analyze failure patterns.

    Args:
        num_episodes: Number of episodes to run.
        early_stop_on_goal: Whether to early-stop when goal achieved.

    Returns:
        Dict with failure analysis results including timeout subtypes.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
    from vlm_nsm.core.planner_adapter import PlannerAdapter
    from vlm_nsm.core.data_types import Predicate

    cubes = ["cube1", "cube2"]
    target_stack = cubes[::-1]

    def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
        return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]

    goal_predicates = build_stacking_goal(target_stack)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        return {"error": f"Planner failed: {plan_result.error}"}

    env = RobosuiteEnvWrapper(env_name="Stack", robots="Panda", cube_names=cubes, render_mode="none")
    horizon = env.env.horizon if hasattr(env.env, "horizon") else 1000

    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode

    per_episode: list[dict] = []
    failure_type_counts: dict[str, int] = {}
    failure_stage_counts: dict[str, int] = {}
    total_episode_success = 0
    total_grasp_failed = 0
    total_place_failed = 0
    total_timeout = 0
    total_timeout_before = 0
    total_timeout_after = 0
    total_strict_success = 0
    total_semantic_success = 0
    all_step_counts: list[int] = []

    for ep in range(num_episodes):
        result = _run_robosuite_episode(
            env, plan_result.actions, cubes, goal_predicates,
            horizon=horizon, early_stop_on_goal=early_stop_on_goal,
        )

        # Normalize failure type
        ftype = result.get("failure_type", "unknown") or "success"
        failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1

        fd = result.get("failure_stage_distribution", {})
        for stage, count in fd.items():
            failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + count

        if result.get("episode_success", False):
            total_episode_success += 1
        if result.get("strict_physical_success", False):
            total_strict_success += 1
        if result.get("semantic_goal_success", False):
            total_semantic_success += 1

        if ftype in ("grasp_failed",):
            total_grasp_failed += 1
        elif ftype == "place_failed":
            total_place_failed += 1
        elif ftype in ("timeout", "horizon", "timeout_before_goal", "timeout_after_goal"):
            total_timeout += 1

        tt = result.get("timeout_type")
        if tt == "timeout_before_goal":
            total_timeout_before += 1
        elif tt == "timeout_after_goal":
            total_timeout_after += 1

        all_step_counts.append(result.get("num_steps", 0))

        per_episode.append({
            "episode": ep + 1,
            "execution_success": result.get("execution_success", False),
            "goal_success": result.get("goal_success", False),
            "strict_physical_success": result.get("strict_physical_success", False),
            "semantic_goal_success": result.get("semantic_goal_success", False),
            "failure_type": ftype,
            "timeout_type": result.get("timeout_type"),
            "failure_stage_distribution": fd,
            "step_success_rate": result.get("step_success_rate", 0.0),
            "skill_success_rate": result.get("skill_success_rate", 0.0),
            "predicate_match_rate": result.get("predicate_match_rate", 0.0),
            "num_steps": result.get("num_steps", 0),
            "early_stopped": result.get("early_stopped", False),
            "goal_achieved_at_step": result.get("goal_achieved_at_step"),
        })

    env.env.close()

    n = max(num_episodes, 1)

    analysis: dict = {
        "num_episodes": num_episodes,
        "overall_episode_success_rate": total_episode_success / n,
        "strict_success_rate": total_strict_success / n,
        "semantic_success_rate": total_semantic_success / n,
        "failure_type_rates": {k: v / n for k, v in failure_type_counts.items()},
        "failure_type_counts": failure_type_counts,
        "failure_stage_counts": failure_stage_counts,
        "grasp_failure_rate": total_grasp_failed / n,
        "place_failure_rate": total_place_failed / n,
        "timeout_rate": total_timeout / n,
        "timeout_before_goal_rate": total_timeout_before / n,
        "timeout_after_goal_rate": total_timeout_after / n,
        "mean_step_success_rate": sum(e["step_success_rate"] for e in per_episode) / n,
        "mean_skill_success_rate": sum(e["skill_success_rate"] for e in per_episode) / n,
        "mean_predicate_match_rate": sum(e["predicate_match_rate"] for e in per_episode) / n,
        "mean_steps": float(np.mean(all_step_counts)) if all_step_counts else 0.0,
        "p95_steps": float(np.percentile(all_step_counts, 95)) if all_step_counts else 0.0,
        "per_episode": per_episode,
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze robosuite failure patterns")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stop on goal achievement")
    args = parser.parse_args()

    if not _try_import_robosuite():
        print("ERROR: Robosuite not available.", file=sys.stderr)
        sys.exit(1)

    print(f"Running failure analysis: {args.num_episodes} episodes...")
    analysis = analyze_failures(num_episodes=args.num_episodes,
                                 early_stop_on_goal=not args.no_early_stop)

    if "error" in analysis:
        print(f"ERROR: {analysis['error']}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Episode success rate:      {analysis['overall_episode_success_rate'] * 100:.1f}%")
    print(f"Strict success rate:       {analysis['strict_success_rate'] * 100:.1f}%")
    print(f"Semantic success rate:     {analysis['semantic_success_rate'] * 100:.1f}%")
    print(f"Grasp failure rate:        {analysis['grasp_failure_rate'] * 100:.1f}%")
    print(f"Place failure rate:        {analysis['place_failure_rate'] * 100:.1f}%")
    print(f"Timeout rate (total):      {analysis['timeout_rate'] * 100:.1f}%")
    print(f"  timeout_before_goal:     {analysis['timeout_before_goal_rate'] * 100:.1f}%")
    print(f"  timeout_after_goal:      {analysis['timeout_after_goal_rate'] * 100:.1f}%")
    print(f"Mean steps:                {analysis['mean_steps']:.0f}")
    print(f"P95 steps:                 {analysis['p95_steps']:.0f}")
    print(f"\nFailure type distribution:")
    for ftype, rate in sorted(analysis["failure_type_rates"].items()):
        print(f"  {ftype}: {rate * 100:.1f}% ({analysis['failure_type_counts'].get(ftype, 0)} episodes)")
    print(f"\nMean step success rate:   {analysis['mean_step_success_rate'] * 100:.1f}%")
    print(f"Mean skill success rate: {analysis['mean_skill_success_rate'] * 100:.1f}%")
    print(f"Mean predicate match:    {analysis['mean_predicate_match_rate'] * 100:.1f}%")

    # Save
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    output_path = results_dir / "robosuite_failure_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
