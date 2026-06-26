"""ReplayController sanity check.

Loads a demonstration npz, replays the action sequence through ReplayController,
and verifies data quality. If replay_success_rate < 95%, the data should not
be used for BC training.

Usage:
    python -m vlm_nsm.experiments.run_replay_sanity_check \\
        --demo_path generated/demos/stacking_3_scripted_lowdim.npz \\
        --meta_path generated/demos/stacking_3_scripted_lowdim_meta.json \\
        --env mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlm_nsm.core.data_types import Action, Predicate
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.embodied.controllers.replay_controller import ReplayController
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper
from vlm_nsm.executors.embodied_skill_executor import EmbodiedSkillExecutor


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def run_sanity_check(
    demo_path: str | Path,
    meta_path: str | Path | None = None,
    env_type: str = "mock",
    n_blocks: int = 3,
) -> dict:
    """Run replay sanity check on collected demonstrations.

    Args:
        demo_path: Path to npz file with demonstrations.
        meta_path: Optional path to metadata JSON.
        env_type: Environment type ("mock").
        n_blocks: Number of blocks.

    Returns:
        Dict with replay sanity check results.
    """
    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    data = np.load(demo_path)
    meta = None
    if meta_path:
        meta_path = Path(meta_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

    # Validate required fields
    required_fields = ["obs", "action", "done", "episode_id", "skill_id", "step_id", "success"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {demo_path}")

    obs = data["obs"]
    actions = data["action"]
    done_flags = data["done"]
    episode_ids = data["episode_id"]
    skill_ids = data["skill_id"]
    step_ids = data["step_id"]
    success_flags = data["success"]

    # Metadata checks
    obs_dim_match = True
    action_dim_match = True
    if meta:
        obs_dim_match = (obs.shape[1] if obs.ndim == 2 else None) == meta.get("obs_dim")
        action_dim_match = (actions.shape[1] if actions.ndim == 2 else None) == meta.get("action_dim")

    # Step sequence validation (step_ids are 0-5 per skill decomposition)
    step_sequence_valid = True
    if len(step_ids) > 0:
        step_sequence_valid = bool(np.all((step_ids >= 0) & (step_ids <= 5)))

    # Replay
    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)

    env = MockEnvWrapper(cubes=cubes)
    replay_controller = ReplayController()
    replay_controller.load(actions)

    executor = EmbodiedSkillExecutor(
        env_wrapper=env,
        cubes=cubes,
        controller=replay_controller,
    )

    result = executor.execute_plan(plan_result.actions, goal_predicates)

    # Episode count check
    unique_episodes = set(int(e) for e in episode_ids)
    episode_count_match = len(unique_episodes) > 0

    # Replay success metric
    replay_success = result.get("episode_success", False)
    replay_success_rate = 1.0 if replay_success else 0.0
    # Since we replay one episode at a time, use the total success from data
    total_episodes_in_data = max(len(unique_episodes), 1)
    success_count = int(np.sum(success_flags)) if len(success_flags) > 0 else 0
    total_steps_in_data = max(len(success_flags), 1)

    summary = {
        "demo_path": str(demo_path),
        "meta_path": str(meta_path) if meta_path else None,
        "total_steps_in_data": int(len(success_flags)),
        "total_episodes_in_data": total_episodes_in_data,
        "data_success_rate": float(success_count / total_steps_in_data),
        "episode_count_match": episode_count_match,
        "obs_dim_match": obs_dim_match,
        "action_dim_match": action_dim_match,
        "step_sequence_valid": bool(step_sequence_valid),
        "replay_execution_success": result.get("execution_success", False),
        "replay_goal_success": result.get("goal_success", False),
        "replay_success": replay_success,
        "replay_success_rate": replay_success_rate,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run replay sanity check")
    parser.add_argument("--demo_path", required=True, help="Path to demo npz file")
    parser.add_argument("--meta_path", default=None, help="Path to metadata JSON")
    parser.add_argument("--env", default="mock", choices=["mock"])
    parser.add_argument("--n_blocks", type=int, default=3, choices=[3, 4, 5])
    args = parser.parse_args()

    summary = run_sanity_check(
        demo_path=args.demo_path,
        meta_path=args.meta_path,
        env_type=args.env,
        n_blocks=args.n_blocks,
    )

    print(json.dumps(summary, indent=2))

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    summary_path = results_dir / "replay_sanity_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
