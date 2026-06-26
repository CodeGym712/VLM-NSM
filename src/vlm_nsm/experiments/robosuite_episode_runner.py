"""Shared robosuite episode runner with policy override support.

Wraps ``_run_robosuite_episode`` and passes ``policy_override`` /
``override_phases`` directly to the internal ``_RobosuiteRecordingWrapper``
so that BC overridden actions are recorded correctly.

Usage:
    from vlm_nsm.experiments.robosuite_episode_runner import run_episode

    # Fully scripted (no override)
    result = run_episode(env, actions, cubes, goal_predicates)

    # Hybrid: BC for reach phases only
    def bc_policy(obs, phase_id):
        return model(obs_tensor).numpy()

    result = run_episode(env, actions, cubes, goal_predicates,
                         policy_override=bc_policy,
                         override_phases={0, 3})
"""

from __future__ import annotations

from typing import Callable

from vlm_nsm.experiments.run_embodied_stacking_robosuite import _run_robosuite_episode


_PHASE_TO_SKILL_ID = {
    "reach_pick": 0, "grasp": 1, "lift": 2,
    "reach_place": 3, "release": 4, "retreat": 5,
}


def run_episode(
    env,
    actions: list,
    cubes: list[str],
    goal_predicates,
    *,
    policy_override: Callable | None = None,
    override_phases: set[int] | None = None,
    horizon: int = 1000,
    early_stop_on_goal: bool = True,
    recorder=None,
    **kwargs,
) -> dict:
    """Run a robosuite episode with optional BC policy override.

    Policy override is applied inside ``_RobosuiteRecordingWrapper`` so
    that overridden actions are recorded, not the original scripted ones.

    Args:
        env: RobosuiteEnvWrapper instance.
        actions: List of planner actions.
        cubes: Cube name list.
        goal_predicates: Goal predicates for success checking.
        policy_override: Callable(obs, phase_id) -> action. If None, all
            phases run the scripted controller.
        override_phases: Set of phase IDs to override with policy.
        horizon: Episode horizon.
        early_stop_on_goal: Whether to stop when goal achieved.
        recorder: Optional DemonstrationRecorder for recording.
        **kwargs: Additional kwargs forwarded to _run_robosuite_episode.

    Returns:
        Result dict from _run_robosuite_episode with additional keys:
        - bc_steps: Number of steps controlled by BC
        - scripted_steps: Number of steps controlled by scripted
        - override_phases: List of overridden phase IDs
    """
    result = _run_robosuite_episode(
        env, actions, cubes, goal_predicates,
        horizon=horizon,
        early_stop_on_goal=early_stop_on_goal,
        recorder=recorder,
        policy_override=policy_override,
        override_phases=override_phases or set(),
        **kwargs,
    )

    # Ensure result has bc_steps/scripted_steps even if recorder was not used
    result.setdefault("bc_steps", 0)
    result.setdefault("scripted_steps", 0)
    result["override_phases"] = sorted(override_phases) if override_phases else []
    return result
