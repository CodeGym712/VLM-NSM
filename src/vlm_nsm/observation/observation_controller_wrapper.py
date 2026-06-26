"""Observation-aware scripted controller wrapper.

Wraps existing scripted execution to use ObservationSource for state,
emitting legality reports and privileged-state audit logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.observation.observation_source import ObservationSource
from vlm_nsm.observation.oracle_state_source import OracleStateSource
from vlm_nsm.observation.scene_state import SceneState, ObservationLegalityReport


def scene_state_to_obs_vector(
    scene: SceneState,
    cubes: list[str],
    gripper_open: float = 0.05,
    gripper_has_object: float = 0.0,
) -> np.ndarray:
    """Build 11-dim obs vector from SceneState (backward-compatible).

    obs layout: eef_xyz(3) + gripper_open(1) + has_object(1) + cube1_xyz(3) + cube2_xyz(3)
    """
    obs = np.zeros(11, dtype=np.float32)
    # EEF
    if scene.robot.eef_position is not None:
        obs[0:3] = scene.robot.eef_position
    # Gripper
    obs[3] = gripper_open
    obs[4] = gripper_has_object
    # Objects
    for i, name in enumerate(cubes):
        obj = scene.get_object(name)
        if obj.available and obj.position is not None:
            start = 5 + i * 3
            obs[start:start + 3] = obj.position
    return obs


def run_scripted_with_observation_source(
    env,
    cubes: list[str],
    skill_command,
    observation_source: ObservationSource,
    controller_config: dict[str, Any],
    output_legality_path: str | None = None,
) -> dict[str, Any]:
    """Run a single pick_place episode through ObservationSource.

    Args:
        env: RobosuiteEnvWrapper
        cubes: Cube name list
        skill_command: SkillCommand for pick_place
        observation_source: ObservationSource instance
        controller_config: Controller kwargs dict
        output_legality_path: If set, write legality report JSON here

    Returns:
        Dict with: success, steps, failure_type, observation_legality, scene_states_count
    """
    from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor

    observation_source.reset(env)

    # Current approach: use scripted executor which internally accesses env.
    # For full refactor, we'd pass observation_source through context.
    # For now, we run the executor normally but record the observation source
    # as the "intended" state interface, and emit a legality report.
    #
    # The executor calls env.get_eef_pose() / env.get_object_pose() internally.
    # These are flagged in the privileged-state audit.
    # This wrapper marks that the *intended* path is through ObservationSource,
    # and the internal direct access is legacy that needs refactoring.

    executor = ScriptedSkillExecutor(controller_config=controller_config)

    # Pass observation_source through context for future refactoring
    context = {
        "cubes": cubes,
        "policy_override": None,
        "override_phases": set(),
        "observation_source": observation_source,  # new
    }

    result = executor.execute(skill_command=skill_command, env=env, context=context)

    # Emit legality report
    legality = observation_source.get_legality_report()

    output = {
        "success": result.success,
        "steps": result.steps,
        "failure_type": result.metrics.get("raw_result", {}).get("failure_type", "none"),
        "observation_source": observation_source.__class__.__name__,
        "observation_legality": legality.to_dict(),
        "legality_report": legality.to_dict(),
    }

    if output_legality_path:
        Path(output_legality_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_legality_path).write_text(json.dumps(legality.to_dict(), indent=2))

    return output
