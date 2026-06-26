"""Scripted skill executor adapter.

This module wraps the existing Robosuite scripted/hybrid episode runner behind
the new skill-level interface. It does not change controller parameters,
checkpoint paths, or physical success semantics.
"""

from __future__ import annotations

from typing import Any

from vlm_nsm.core.data_types import Action
from vlm_nsm.skills.skill_types import SkillCommand, SkillExecutionResult


class ScriptedSkillExecutor:
    """Execute skill commands with the existing scripted/hybrid controller."""

    def __init__(self, controller_config: dict[str, Any] | None = None):
        self.controller_config = dict(controller_config or {})

    def execute(
        self,
        skill_command: SkillCommand,
        env: Any,
        context: dict[str, Any] | None = None,
    ) -> SkillExecutionResult:
        """Execute one skill command.

        The first supported command is ``pick_place``. Runtime dependencies on
        Robosuite are kept inside this method so unit tests can use mocks.
        """
        context = dict(context or {})
        if skill_command.skill_name != "pick_place":
            raise NotImplementedError(skill_command.skill_name)
        return self._execute_pick_place(skill_command, env, context)

    def _execute_pick_place(
        self,
        command: SkillCommand,
        env: Any,
        context: dict[str, Any],
    ) -> SkillExecutionResult:
        target = command.target_object_id
        if not target:
            raise ValueError("pick_place requires target_object_id")

        source = command.source or "table"
        action = Action("move", [command.object_id, source, target])
        cubes = context.get("cubes") or _infer_cubes(command)
        goal_predicates = context.get("goal_predicates") or [["on", command.object_id, target]]

        episode_kwargs = dict(self.controller_config)
        episode_kwargs.update(context.get("episode_kwargs") or {})

        from vlm_nsm.experiments.robosuite_episode_runner import run_episode

        result = run_episode(
            env=env,
            actions=[action],
            cubes=cubes,
            goal_predicates=goal_predicates,
            policy_override=context.get("policy_override"),
            override_phases=context.get("override_phases"),
            recorder=context.get("recorder"),
            **episode_kwargs,
        )

        success = bool(result.get("goal_success", result.get("success", False)))
        steps = int(result.get("num_steps", result.get("steps", 0)) or 0)
        return SkillExecutionResult(
            success=success,
            skill_command=command,
            steps=steps,
            failure_type=result.get("failure_type"),
            trajectory_path=context.get("trajectory_path"),
            metrics={
                "raw_result": result,
                "symbolic_action": {"name": action.name, "args": action.args},
                "executor": type(self).__name__,
            },
        )


def _infer_cubes(command: SkillCommand) -> list[str]:
    cubes = [command.object_id]
    if command.target_object_id and command.target_object_id != "table":
        cubes.append(command.target_object_id)
    return list(dict.fromkeys(cubes))
