"""Base skill abstractions for task-agnostic PID-controlled execution."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_OSC_POSE_MAX_DELTA = 0.05


@dataclass
class SkillParams:
    """Parameters for a single skill execution.

    Attributes:
        target_object: Object to manipulate (e.g. "cube1").
        target_location: Where to move/place (e.g. "cube2" or "table").
        approach_height: Height offset above object for approach.
        gripper_cmd: -1.0 open, 1.0 close.
        reach_threshold: Distance threshold for convergence (meters).
        max_steps: Maximum steps before timeout.
        phase_id: Integer phase ID for ID model input.
    """

    target_object: str = ""
    target_location: str = ""
    approach_height: float = 0.10
    gripper_cmd: float = -1.0
    reach_threshold: float = 0.02
    max_steps: int = 500
    phase_id: int = 0


@dataclass
class SkillResult:
    """Result of a skill execution.

    Attributes:
        success: Whether the skill completed successfully.
        steps: Number of steps taken.
        failure_type: Short string describing the failure (e.g. "timeout").
        diagnostics: Optional per-step diagnostics.
    """

    success: bool = False
    steps: int = 0
    failure_type: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # OptionResult fields (improvement_plan.md A2, 2026-06-20)
    pre_predicates: list = field(default_factory=list)
    post_predicates: list = field(default_factory=list)
    expected_effect: list = field(default_factory=list)
    effect_satisfied: bool = False
    controller: str = "pid"
    replan_allowed: bool = True


def _dist(a: dict, b: dict) -> float:
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
    )


def norm_action(dx: float, dy: float, dz: float, gripper: float) -> np.ndarray:
    """Convert physical deltas to 7-dim normalized action [-1, 1]."""
    return np.array(
        [
            np.clip(dx / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            np.clip(dy / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            np.clip(dz / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
            0.0,
            0.0,
            0.0,
            gripper,
        ],
        dtype=np.float32,
    )


class BaseSkill(ABC):
    """Abstract skill — all manipulation skills implement this interface.

    Each skill is self-contained: it knows how to compute targets,
    check completion, and execute using a PID controller.
    """

    name: str = "base"

    @abstractmethod
    def get_target(self, env, params: SkillParams) -> dict:
        """Compute target position from environment state and params.

        Returns:
            {"x", "y", "z"} target position for EEF.
        """

    @abstractmethod
    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        """Check if the skill has completed successfully.

        Args:
            env: Environment wrapper.
            eef: Current EEF pose {"x","y","z"}.
            target: Current target position.
            params: Skill parameters.
        """

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        """Execute skill using PID controller.

        This is the default implementation. Subclasses may override
        for skills that need custom execution (e.g. grasp_state_machine).

        Args:
            env: Environment wrapper (with get_eef_pose, get_object_pose, step).
            pid_ctrl: OperationalSpacePID instance.
            params: Skill parameters.

        Returns:
            SkillResult with success/failure and step count.
        """
        target = self.get_target(env, params)
        pid_ctrl.set_phase(self.name)
        pid_ctrl.set_target(
            pos=target, vel=None, close_gripper=(params.gripper_cmd > 0)
        )

        steps = 0
        for _ in range(params.max_steps):
            try:
                eef = env.get_eef_pose()
            except (ValueError, AttributeError):
                return SkillResult(False, steps, "env_error")

            pid_ctrl.set_current(eef)
            action = pid_ctrl.act(self.name, None)
            action[6] = params.gripper_cmd

            try:
                env.step(action)
            except (ValueError, AttributeError):
                return SkillResult(False, steps, "env_error")

            steps += 1
            if self.check_completion(env, eef, target, params):
                return SkillResult(True, steps)

        return SkillResult(False, steps, "timeout")


class SkillLibrary:
    """Registry of available skills, indexed by name."""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not registered. Available: {list(self._skills.keys())}")
        return self._skills[name]

    def __contains__(self, name: str) -> bool:
        return name in self._skills
