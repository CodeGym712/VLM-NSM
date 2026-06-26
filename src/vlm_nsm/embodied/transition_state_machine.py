"""Transition State Machine — semantic substate definitions for skill phases.

Replaces hardcoded target positions (e.g. ``target_z = obj_z + 0.10``)
with semantic desired substates and transition rules.

Design:
  - Each phase defines: entry_condition, desired_substate, target_fn,
    controller_config, completion_condition, failure_conditions
  - target_fn computes physical target from semantic state
  - completion is verified by condition check, not step count
  - failure detection triggers retry or recovery

Usage:
    from vlm_nsm.embodied.transition_state_machine import (
        TransitionStateMachine, Substates, PhaseDefinition
    )

    tsm = TransitionStateMachine(n_objects=2)
    target = tsm.get_target("reach_pick", pick_obj="cube1")
    # → {"x": obj_x, "y": obj_y, "z": obj_z + 0.10}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ── Semantic Substate Definitions ─────────────────────────────────────────────

class Substate(str, Enum):
    """Semantic substates that define desired robot-object relationships."""
    # Pick-related
    EEF_ABOVE_PICK = "eef_above_pick_object"
    EEF_AT_PICK = "eef_at_pick_surface"
    OBJECT_IN_GRIPPER = "object_in_gripper"
    OBJECT_LIFTED = "object_lifted"

    # Place-related
    EEF_ABOVE_TARGET = "eef_above_target"
    EEF_AT_TARGET = "eef_at_target_surface"
    OBJECT_PLACED = "object_placed"

    # General
    EEF_RETREATED = "eef_retreated"
    IDLE = "idle"
    FAILED = "failed"


# ── Phase Definition ──────────────────────────────────────────────────────────

@dataclass
class PhaseDefinition:
    """Complete definition of a skill execution phase.

    Attributes:
        name: Phase name (e.g. "reach_pick").
        desired_substate: Semantic state this phase aims to achieve.
        target_fn: Function(env, pick_obj, target_obj, **kwargs) → {"x","y","z"}.
            Computes the physical target position from current environment state.
        approach_height: Height offset above object for approach (meters).
        gripper_cmd: Gripper command during this phase (-1 open, 1 close).
        completion_dist: Distance threshold for position convergence (meters).
        completion_extra: Optional extra condition function.
        max_steps: Maximum steps before timeout.
        use_learned_id: Whether to enable Learned ID for this phase.
        holding_check: Whether to periodically check object holding.

        Note: PID gains are managed by GainSchedule in operational_space_pid.py.
        This class does NOT override gains.
    """

    name: str
    desired_substate: Substate
    approach_height: float = 0.10
    gripper_cmd: float = -1.0
    completion_dist: float = 0.02
    max_steps: int = 500
    use_learned_id: bool = False
    holding_check: bool = False
    completion_extra: Optional[Callable] = None

    def get_target(self, pick_pose: dict, target_pose: dict | None = None) -> dict:
        """Compute physical target from object poses.

        Args:
            pick_pose: {"x","y","z"} of the pick object.
            target_pose: {"x","y","z"} of the target (for place phases).

        Returns:
            {"x","y","z"} target position for EEF.
        """
        if "pick" in self.name or self.name in ("grasp", "lift"):
            # Target above or at pick object
            z_offset = self._resolve_z_offset(pick_pose.get("z", 0.8))
            return {
                "x": pick_pose["x"],
                "y": pick_pose["y"],
                "z": pick_pose["z"] + z_offset,
            }
        elif "place" in self.name or self.name in ("release",):
            # Target above or at target object
            tgt = target_pose if target_pose else pick_pose
            return {
                "x": tgt["x"],
                "y": tgt["y"],
                "z": tgt["z"] + self.approach_height,
            }
        else:
            return {"x": 0.0, "y": 0.0, "z": 1.0}

    def _resolve_z_offset(self, obj_z: float) -> float:
        """Resolve Z offset based on phase semantics."""
        if self.name == "reach_pick":
            return 0.10  # approach from above
        elif self.name == "grasp":
            return 0.005  # close to surface
        elif self.name == "lift":
            return 0.10 + 0.05  # approach_height + lift_height
        return self.approach_height


# ── Default 2-Block Phase Definitions ─────────────────────────────────────────

def _default_phases_2block() -> dict[str, PhaseDefinition]:
    """Default phase definitions for 2-block stacking task."""
    return {
        "reach_pick": PhaseDefinition(
            name="reach_pick",
            desired_substate=Substate.EEF_ABOVE_PICK,
            approach_height=0.10,
            gripper_cmd=-1.0,
            completion_dist=0.02,
            max_steps=500,
            use_learned_id=False,  # VLA handles this phase
        ),
        "grasp": PhaseDefinition(
            name="grasp",
            desired_substate=Substate.OBJECT_IN_GRIPPER,
            approach_height=0.005,
            gripper_cmd=1.0,
            completion_dist=0.015,
            max_steps=300,
            use_learned_id=True,  # Learned ID for adaptive grasp
        ),
        "lift": PhaseDefinition(
            name="lift",
            desired_substate=Substate.OBJECT_LIFTED,
            approach_height=0.15,
            gripper_cmd=1.0,
            completion_dist=0.02,
            max_steps=200,
            use_learned_id=False,
        ),
        "reach_place": PhaseDefinition(
            name="reach_place",
            desired_substate=Substate.EEF_ABOVE_TARGET,
            approach_height=0.15,
            gripper_cmd=1.0,
            completion_dist=0.02,
            max_steps=500,
            use_learned_id=True,  # Learned ID for transport
            holding_check=True,   # Check holding during transport
        ),
        "release": PhaseDefinition(
            name="release",
            desired_substate=Substate.OBJECT_PLACED,
            approach_height=0.02,
            gripper_cmd=-1.0,
            completion_dist=0.02,
            max_steps=100,
            use_learned_id=False,
        ),
        "retreat": PhaseDefinition(
            name="retreat",
            desired_substate=Substate.EEF_RETREATED,
            approach_height=0.15,  # retreat height
            gripper_cmd=-1.0,
            completion_dist=0.02,
            max_steps=200,
            use_learned_id=False,
        ),
    }


# ── Transition State Machine ──────────────────────────────────────────────────

class TransitionStateMachine:
    """State machine that computes targets from semantic substates.

    Args:
        n_objects: Number of objects in the scene (2 or 3).
        phases: Optional custom phase definitions.
    """

    def __init__(
        self,
        n_objects: int = 2,
        phases: dict[str, PhaseDefinition] | None = None,
    ):
        self.n_objects = n_objects
        self.phases = phases or _default_phases_2block()
        self._current_phase: str = "reach_pick"
        self._phase_history: list[str] = []

    # ── Query API ────────────────────────────────────────────────────────

    def get_target(
        self,
        phase_name: str,
        pick_pose: dict,
        target_pose: dict | None = None,
    ) -> dict:
        """Get physical target position for a phase.

        Args:
            phase_name: One of reach_pick, grasp, lift, reach_place, release, retreat.
            pick_pose: {"x","y","z"} of the object being manipulated.
            target_pose: {"x","y","z"} of the target location (for place phases).

        Returns:
            {"x","y","z"} target for EEF motion.
        """
        phase = self.phases.get(phase_name)
        if phase is None:
            return {"x": 0.0, "y": 0.0, "z": 1.0}
        return phase.get_target(pick_pose, target_pose)

    def get_phase_def(self, phase_name: str) -> PhaseDefinition | None:
        """Get the full phase definition."""
        return self.phases.get(phase_name)

    def get_pid_overrides(self, phase_name: str) -> dict[str, float]:
        """DEPRECATED: Gains managed by GainSchedule in operational_space_pid.py."""
        return {}

    def should_use_learned_id(self, phase_name: str) -> bool:
        """Whether to enable Learned ID for this phase."""
        phase = self.phases.get(phase_name)
        return phase.use_learned_id if phase else False

    def should_check_holding(self, phase_name: str) -> bool:
        """Whether to periodically check holding during this phase."""
        phase = self.phases.get(phase_name)
        return phase.holding_check if phase else False

    # ── Transition API ───────────────────────────────────────────────────

    def transition(self, phase_name: str):
        """Record a phase transition."""
        self._phase_history.append(self._current_phase)
        self._current_phase = phase_name

    def reset(self):
        """Reset for a new episode."""
        self._current_phase = "reach_pick"
        self._phase_history = []

    @property
    def current_phase(self) -> str:
        return self._current_phase

    @property
    def phase_history(self) -> list[str]:
        return list(self._phase_history)
