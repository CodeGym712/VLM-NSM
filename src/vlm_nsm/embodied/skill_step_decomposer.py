"""SkillStepDecomposer.

Decomposes symbolic move actions into fixed step sequences for execution.
Current version uses a fixed task-specific template for Stacking pick-and-place.
Dynamic step generation is future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vlm_nsm.core.data_types import Action


SKILL_STEP_NAMES = [
    "reach_pick",
    "grasp",
    "lift",
    "reach_place",
    "release",
    "retreat",
]


@dataclass
class DecomposedSteps:
    """Result of decomposing a symbolic action into executable steps."""
    action: Action
    steps: list[str] = field(default_factory=list)
    step_count: int = 0


class SkillStepDecomposer:
    """Decomposes symbolic actions into fixed step sequences.

    Current version: Fixed template for Stacking pick-and-place move actions.
    Future work: Dynamic step generation based on action type, object properties,
    and scene context.

    For move(obj, source, target), the decomposition is:
        1. reach_pick — move to grasp pose above object
        2. grasp — close gripper
        3. lift — lift object
        4. reach_place — move to place pose above target
        5. release — open gripper
        6. retreat — move away
    """

    def decompose(self, action: Action) -> DecomposedSteps:
        """Decompose a symbolic action into skill steps.

        Args:
            action: Symbolic action from the planner.

        Returns:
            DecomposedSteps with the step sequence.

        Raises:
            ValueError: If the action type is not supported.
        """
        if action.name == "move":
            return DecomposedSteps(
                action=action,
                steps=list(SKILL_STEP_NAMES),
                step_count=len(SKILL_STEP_NAMES),
            )
        elif action.name in ("pick", "place"):
            # Atomic pick/place (used by some planner configurations)
            if action.name == "pick":
                return DecomposedSteps(
                    action=action,
                    steps=["reach_pick", "grasp", "lift"],
                    step_count=3,
                )
            else:
                return DecomposedSteps(
                    action=action,
                    steps=["reach_place", "release", "retreat"],
                    step_count=3,
                )
        else:
            raise ValueError(f"Unsupported action type for decomposition: {action.name}")
