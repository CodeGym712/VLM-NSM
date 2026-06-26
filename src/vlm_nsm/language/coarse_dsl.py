"""Coarse Option DSL — pick/place/move/ensure/achieve for upper layer.

Phase-level skills (reach/grasp/lift/reach_place/release_place) are FORBIDDEN
as upper-layer actions. They exist only as internal implementation of pick/place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Phase-level names that must NOT appear in coarse DSL output
_FORBIDDEN_OPS = {"reach", "grasp", "lift", "reach_place", "release_place", "retreat"}

# Allowed coarse operations
_VALID_OPS = {"pick", "place", "move", "ensure", "achieve"}


@dataclass
class CoarseStep:
    """A single coarse operation step."""
    op: str = ""                      # pick | place | move | ensure | achieve
    object: str = ""                  # target object
    target: str = ""                  # for place/move
    source: str = ""                  # for move
    predicate: list = field(default_factory=list)  # for ensure/achieve

    def is_valid(self) -> tuple[bool, str]:
        """Validate this step."""
        if self.op not in _VALID_OPS:
            return False, f"unknown op '{self.op}'. Allowed: {_VALID_OPS}"
        if self.op in _FORBIDDEN_OPS:
            return False, f"'{self.op}' is a phase-level skill, forbidden in coarse DSL"
        if self.op in ("pick", "place", "move") and not self.object:
            return False, f"'{self.op}' requires 'object' field"
        if self.op in ("place", "move") and not self.target:
            return False, f"'{self.op}' requires 'target' field"
        if self.op in ("ensure", "achieve") and not self.predicate:
            return False, f"'{self.op}' requires 'predicate' field"
        if self.op == "place" and self.object == self.target:
            return False, f"self-place: {self.object} on {self.object}"
        return True, ""


@dataclass
class CoarseTaskProgram:
    """A task program using only coarse semantic options."""
    steps: list[CoarseStep] = field(default_factory=list)
    version: str = "1.0"

    @classmethod
    def from_json(cls, data: dict) -> "CoarseTaskProgram | None":
        """Parse from JSON dict. Returns None if invalid."""
        steps_data = data.get("steps", [])
        if not steps_data:
            return None
        steps = []
        for s in steps_data:
            step = CoarseStep(
                op=s.get("op", ""),
                object=s.get("object", ""),
                target=s.get("target", ""),
                source=s.get("source", ""),
                predicate=s.get("predicate", []),
            )
            valid, reason = step.is_valid()
            if not valid:
                return None
            steps.append(step)
        return cls(steps=steps)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "steps": [
                {"op": s.op, "object": s.object, "target": s.target,
                 "source": s.source, "predicate": s.predicate}
                for s in self.steps
            ],
        }

    def is_valid(self) -> tuple[bool, str]:
        """Validate the entire program."""
        if not self.steps:
            return False, "empty steps"
        holding = None
        for i, s in enumerate(self.steps):
            valid, reason = s.is_valid()
            if not valid:
                return False, f"step {i}: {reason}"
            # Sequence check
            if s.op == "pick":
                if holding is not None:
                    return False, f"step {i}: pick({s.object}) but already holding {holding}"
                holding = s.object
            elif s.op == "place":
                if holding != s.object:
                    return False, f"step {i}: place({s.object}) but holding {holding}"
                holding = None
        return True, ""

    def to_goal_predicates(self) -> list[list[str]]:
        """Extract goal predicates from achieve/place steps."""
        goals = []
        for s in self.steps:
            if s.op == "achieve" and s.predicate:
                goals.append(s.predicate)
            elif s.op == "place":
                goals.append(["on", s.object, s.target])
        return goals
