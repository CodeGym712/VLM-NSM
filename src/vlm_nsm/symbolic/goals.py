"""Symbolic goal representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_nsm.symbolic.predicates import Predicate, on, ontable


@dataclass
class SymbolicGoal:
    predicates: set[Predicate] = field(default_factory=set)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicates": [p.to_dict() for p in sorted(self.predicates, key=str)],
            "description": self.description,
        }

    def __str__(self) -> str:
        return f"Goal({', '.join(str(p) for p in self.predicates)})"


class StackOrderGoal(SymbolicGoal):
    """Goal: stack objects in a specific bottom-to-top order."""

    def __init__(self, order: list[str], description: str = ""):
        if len(order) < 2:
            raise ValueError(f"Stack order needs at least 2 objects: {order}")
        if len(set(order)) != len(order):
            raise ValueError(f"Duplicate objects in stack order: {order}")

        preds: set[Predicate] = set()
        preds.add(ontable(order[0]))
        for i in range(len(order) - 1):
            preds.add(on(order[i + 1], order[i]))
        desc = description or f"stack_order({', '.join(order)})"
        super().__init__(predicates=preds, description=desc)


def single_stack_goal(top: str, bottom: str) -> SymbolicGoal:
    """Create goal: on(top, bottom) + ontable(bottom)."""
    return SymbolicGoal(
        predicates={on(top, bottom), ontable(bottom)},
        description=f"on({top}, {bottom})",
    )
