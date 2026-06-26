"""Goal validator — checks goals for legality and feasibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_nsm.symbolic.predicates import Predicate
from vlm_nsm.symbolic.goals import SymbolicGoal


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "reason": self.reason, "warnings": self.warnings}


class GoalValidator:
    """Validate symbolic goals against a set of known object IDs."""

    def __init__(self, available_objects: list[str]):
        self.objects = set(available_objects)

    def validate(self, goal: SymbolicGoal) -> ValidationResult:
        if not goal.predicates:
            return ValidationResult(False, "empty goal")

        for p in goal.predicates:
            result = self._validate_predicate(p)
            if not result.valid:
                return result

        # Cross-predicate checks
        on_pairs = [(p.args[0], p.args[1]) for p in goal.predicates if p.name == "on"]
        return self._validate_cross(on_pairs)

    def _validate_predicate(self, p: Predicate) -> ValidationResult:
        for arg in p.args:
            if arg not in self.objects:
                return ValidationResult(False, f"unknown object '{arg}' in {p}")

        if p.name == "on":
            if p.args[0] == p.args[1]:
                return ValidationResult(False, f"self-reference on({p.args[0]}, {p.args[0]})")

        # Check object IDs exist
        if p.name == "ontable":
            if p.args[0] not in self.objects:
                return ValidationResult(False, f"unknown object in {p}")

        return ValidationResult(True)

    def _validate_cross(self, on_pairs: list[tuple[str, str]]) -> ValidationResult:
        # Multiple supports
        tops: dict[str, set[str]] = {}
        for top, bottom in on_pairs:
            tops.setdefault(top, set()).add(bottom)
        for top, supports in tops.items():
            if len(supports) > 1:
                return ValidationResult(False, f"multiple supports for {top}: {supports}")

        # Cycles
        for i, (a, b) in enumerate(on_pairs):
            for j, (c, d) in enumerate(on_pairs):
                if i < j and (a == d and b == c):
                    return ValidationResult(False, f"cycle: on({a},{b}) and on({b},{a})")
                if i != j:
                    visited = set()
                    current = b
                    while current in dict(on_pairs):
                        if current == a:
                            return ValidationResult(False, f"chain cycle involving {a}")
                        if current in visited:
                            break
                        visited.add(current)
                        current = dict(on_pairs)[current]

        return ValidationResult(True)
