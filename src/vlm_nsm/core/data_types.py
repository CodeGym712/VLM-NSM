from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PredicateTuple = tuple[str, ...]


@dataclass(frozen=True)
class Predicate:
    name: str
    args: list[str] = field(default_factory=list)

    def as_tuple(self) -> PredicateTuple:
        return tuple([self.name, *self.args])

    def to_pddl(self) -> str:
        return " ".join([self.name, *self.args])


@dataclass(frozen=True)
class Action:
    name: str
    args: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return " ".join([self.name.upper(), *[a.upper() for a in self.args]])


@dataclass
class PlanningProblem:
    domain_name: str
    objects: list[str]
    types: dict[str, str]
    init_predicates: list[Predicate]
    goal_predicates: list[Predicate]
    problem_name: str = "problem_generated"


@dataclass
class PlanResult:
    success: bool
    actions: list[Action]
    raw_output: str
    error: str | None = None


@dataclass
class ExecutionResult:
    success: bool
    final_state: set[PredicateTuple]
    failed_step: int | None = None
    error: str | None = None


@dataclass
class GoalParseResult:
    success: bool
    goal_json: dict[str, Any] | None
    raw_output: str = ""
    error: str | None = None
    inference_time: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw_output_path: str | None = None

    @property
    def parsed_json(self) -> dict[str, Any] | None:
        return self.goal_json


@dataclass
class AlignmentResult:
    success: bool
    goal_predicates: list[Predicate]
    repaired: bool = False
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# ── Layer 3 Monitor Types ──────────────────────────────────────────────

@dataclass
class StepSignal:
    """Step-level termination signal from TerminationFailureMonitor."""
    done: bool = False
    success: bool = False
    failed: bool = False
    timeout: bool = False
    failure_type: str | None = None


@dataclass
class PredicateState:
    """Symbolic state from PhysicalPredicateExtractor."""
    predicates: list[Predicate] = field(default_factory=list)
    source: str = "physical"  # "symbolic" or "physical"


@dataclass
class GoalSignal:
    """Goal satisfaction evaluation from PredicateMonitor."""
    goal_satisfied: bool = False
    missing_predicates: list[str] = field(default_factory=list)
