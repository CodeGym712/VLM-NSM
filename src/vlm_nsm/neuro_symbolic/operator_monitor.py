"""OperatorMonitor — per-operator precondition & effect verification.

Checks whether symbolic operators (MoveOn, MoveToTable) achieve their
expected effects during physical execution. Maps low-level controller
failure labels to operator-level failures and suggests recovery strategies.

Design:
  - Independent of Robosuite — works with any state that provides
    predicate-like keys.
  - Read-only: does not modify state or trigger actions.
  - Used by the episode runner after each operator execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vlm_nsm.skills.skill_types import SkillCommand


# ── Operator-level failure types ────────────────────────────────────────────

class OperatorFailureType:
    """Operator-level failure classification."""
    # Pick failures
    PICK_NO_CONTACT = "PickFailed.NO_CONTACT"
    PICK_BAD_GRASP = "PickFailed.BAD_GRASP"
    PICK_LIFT_FAIL = "PickFailed.LIFT_FAIL"
    PICK_TIMEOUT = "PickFailed.TIMEOUT"

    # PlaceOn failures
    PLACE_NOT_HOLDING = "PlaceOnFailed.NOT_HOLDING"
    PLACE_XY_MISALIGN = "PlaceOnFailed.XY_MISALIGN"
    PLACE_Z_HEIGHT_ERROR = "PlaceOnFailed.Z_HEIGHT_ERROR"
    PLACE_TARGET_MOVED = "PlaceOnFailed.TARGET_MOVED"
    PLACE_UNSTABLE = "PlaceOnFailed.UNSTABLE_RELEASE"
    PLACE_VERIFY_FAILED = "PlaceOnFailed.VERIFY_FAILED"
    PLACE_TIMEOUT = "PlaceOnFailed.TIMEOUT"

    # Global
    EXECUTION_TIMEOUT = "ExecutionFailed.TIMEOUT"
    UNKNOWN = "Unknown"


# ── Recovery hints ──────────────────────────────────────────────────────────

_RECOVERY_MAP: dict[str, str] = {
    OperatorFailureType.PICK_NO_CONTACT: "regrasp",
    OperatorFailureType.PICK_BAD_GRASP: "regrasp",
    OperatorFailureType.PICK_LIFT_FAIL: "regrasp",
    OperatorFailureType.PLACE_NOT_HOLDING: "re_pick",
    OperatorFailureType.PLACE_XY_MISALIGN: "micro_place",
    OperatorFailureType.PLACE_Z_HEIGHT_ERROR: "adjust_height",
    OperatorFailureType.PLACE_TARGET_MOVED: "replan",
    OperatorFailureType.PLACE_UNSTABLE: "stabilize",
    OperatorFailureType.PLACE_VERIFY_FAILED: "micro_place",
    OperatorFailureType.PLACE_TIMEOUT: "replan",
    OperatorFailureType.EXECUTION_TIMEOUT: "replan",
}


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class OperatorEffectResult:
    """Result of checking a single expected effect."""
    predicate_name: str
    predicate_args: list[str]
    satisfied: bool


@dataclass
class OperatorCheckResult:
    """Complete result of per-operator verification."""
    operator_name: str
    bindings: dict[str, str] = field(default_factory=dict)

    # Preconditions
    preconditions_satisfied: bool = True
    missing_preconditions: list[str] = field(default_factory=list)

    # Effects
    add_effects_ok: list[OperatorEffectResult] = field(default_factory=list)
    del_effects_ok: list[OperatorEffectResult] = field(default_factory=list)
    all_effects_satisfied: bool = False

    # Failure classification
    low_level_failure: str | None = None
    operator_failure: str | None = None
    recovery_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "preconditions_satisfied": self.preconditions_satisfied,
            "missing_preconditions": self.missing_preconditions,
            "all_effects_satisfied": self.all_effects_satisfied,
            "add_effects": [
                {"predicate": f"{e.predicate_name}({','.join(e.predicate_args)})",
                 "satisfied": e.satisfied}
                for e in self.add_effects_ok
            ],
            "low_level_failure": self.low_level_failure,
            "operator_failure": self.operator_failure,
            "recovery_hint": self.recovery_hint,
        }


# ── OperatorMonitor ───────────────────────────────────────────────────────

class OperatorMonitor:
    """Checks operator preconditions and effects during execution.

    Usage::

        monitor = OperatorMonitor()
        result = monitor.check_after_execution(
            command=skill_command,       # with preconditions/effects
            state_after=predicate_set,   # from physical_predicate_extractor
            low_level_failure="preplace",
        )
        print(result.operator_failure)   # → "PlaceOnFailed.XY_MISALIGN"
        print(result.recovery_hint)      # → "micro_place"
    """

    # ── Low-level → operator-level failure mapping ──────────────────────

    def classify_failure(
        self,
        command: SkillCommand,
        low_level_failure: str | None,
        effect_results: list[OperatorEffectResult] | None = None,
    ) -> tuple[str | None, str | None]:
        """Map low-level failure to operator-level failure + recovery hint.

        Args:
            command: The SkillCommand that was executed.
            low_level_failure: Low-level failure string (e.g. "preplace").
            effect_results: Optional per-effect satisfaction results.

        Returns:
            (operator_failure, recovery_hint) tuple.
        """
        if not low_level_failure:
            return None, None

        ft = low_level_failure.lower().strip()

        # Grasp/pick failures
        if ft in ("nocontact", "no_contact"):
            return (OperatorFailureType.PICK_NO_CONTACT,
                    _RECOVERY_MAP[OperatorFailureType.PICK_NO_CONTACT])
        if ft in ("grasp_failed", "grasp_failure"):
            return (OperatorFailureType.PICK_BAD_GRASP,
                    _RECOVERY_MAP[OperatorFailureType.PICK_BAD_GRASP])
        if ft in ("lift_fail", "lift_failed"):
            return (OperatorFailureType.PICK_LIFT_FAIL,
                    _RECOVERY_MAP[OperatorFailureType.PICK_LIFT_FAIL])

        # Place/preplace failures — try to disambiguate using effects
        if ft in ("preplace", "preplace_failed", "preplace_misaligned"):
            # If we have effect results, use them to disambiguate
            if effect_results:
                return self._disambiguate_place_failure(effect_results)
            # Default: assume XY misalignment (most common)
            return (OperatorFailureType.PLACE_XY_MISALIGN,
                    _RECOVERY_MAP[OperatorFailureType.PLACE_XY_MISALIGN])

        if ft in ("verify_failed", "verification_failed"):
            return (OperatorFailureType.PLACE_VERIFY_FAILED,
                    _RECOVERY_MAP[OperatorFailureType.PLACE_VERIFY_FAILED])

        # Horizon/timeout
        if ft.startswith("horizon"):
            if "preplace" in ft:
                return (OperatorFailureType.PLACE_TIMEOUT,
                        _RECOVERY_MAP[OperatorFailureType.PLACE_TIMEOUT])
            return (OperatorFailureType.EXECUTION_TIMEOUT,
                    _RECOVERY_MAP[OperatorFailureType.EXECUTION_TIMEOUT])
        if ft == "timeout":
            return (OperatorFailureType.EXECUTION_TIMEOUT,
                    _RECOVERY_MAP[OperatorFailureType.EXECUTION_TIMEOUT])

        return (ft, None)

    def _disambiguate_place_failure(
        self, effect_results: list[OperatorEffectResult],
    ) -> tuple[str, str]:
        """Use per-effect results to determine the specific place failure."""
        for e in effect_results:
            if e.predicate_name == "on" and not e.satisfied:
                # on(x,y) failed — check if holding was maintained
                holding_ok = any(
                    r.predicate_name == "holding" and r.satisfied
                    for r in effect_results
                )
                if not holding_ok:
                    return (OperatorFailureType.PLACE_NOT_HOLDING,
                            _RECOVERY_MAP[OperatorFailureType.PLACE_NOT_HOLDING])
                return (OperatorFailureType.PLACE_XY_MISALIGN,
                        _RECOVERY_MAP[OperatorFailureType.PLACE_XY_MISALIGN])
        return (OperatorFailureType.PLACE_VERIFY_FAILED,
                _RECOVERY_MAP[OperatorFailureType.PLACE_VERIFY_FAILED])

    # ── Effect checking ─────────────────────────────────────────────────

    def check_effects(
        self,
        command: SkillCommand,
        state_predicates: set | list,
    ) -> list[OperatorEffectResult]:
        """Check which expected effects are satisfied in the current state.

        Args:
            command: SkillCommand with expected_add_effects/expected_del_effects.
            state_predicates: Current predicates as a set of tuples or strings.

        Returns:
            List of OperatorEffectResult for each add effect.
        """
        # Normalize state predicates to (name, arg0, arg1, ...) tuples
        normalized: set[tuple[str, ...]] = set()
        for p in state_predicates:
            if isinstance(p, tuple):
                normalized.add(tuple(str(x) for x in p))
            elif isinstance(p, str):
                parts = p.replace("(", " ").replace(")", " ").replace(",", " ").split()
                normalized.add(tuple(parts))
            elif hasattr(p, "name") and hasattr(p, "args"):
                normalized.add((str(p.name),) + tuple(str(a) for a in p.args))

        results = []
        for effect in command.expected_add_effects:
            name = effect.get("name", "")
            args = effect.get("args", [])
            key = (name,) + tuple(str(a) for a in args)
            satisfied = key in normalized
            results.append(OperatorEffectResult(
                predicate_name=name,
                predicate_args=list(args),
                satisfied=satisfied,
            ))
        return results

    def all_effects_satisfied(self, results: list[OperatorEffectResult]) -> bool:
        return all(r.satisfied for r in results) if results else True

    # ── Full check ──────────────────────────────────────────────────────

    def check_after_execution(
        self,
        command: SkillCommand,
        state_after: set | list | None = None,
        low_level_failure: str | None = None,
    ) -> OperatorCheckResult:
        """Run full operator-level check after execution.

        Args:
            command: The executed SkillCommand.
            state_after: Current predicate set (from physical_predicate_extractor).
            low_level_failure: Low-level failure label from controller.

        Returns:
            OperatorCheckResult with operator-level diagnosis.
        """
        result = OperatorCheckResult(
            operator_name=command.skill_name,
            bindings={
                "object_id": command.object_id,
                "target_object_id": command.target_object_id or "",
            },
        )

        # Check effects
        if state_after and command.expected_add_effects:
            effect_results = self.check_effects(command, state_after)
            result.add_effects_ok = effect_results
            result.all_effects_satisfied = self.all_effects_satisfied(effect_results)
        else:
            result.all_effects_satisfied = (low_level_failure is None)

        # Classify failure
        result.low_level_failure = low_level_failure
        if low_level_failure or not result.all_effects_satisfied:
            op_failure, recovery = self.classify_failure(
                command, low_level_failure, result.add_effects_ok)
            result.operator_failure = op_failure
            result.recovery_hint = recovery

        return result
