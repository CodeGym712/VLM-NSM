"""Unified FailureType enum — single source of truth for all failure classification.

Replaces string-based fuzzy matching across 10+ files with a shared enum.
All skills, state machines, monitors, and summarizers MUST use these constants.

Usage:
    from vlm_nsm.core.failure_types import FailureType

    result.failure_type = FailureType.NOCONTACT
    if result.failure_type == FailureType.NOCONTACT: ...
"""

from __future__ import annotations

from enum import Enum


class FailureType(str, Enum):
    """Unified failure type enumeration.

    Categorization:
      - GRASP: failures during grasp phase
      - PLACE: failures during place/release phase
      - REACH: failures during reach/approach phase
      - LIFT: failures during lift phase
      - SYSTEM: environment or system-level failures
      - UNKNOWN: unclassified (should be investigated)
    """

    # ── Grasp failures ──────────────────────────────────────────────────
    NOCONTACT = "nocontact"             # gripper closed without contacting object
    SLIP = "slip"                       # object slipped during lift
    PREREACH = "prereach"               # failed to reach pick object
    GRASP_FAILED = "grasp_failed"       # generic grasp failure

    # ── Place failures ──────────────────────────────────────────────────
    PREPLACE = "preplace"               # failed to reach place target
    VERIFY_FAILED = "verify_failed"     # on-predicate not satisfied after place
    PLACE_FAILED = "place_failed"       # generic place failure
    TRANSPORT_DROP = "transport_drop"   # object dropped during transport
    PLACE_UNSTABLE = "place_unstable"   # object placed but unstable

    # ── Reach failures ──────────────────────────────────────────────────
    REACH_FAILED = "reach_failed"       # failed to converge to target
    REACH_PICK_FAILED = "reach_pick_failed"  # specific to reach_pick phase
    APPROACH_BLOCKED = "approach_blocked"    # path to target is blocked

    # ── Lift failures ───────────────────────────────────────────────────
    LIFT_FAILED = "lift_failed"         # failed to lift object

    # ── System / env failures ───────────────────────────────────────────
    TIMEOUT = "timeout"                 # skill exceeded step budget
    ENV_ERROR = "env_error"             # environment exception (terminated, etc.)
    HORIZON = "horizon"                 # episode horizon exhausted
    HORIZON_PREPLACE = "horizon_preplace"  # horizon exhausted before preplace

    # ── Unknown ─────────────────────────────────────────────────────────
    UNKNOWN = "unknown"                 # unclassified failure

    # ── Recovery-specific ───────────────────────────────────────────────
    RETREAT_FAILED = "retreat_failed"   # retreat phase failure (recovery context)

    @classmethod
    def from_string(cls, s: str | None) -> "FailureType":
        """Parse from string, with fallback to UNKNOWN."""
        if s is None:
            return cls.UNKNOWN
        try:
            return cls(s.lower().strip())
        except ValueError:
            # Partial match fallback
            s_lower = s.lower().strip()
            for member in cls:
                if member.value in s_lower:
                    return member
            return cls.UNKNOWN

    @classmethod
    def is_recoverable(cls, ft: "FailureType") -> bool:
        """Whether this failure type can be recovered from."""
        return ft not in {
            cls.ENV_ERROR,
            cls.UNKNOWN,
        }

    @classmethod
    def recommended_repair(cls, ft: "FailureType") -> str:
        """Get recommended repair strategy for this failure type."""
        mapping = {
            cls.NOCONTACT: "retry_grasp",
            cls.SLIP: "regrasp",
            cls.PREREACH: "xy_realign",
            cls.GRASP_FAILED: "retry_grasp",
            cls.PREPLACE: "regrasp_and_replan",
            cls.TRANSPORT_DROP: "regrasp_and_replan",
            cls.VERIFY_FAILED: "micro_place",
            cls.PLACE_FAILED: "micro_place",
            cls.PLACE_UNSTABLE: "micro_place",
            cls.REACH_FAILED: "xy_realign",
            cls.REACH_PICK_FAILED: "xy_realign",
            cls.APPROACH_BLOCKED: "xy_realign",
            cls.LIFT_FAILED: "retry_with_params",
            cls.TIMEOUT: "retry_with_params",
            cls.HORIZON: "none",
            cls.HORIZON_PREPLACE: "none",
            cls.ENV_ERROR: "none",
            cls.RETREAT_FAILED: "retreat",
        }
        return mapping.get(ft, "retry")
