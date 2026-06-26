"""Failure Taxonomy — standardized failure type definitions.

Provides detailed, actionable failure classifications for execution
diagnostics, VLA training data labeling, and paper reporting.
"""

from __future__ import annotations

from typing import Any


# ── Failure Types ───────────────────────────────────────────────────────────

class FailureType:
    """Standardized failure type constants with descriptions."""

    # Reach/Motion failures
    REACH_TIMEOUT = "reach_timeout"
    PREGRASP_MISALIGNED = "pregrasp_misaligned"

    # Contact/Grasp failures
    NO_CONTACT = "nocontact"
    SINGLE_SIDED_CONTACT = "single_sided_contact"
    GRASP_SLIP = "grasp_slip"
    GRASP_TIMEOUT = "grasp_timeout"

    # Lift/Transport failures
    LIFT_FAIL = "lift_fail"
    TRANSPORT_DROP = "transport_drop"

    # Place/Release failures
    PREPLACE_MISALIGNED = "preplace_misaligned"
    PLACE_UNSTABLE = "place_unstable"
    RELEASE_FAIL = "release_fail"

    # Retreat
    RETREAT_COLLISION = "retreat_collision"

    # Global
    TIMEOUT = "timeout"
    HORIZON = "horizon"
    HORIZON_PREPLACE = "horizon_preplace"

    # Verification
    VERIFY_FAILED = "verify_failed"

    # VLA-specific
    VLA_LOW_CONFIDENCE = "vla_low_confidence"
    VLA_OOD_STATE = "vla_ood_state"
    VLA_ACTION_REJECTED = "vla_action_rejected"

    # Uncategorized
    UNKNOWN = "unknown"
    RUNTIME_ERROR = "runtime_error"


# ── Failure metadata ────────────────────────────────────────────────────────

FAILURE_CATEGORIES: dict[str, dict[str, Any]] = {
    FailureType.REACH_TIMEOUT: {
        "category": "motion",
        "recoverable": True,
        "description": "Failed to reach target position within step budget",
    },
    FailureType.PREGRASP_MISALIGNED: {
        "category": "motion",
        "recoverable": True,
        "description": "EEF not aligned with object before grasp",
    },
    FailureType.NO_CONTACT: {
        "category": "grasp",
        "recoverable": True,
        "description": "Gripper closed without contacting object",
    },
    FailureType.SINGLE_SIDED_CONTACT: {
        "category": "grasp",
        "recoverable": True,
        "description": "Only one finger contacted object",
    },
    FailureType.GRASP_SLIP: {
        "category": "grasp",
        "recoverable": True,
        "description": "Object slipped out of gripper after contact",
    },
    FailureType.GRASP_TIMEOUT: {
        "category": "grasp",
        "recoverable": True,
        "description": "Grasp phase exceeded step budget",
    },
    FailureType.LIFT_FAIL: {
        "category": "transport",
        "recoverable": True,
        "description": "Object not lifted after grasp",
    },
    FailureType.TRANSPORT_DROP: {
        "category": "transport",
        "recoverable": False,
        "description": "Object dropped during transport",
    },
    FailureType.PREPLACE_MISALIGNED: {
        "category": "placement",
        "recoverable": True,
        "description": "EEF not aligned with target before release",
    },
    FailureType.PLACE_UNSTABLE: {
        "category": "placement",
        "recoverable": True,
        "description": "Object not stable on target after release",
    },
    FailureType.RELEASE_FAIL: {
        "category": "placement",
        "recoverable": True,
        "description": "Gripper failed to release object",
    },
    FailureType.RETREAT_COLLISION: {
        "category": "motion",
        "recoverable": False,
        "description": "EEF collided with object during retreat",
    },
    FailureType.TIMEOUT: {
        "category": "global",
        "recoverable": False,
        "description": "Episode exceeded global step budget",
    },
    FailureType.HORIZON: {
        "category": "global",
        "recoverable": False,
        "description": "Episode reached horizon limit",
    },
    FailureType.UNKNOWN: {
        "category": "global",
        "recoverable": False,
        "description": "Unclassified failure",
    },
}


# ── Mapping from existing failure types ─────────────────────────────────────

def normalize_failure_type(raw: str | None) -> str:
    """Map existing failure type strings to standardized taxonomy."""
    if not raw:
        return FailureType.UNKNOWN

    raw_lower = raw.lower().strip()

    # Direct matches
    if raw_lower in ("nocontact", "no_contact"):
        return FailureType.NO_CONTACT
    if raw_lower in ("preplace", "preplace_failed"):
        return FailureType.PREPLACE_MISALIGNED
    if raw_lower in ("horizon", "horizon_timeout"):
        return FailureType.HORIZON
    if raw_lower in ("horizon_preplace",):
        return FailureType.HORIZON_PREPLACE
    if raw_lower in ("verify_failed", "verification_failed"):
        return FailureType.VERIFY_FAILED
    if raw_lower in ("slip", "grasp_slip"):
        return FailureType.GRASP_SLIP
    if raw_lower in ("lift_fail", "lift_failed"):
        return FailureType.LIFT_FAIL
    if raw_lower in ("timeout",):
        return FailureType.TIMEOUT
    if raw_lower in ("runtime_error", "error"):
        return FailureType.RUNTIME_ERROR
    if raw_lower in ("grasp_failure", "grasp_failed"):
        return FailureType.NO_CONTACT  # most common grasp failure

    return raw_lower  # return as-is if no mapping


def is_recoverable(failure_type: str) -> bool:
    """Check if a failure type is typically recoverable."""
    info = FAILURE_CATEGORIES.get(failure_type, {})
    return info.get("recoverable", False)


def failure_category(failure_type: str) -> str:
    """Get the category of a failure type."""
    info = FAILURE_CATEGORIES.get(failure_type, {})
    return info.get("category", "global")


# ── Failure Distribution Summary ───────────────────────────────────────────

def summarize_failures(
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a failure distribution summary from a list of execution traces.

    Args:
        traces: List of trace dicts with "success" and "failure_type" fields.

    Returns:
        Dict with success_rate, failure_counts, per_category_counts.
    """
    total = len(traces)
    successes = sum(1 for t in traces if t.get("success", False))
    failures = total - successes

    failure_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for t in traces:
        if not t.get("success", False):
            ft = normalize_failure_type(t.get("failure_type"))
            failure_counts[ft] = failure_counts.get(ft, 0) + 1
            cat = failure_category(ft)
            category_counts[cat] = category_counts.get(cat, 0) + 1

    return {
        "total": total,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / max(total, 1),
        "failure_counts": failure_counts,
        "category_counts": category_counts,
    }
