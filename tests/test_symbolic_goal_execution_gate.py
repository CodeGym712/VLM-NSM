"""Offline symbolic-goal validation tests — no robosuite, no Qwen.

Validates that unsafe/invalid symbolic goals are blocked before they
reach the controller.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Minimal symbolic goal validator (mirrors the real pipeline gate logic)
# ---------------------------------------------------------------------------

class SymbolicGoalValidator:
    """Validates symbolic goals without robosuite or LLM dependencies."""

    def __init__(self, available_objects: list[str] | None = None):
        self.available_objects = available_objects or ["cube1", "cube2"]

    def validate(self, goal: dict) -> dict:
        """Validate a symbolic goal dict.

        Returns a dict with:
          - allowed: bool
          - reason: str (why blocked, or "ok" if allowed)
          - checks: dict of individual check results
        """
        checks = {}
        reasons: list[str] = []

        # Check 1: Schema
        checks["has_required_keys"] = all(
            k in goal for k in ["task_type", "object_bindings", "goal_predicates"]
        )
        if not checks["has_required_keys"]:
            reasons.append("missing required keys (task_type, object_bindings, goal_predicates)")
            return {"allowed": False, "reason": "; ".join(reasons), "checks": checks}

        # Check 2: Ungroundable status
        status = goal.get("status", "ok")
        checks["status_not_ungroundable"] = status != "ungroundable"
        if not checks["status_not_ungroundable"]:
            reasons.append(f"parser status is 'ungroundable': {goal.get('reason', 'no reason given')}")

        # Check 3: Valid goal_predicates structure
        predicates = goal.get("goal_predicates", [])
        checks["predicates_is_list"] = isinstance(predicates, list)
        checks["predicates_non_empty"] = len(predicates) > 0
        if not checks["predicates_non_empty"]:
            reasons.append("goal_predicates is empty")

        # Check 4: Each predicate is [name, arg1, arg2]
        valid_preds = True
        for p in predicates:
            if not isinstance(p, list) or len(p) < 2:
                valid_preds = False
                break
        checks["predicates_well_formed"] = valid_preds
        if not valid_preds:
            reasons.append("malformed predicate (expected [name, arg1, arg2])")

        # Check 5: Object IDs are valid
        all_objects: set[str] = set()
        for p in predicates:
            if isinstance(p, list) and len(p) >= 2:
                for arg in p[1:]:
                    if arg != "table":
                        all_objects.add(str(arg).lower())
        unknown = all_objects - set(self.available_objects)
        checks["all_objects_known"] = len(unknown) == 0
        if not checks["all_objects_known"]:
            reasons.append(f"unknown objects: {sorted(unknown)}")

        # Check 6: on-predicates only (no unsupported relations)
        supported = {"on"}
        unsupported = set()
        for p in predicates:
            if isinstance(p, list) and len(p) >= 1:
                name = str(p[0]).lower()
                if name not in supported:
                    unsupported.add(name)
        checks["all_relations_supported"] = len(unsupported) == 0
        if not checks["all_relations_supported"]:
            reasons.append(f"unsupported relations: {sorted(unsupported)}")

        # Check 7: No self-referential on(block, block)
        has_self_ref = False
        for p in predicates:
            if isinstance(p, list) and len(p) >= 3:
                name = str(p[0]).lower()
                if name == "on" and str(p[1]).lower() == str(p[2]).lower():
                    has_self_ref = True
                    break
        checks["no_self_reference"] = not has_self_ref
        if has_self_ref:
            reasons.append("self-referential on predicate (top == bottom)")

        # Check 8: Ambiguous reference rejected
        object_bindings = goal.get("object_bindings", {})
        top = object_bindings.get("top block", "")
        bottom = object_bindings.get("bottom block", "")
        checks["distinct_top_bottom"] = top != bottom or (top == "" and bottom == "")
        if not checks["distinct_top_bottom"]:
            reasons.append(f"top and bottom are the same object: {top}")

        allowed = len(reasons) == 0
        return {
            "allowed": allowed,
            "reason": "; ".join(reasons) if reasons else "ok",
            "checks": checks,
        }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

VALID_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
    "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]],
    "status": "ok",
}

UNGROUNDABLE_GOAL = {
    "task_type": "stacking",
    "object_bindings": {},
    "goal_predicates": [],
    "status": "ungroundable",
    "reason": "cannot resolve 'the cube on the table' — both cubes are on table",
}

MISSING_OBJECT_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube3", "bottom block": "cube2"},
    "goal_predicates": [["on", "cube3", "cube2"], ["on", "cube2", "table"]],
    "status": "ok",
}

SAME_TOP_BOTTOM_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube1"},
    "goal_predicates": [["on", "cube1", "cube1"]],
    "status": "ok",
}

INVALID_RELATION_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
    "goal_predicates": [["next_to", "cube1", "cube2"]],
    "status": "ok",
}

AMBIGUOUS_REFERENCE_GOAL = {
    "task_type": "stacking",
    "object_bindings": {},
    "goal_predicates": [],
    "status": "ungroundable",
    "reason": "ambiguous reference: multiple cubes share same predicate",
}

MALFORMED_PREDICATE_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1"},
    "goal_predicates": [["on"]],
    "status": "ok",
}

SELF_REFERENCE_GOAL = {
    "task_type": "stacking",
    "object_bindings": {"top block": "cube1", "bottom block": "cube1"},
    "goal_predicates": [["on", "cube1", "cube1"], ["on", "cube1", "table"]],
    "status": "ok",
}


class TestSymbolicGoalExecutionGate:
    """Test that unsafe/invalid symbolic goals are blocked before execution."""

    def setup_method(self):
        self.validator = SymbolicGoalValidator(available_objects=["cube1", "cube2"])

    # ---- Positive cases ----

    def test_valid_goal_allowed(self):
        """Parser output status='ok', top=red(cube1), bottom=green(cube2) -> allowed."""
        result = self.validator.validate(VALID_GOAL)
        assert result["allowed"], f"Valid goal blocked: {result['reason']}"
        assert result["checks"]["status_not_ungroundable"]
        assert result["checks"]["all_objects_known"]
        assert result["checks"]["distinct_top_bottom"]

    # ---- Negative cases ----

    def test_ungroundable_blocked(self):
        """Parser output status='ungroundable' -> execution blocked."""
        result = self.validator.validate(UNGROUNDABLE_GOAL)
        assert not result["allowed"], "Ungroundable goal should be blocked"
        assert not result["checks"]["status_not_ungroundable"]
        assert "ungroundable" in result["reason"].lower()

    def test_missing_object_blocked(self):
        """Parser references cube3 which is not in available_objects -> blocked."""
        result = self.validator.validate(MISSING_OBJECT_GOAL)
        assert not result["allowed"], "Goal with missing object should be blocked"
        assert not result["checks"]["all_objects_known"]
        assert "cube3" in result["reason"]

    def test_same_top_bottom_blocked(self):
        """Parser has same top and bottom -> blocked."""
        result = self.validator.validate(SAME_TOP_BOTTOM_GOAL)
        assert not result["allowed"], "Goal with same top/bottom should be blocked"
        assert not result["checks"]["no_self_reference"]
        assert "self-referential" in result["reason"]

    def test_invalid_relation_blocked(self):
        """Parser has unsupported relation 'next_to' -> blocked."""
        result = self.validator.validate(INVALID_RELATION_GOAL)
        assert not result["allowed"], "Goal with invalid relation should be blocked"
        assert not result["checks"]["all_relations_supported"]
        assert "next_to" in result["reason"]

    def test_ambiguous_reference_blocked(self):
        """Parser output from ambiguous reference -> blocked."""
        result = self.validator.validate(AMBIGUOUS_REFERENCE_GOAL)
        assert not result["allowed"], "Ambiguous reference should be blocked"
        assert not result["checks"]["status_not_ungroundable"]

    # ---- Edge cases ----

    def test_empty_predicates_blocked(self):
        """Empty goal_predicates -> blocked."""
        result = self.validator.validate({
            "task_type": "stacking",
            "object_bindings": {},
            "goal_predicates": [],
            "status": "ok",
        })
        assert not result["allowed"]
        assert not result["checks"]["predicates_non_empty"]

    def test_missing_keys_blocked(self):
        """Goal missing required keys -> blocked."""
        result = self.validator.validate({"task_type": "stacking"})
        assert not result["allowed"]
        assert not result["checks"]["has_required_keys"]

    def test_malformed_predicate_blocked(self):
        """Predicate with only one element -> blocked."""
        result = self.validator.validate(MALFORMED_PREDICATE_GOAL)
        assert not result["allowed"]

    def test_unknown_object_in_predicate_blocked(self):
        """Predicate references unknown object -> blocked."""
        result = self.validator.validate({
            "task_type": "stacking",
            "object_bindings": {"top block": "cube5", "bottom block": "cube2"},
            "goal_predicates": [["on", "cube5", "cube2"], ["on", "cube2", "table"]],
            "status": "ok",
        })
        assert not result["allowed"]
        assert "cube5" in result["reason"]

    def test_table_reference_allowed_in_predicate(self):
        """Table is a valid support in predicates."""
        result = self.validator.validate({
            "task_type": "stacking",
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
            "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]],
            "status": "ok",
        })
        assert result["allowed"]

    def test_single_object_scene_stacking_self_blocked(self):
        """Stacking with single available object and on(obj, obj) -> blocked."""
        validator = SymbolicGoalValidator(available_objects=["cube1"])
        result = validator.validate({
            "task_type": "stacking",
            "object_bindings": {"top block": "cube1", "bottom block": "cube1"},
            "goal_predicates": [["on", "cube1", "cube1"]],
            "status": "ok",
        })
        assert not result["allowed"]

    def test_validator_returns_all_checks(self):
        """Validator result includes check-level details."""
        result = self.validator.validate(VALID_GOAL)
        assert "checks" in result
        assert "status_not_ungroundable" in result["checks"]
        assert "all_objects_known" in result["checks"]
        assert "no_self_reference" in result["checks"]
