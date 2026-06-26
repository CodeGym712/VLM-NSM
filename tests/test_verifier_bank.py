"""Tests for Verifier Bank — Commit 3."""

import pytest
from vlm_nsm.neuro_symbolic.verifier_bank import (
    GoalVerifier, SubgoalVerifier, PlanVerifier,
    OptionContractVerifier, RepairVerifier,
    VerifierBank, VerificationReport, standard_verifier_bank,
)
from vlm_nsm.core.option_contract import get_contract
from vlm_nsm.symbolic.predicates import SymbolicSceneGraph, on, ontable, clear, handempty, graspable


class TestGoalVerifier:
    def test_valid_goal(self):
        gv = GoalVerifier(["cube1", "cube2", "table"])
        r = gv.check({"goal_predicates": [["on", "cube1", "cube2"]]})
        assert r.passed

    def test_none_goal(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check(None)
        assert not r.passed
        assert "None" in r.failures[0][1]

    def test_empty_predicates(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check({"goal_predicates": []})
        assert not r.passed

    def test_unknown_predicate(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check({"goal_predicates": [["inside", "cube1", "cube2"]]})
        assert not r.passed
        assert any("unknown predicate" in f[1] for f in r.failures)

    def test_unknown_object(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check({"goal_predicates": [["on", "cube99", "cube2"]]})
        assert not r.passed
        assert any("unknown object" in f[1] for f in r.failures)

    def test_self_reference(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check({"goal_predicates": [["on", "cube1", "cube1"]]})
        assert not r.passed
        assert any("self-reference" in f[1] for f in r.failures)

    def test_multiple_supports(self):
        gv = GoalVerifier(["cube1", "cube2", "cube3"])
        r = gv.check({"goal_predicates": [
            ["on", "cube1", "cube2"], ["on", "cube1", "cube3"]]})
        assert not r.passed
        assert any("multiple supports" in f[1] for f in r.failures)

    def test_cycle(self):
        gv = GoalVerifier(["cube1", "cube2"])
        r = gv.check({"goal_predicates": [
            ["on", "cube1", "cube2"], ["on", "cube2", "cube1"]]})
        assert not r.passed


class TestSubgoalVerifier:
    def test_empty(self):
        sv = SubgoalVerifier(["cube1", "cube2"])
        r = sv.check([])
        assert r.passed  # empty subgoals = warning, not failure

    def test_valid_subgoals(self):
        sv = SubgoalVerifier(["cube1", "cube2"])
        r = sv.check([[["on", "cube1", "cube2"]]])
        assert r.passed

    def test_conflicting_subgoals(self):
        sv = SubgoalVerifier(["cube1", "cube2", "cube3"])
        r = sv.check([
            [["on", "cube1", "cube2"]],
            [["on", "cube1", "cube3"]],
        ])
        # cube1 placed on cube2 then on cube3 — conflict on cube1's support
        assert not r.passed, f"Expected failure, got passed: {r}"


class TestPlanVerifier:
    def test_empty_plan(self):
        pv = PlanVerifier()
        r = pv.check([], None, [])
        assert not r.passed

    def test_invalid_action_format(self):
        pv = PlanVerifier()
        r = pv.check([42], None, [])
        assert not r.passed


class TestOptionContractVerifier:
    def test_valid_initiation(self):
        cv = OptionContractVerifier()
        contract = get_contract("reach")
        state = [["handempty"], ["graspable", "cube1"]]
        r = cv.check(contract, state, target_object="cube1")
        assert r.passed

    def test_missing_initiation(self):
        cv = OptionContractVerifier()
        contract = get_contract("grasp")
        state = [["handempty"]]  # missing clear(cube1), graspable(cube1)
        r = cv.check(contract, state, target_object="cube1")
        assert not r.passed


class TestRepairVerifier:
    def test_valid_repair(self):
        rv = RepairVerifier(["cube1", "cube2"], max_retries=3)
        r = rv.check({
            "repair_type": "retry_grasp",
            "modified_option": "grasp",
            "parameter_patch": {"target_object": "cube1"},
            "retry_budget": 1,
            "required_verification": ["GoalVerifier"],
        })
        assert r.passed

    def test_budget_exceeded(self):
        rv = RepairVerifier(["cube1", "cube2"], max_retries=3)
        r = rv.check({"repair_type": "retry", "retry_budget": 2}, retries_used=2)
        assert not r.passed

    def test_unknown_option(self):
        rv = RepairVerifier(["cube1", "cube2"])
        r = rv.check({"repair_type": "retry", "modified_option": "fly"})
        assert not r.passed


class TestVerifierBank:
    def test_create_and_run(self):
        bank = standard_verifier_bank(["cube1", "cube2", "table"])
        assert len(bank.verifier_names) == 5

        # Run with a valid goal
        r = bank.verify_all(
            goal_json={"goal_predicates": [["on", "cube1", "cube2"]]},
            task_type="stacking",
        )
        assert r.passed

    def test_bad_goal_caught(self):
        bank = standard_verifier_bank(["cube1", "cube2", "table"])
        r = bank.verify_all(
            goal_json={"goal_predicates": [["on", "cube1", "cube1"]]},
            task_type="stacking",
        )
        assert not r.passed


class TestVerificationReport:
    def test_merge(self):
        r1 = VerificationReport(verifier_name="A")
        r1.add_failure("check1", "bad")
        r2 = VerificationReport(verifier_name="B")
        r2.add_warning("warn1", "maybe bad")
        r1.merge(r2)
        assert not r1.passed
        assert len(r1.failures) == 1
        assert len(r1.warnings) == 1

    def test_explain(self):
        r = VerificationReport(verifier_name="Test")
        r.add_failure("check1", "something wrong")
        text = r.explain()
        assert "Test" in text
        assert "something wrong" in text
