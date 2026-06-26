"""Tests for OperatorMonitor."""

import pytest

from vlm_nsm.neuro_symbolic.operator_monitor import (
    OperatorMonitor, OperatorFailureType, OperatorCheckResult, OperatorEffectResult,
)
from vlm_nsm.skills.skill_types import SkillCommand


class TestOperatorMonitor:
    @pytest.fixture
    def monitor(self):
        return OperatorMonitor()

    @pytest.fixture
    def move_on_command(self):
        return SkillCommand(
            skill_name="pick_place",
            object_id="cube1",
            target_object_id="cube2",
            relation="on",
            preconditions=[
                {"name": "clear", "args": ["cube1"]},
                {"name": "clear", "args": ["cube2"]},
                {"name": "handempty", "args": []},
            ],
            expected_add_effects=[
                {"name": "on", "args": ["cube1", "cube2"]},
                {"name": "clear", "args": ["cube1"]},
                {"name": "handempty", "args": []},
            ],
        )

    def test_classify_nocontact(self, monitor, move_on_command):
        op_failure, recovery = monitor.classify_failure(move_on_command, "nocontact", None)
        assert op_failure == OperatorFailureType.PICK_NO_CONTACT
        assert recovery == "regrasp"

    def test_classify_preplace(self, monitor, move_on_command):
        op_failure, recovery = monitor.classify_failure(move_on_command, "preplace", None)
        assert op_failure == OperatorFailureType.PLACE_XY_MISALIGN
        assert recovery == "micro_place"

    def test_classify_preplace_with_not_holding_effect(self, monitor, move_on_command):
        effects = [
            OperatorEffectResult("on", ["cube1", "cube2"], False),
            OperatorEffectResult("holding", ["cube1"], False),
        ]
        op_failure, recovery = monitor.classify_failure(move_on_command, "preplace", effects)
        assert op_failure == OperatorFailureType.PLACE_NOT_HOLDING
        assert recovery == "re_pick"

    def test_classify_verify_failed(self, monitor, move_on_command):
        op_failure, recovery = monitor.classify_failure(move_on_command, "verify_failed", None)
        assert op_failure == OperatorFailureType.PLACE_VERIFY_FAILED

    def test_classify_horizon_preplace(self, monitor, move_on_command):
        op_failure, recovery = monitor.classify_failure(move_on_command, "horizon_preplace", None)
        assert op_failure == OperatorFailureType.PLACE_TIMEOUT
        assert recovery == "replan"

    def test_classify_none(self, monitor, move_on_command):
        op_failure, recovery = monitor.classify_failure(move_on_command, None, None)
        assert op_failure is None
        assert recovery is None

    def test_check_effects_satisfied(self, monitor, move_on_command):
        state = {("on", "cube1", "cube2"), ("clear", "cube1"), ("handempty",)}
        results = monitor.check_effects(move_on_command, state)
        satisfied = [r.satisfied for r in results]
        assert all(satisfied)

    def test_check_effects_not_satisfied(self, monitor, move_on_command):
        state = {("clear", "cube1"), ("handempty",)}  # missing on(cube1,cube2)
        results = monitor.check_effects(move_on_command, state)
        satisfied = [r.satisfied for r in results]
        assert not all(satisfied)

    def test_check_after_execution_success(self, monitor, move_on_command):
        state = {("on", "cube1", "cube2"), ("clear", "cube1"), ("handempty",)}
        result = monitor.check_after_execution(move_on_command, state, None)
        assert result.all_effects_satisfied
        assert result.operator_failure is None

    def test_check_after_execution_preplace(self, monitor, move_on_command):
        state = set()
        result = monitor.check_after_execution(move_on_command, state, "preplace")
        assert not result.all_effects_satisfied
        assert result.operator_failure is not None
        assert result.recovery_hint is not None

    def test_empty_command(self, monitor):
        cmd = SkillCommand(skill_name="pick_place", object_id="cube1")
        result = monitor.check_after_execution(cmd, None, None)
        assert result.all_effects_satisfied  # no effects → trivially satisfied

    def test_operator_check_result_to_dict(self):
        result = OperatorCheckResult(
            operator_name="pick_place",
            all_effects_satisfied=True,
        )
        d = result.to_dict()
        assert d["operator_name"] == "pick_place"
        assert d["all_effects_satisfied"] is True


class TestOperatorFailureType:
    def test_recovery_map_has_all(self):
        from vlm_nsm.neuro_symbolic.operator_monitor import _RECOVERY_MAP
        for key in [
            OperatorFailureType.PICK_NO_CONTACT,
            OperatorFailureType.PLACE_XY_MISALIGN,
            OperatorFailureType.PLACE_NOT_HOLDING,
        ]:
            assert key in _RECOVERY_MAP, f"Missing recovery for {key}"
