"""Test RobosuiteScriptedController action format.

Verifies the controller produces 7-dim OSC_POSE actions
with correct gripper states per skill step.
Does NOT require robosuite.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_nsm.embodied.controllers.robosuite_scripted_controller import (
    RobosuiteScriptedController,
)


@pytest.fixture
def controller():
    ctrl = RobosuiteScriptedController()
    ctrl.reset()
    return ctrl


class TestActionDimension:
    """Controller should always output 7-dim actions."""

    def test_action_dim_default(self, controller):
        assert controller.action_dim == 7

    def test_act_returns_7dim(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        assert action.shape == (7,), f"Expected (7,), got {action.shape}"
        assert action.dtype == np.float32

    def test_all_step_names_return_7dim(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        for step in ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]:
            action = controller.act(step, np.zeros(7, dtype=np.float32))
            assert action.shape == (7,), f"Step {step}: expected (7,), got {action.shape}"


class TestGripperState:
    """Gripper should follow expected open/close pattern."""

    def test_reach_pick_opens_gripper(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        # gripper open = -1.0
        assert action[6] == -1.0, f"Expected open (-1.0), got {action[6]}"

    def test_grasp_closes_gripper(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("grasp", np.zeros(7, dtype=np.float32))
        assert action[6] == 1.0, f"Expected close (1.0), got {action[6]}"

    def test_lift_keeps_closed(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("lift", np.zeros(7, dtype=np.float32))
        assert action[6] == 1.0, f"Expected closed (1.0), got {action[6]}"

    def test_reach_place_keeps_closed(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("reach_place", np.zeros(7, dtype=np.float32))
        assert action[6] == 1.0, f"Expected closed (1.0), got {action[6]}"

    def test_release_opens_gripper(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("release", np.zeros(7, dtype=np.float32))
        assert action[6] == -1.0, f"Expected open (-1.0), got {action[6]}"

    def test_retreat_keeps_open(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("retreat", np.zeros(7, dtype=np.float32))
        assert action[6] == -1.0, f"Expected open (-1.0), got {action[6]}"


class TestActionStructure:
    """Action should have correct structure: [dx, dy, dz, 0, 0, 0, gripper]."""

    def test_rotation_always_zero(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        for step in ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]:
            action = controller.act(step, np.zeros(7, dtype=np.float32))
            assert action[3] == 0.0, f"Step {step}: droll not zero"
            assert action[4] == 0.0, f"Step {step}: dpitch not zero"
            assert action[5] == 0.0, f"Step {step}: dyaw not zero"

    def test_position_delta_within_bounds(self, controller):
        controller.set_target(pos={"x": 10.0, "y": 10.0, "z": 10.0})  # far target
        controller.set_current_pos(pos={"x": 0.0, "y": 0.0, "z": 0.0})
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        # max_delta = 0.02m -> normalized: 0.02/0.05 = 0.4
        assert abs(action[0]) <= 0.4, f"dx clipped: {action[0]}"
        assert abs(action[1]) <= 0.4, f"dy clipped: {action[1]}"
        assert abs(action[2]) <= 0.4, f"dz clipped: {action[2]}"

    def test_action_in_normalized_range(self, controller):
        """Actions should be in [-1, 1] (robosuite 1.5.2 normalized)."""
        controller.set_target(pos={"x": 10.0, "y": 10.0, "z": 10.0})
        controller.set_current_pos(pos={"x": 0.0, "y": 0.0, "z": 0.0})
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        for i in range(7):
            assert -1.0 <= action[i] <= 1.0, f"action[{i}]={action[i]} out of [-1, 1]"

    def test_set_target_dict_and_array(self, controller):
        """set_target should accept both dict and array."""
        controller.set_target(pos={"x": 0.5, "y": 0.1, "z": 0.3})
        controller.set_current_pos(pos={"x": 0.0, "y": 0.0, "z": 0.0})
        action_dict = controller.act("reach_pick", np.zeros(7, dtype=np.float32))

        controller.reset()
        controller.set_target(pos=np.array([0.5, 0.1, 0.3]))
        controller.set_current_pos(pos=np.array([0.0, 0.0, 0.0]))
        action_arr = controller.act("reach_pick", np.zeros(7, dtype=np.float32))

        np.testing.assert_array_almost_equal(action_dict[:3], action_arr[:3])


class TestReset:
    """Reset should clear controller state."""

    def test_reset_clears_target(self, controller):
        controller.set_target(pos={"x": 0.5, "y": 0.0, "z": 0.3})
        controller.reset()
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        np.testing.assert_array_equal(action[:3], [0.0, 0.0, 0.0])

    def test_reset_resets_step_idx(self, controller):
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        controller.act("grasp", np.zeros(7, dtype=np.float32))
        controller.reset()
        controller.set_target(pos={"x": 0.3, "y": 0.0, "z": 0.2})
        action = controller.act("reach_pick", np.zeros(7, dtype=np.float32))
        assert action[6] == -1.0  # should start from reach_pick
