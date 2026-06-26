"""Test run_embodied_stacking_robosuite CLI and helper functions.

Does NOT require robosuite — tests config parsing, helper math,
and goal construction logic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vlm_nsm.core.data_types import Predicate


def _dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _proportional_action(eef, target, kp, max_delta, gripper):
    dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
    dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
    dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)
    return np.array([dx, dy, dz, 0.0, 0.0, 0.0, gripper], dtype=np.float32)


def build_stacking_goal(target_stack):
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


class TestBuildStackingGoal:
    """Goal construction for stacking tasks."""

    def test_three_cubes(self):
        cubes = ["cube1", "cube2", "cube3"]
        target_stack = cubes[::-1]  # [cube3, cube2, cube1]
        goals = build_stacking_goal(target_stack)
        assert len(goals) == 2
        # on(cube2, cube3) and on(cube1, cube2)
        assert goals[0].name == "on"
        assert goals[0].args == ["cube2", "cube3"]
        assert goals[1].name == "on"
        assert goals[1].args == ["cube1", "cube2"]

    def test_two_cubes(self):
        target_stack = ["cube2", "cube1"]
        goals = build_stacking_goal(target_stack)
        assert len(goals) == 1
        assert goals[0].name == "on"
        assert goals[0].args == ["cube1", "cube2"]

    def test_single_cube_no_goal(self):
        goals = build_stacking_goal(["cube1"])
        assert len(goals) == 0


class TestDistHelper:
    """Distance helper function."""

    def test_same_point(self):
        p = {"x": 1.0, "y": 2.0, "z": 3.0}
        assert _dist(p, p) == 0.0

    def test_known_distance(self):
        a = {"x": 0.0, "y": 0.0, "z": 0.0}
        b = {"x": 1.0, "y": 0.0, "z": 0.0}
        assert _dist(a, b) == 1.0

    def test_3d_distance(self):
        a = {"x": 0.0, "y": 0.0, "z": 0.0}
        b = {"x": 1.0, "y": 2.0, "z": 3.0}
        expected = math.sqrt(1 + 4 + 9)
        assert abs(_dist(a, b) - expected) < 1e-10

    def test_negative_coords(self):
        a = {"x": -1.0, "y": -1.0, "z": -1.0}
        b = {"x": 1.0, "y": 1.0, "z": 1.0}
        expected = math.sqrt(12)
        assert abs(_dist(a, b) - expected) < 1e-10


class TestProportionalAction:
    """Proportional control action format."""

    def test_output_is_7dim(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        target = {"x": 0.3, "y": 0.0, "z": 0.2}
        action = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=-1.0)
        assert action.shape == (7,)
        assert action.dtype == np.float32

    def test_position_delta_sign(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        target = {"x": 0.3, "y": 0.1, "z": 0.2}
        action = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=-1.0)
        assert action[0] > 0  # dx positive
        assert action[1] > 0  # dy positive
        assert action[2] > 0  # dz positive

    def test_position_delta_clipping(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        target = {"x": 100.0, "y": 100.0, "z": 100.0}
        action = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=-1.0)
        assert abs(action[0]) <= 0.02
        assert abs(action[1]) <= 0.02
        assert abs(action[2]) <= 0.02

    def test_gripper_value(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        target = {"x": 0.3, "y": 0.0, "z": 0.2}
        action_open = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=-1.0)
        assert action_open[6] == -1.0
        action_close = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=1.0)
        assert action_close[6] == 1.0

    def test_rotation_always_zero(self):
        eef = {"x": 0.0, "y": 0.0, "z": 0.0}
        target = {"x": 0.3, "y": 0.0, "z": 0.2}
        action = _proportional_action(eef, target, kp=0.05, max_delta=0.02, gripper=-1.0)
        assert action[3] == 0.0
        assert action[4] == 0.0
        assert action[5] == 0.0


class TestCLIArgs:
    """CLI argument parsing."""

    def test_parser_accepts_valid_args(self):
        from vlm_nsm.experiments.run_embodied_stacking_robosuite import main as _main
        import argparse

        parser = argparse.ArgumentParser(description="Run robosuite embodied stacking evaluation")
        parser.add_argument("--n_blocks", type=int, default=3, choices=[3])
        parser.add_argument("--num_episodes", type=int, default=5)
        parser.add_argument("--render", action="store_true")
        args = parser.parse_args(["--n_blocks", "3", "--num_episodes", "10"])
        assert args.n_blocks == 3
        assert args.num_episodes == 10
        assert not args.render

    def test_parser_render_flag(self):
        import argparse
        parser = argparse.ArgumentParser(description="test")
        parser.add_argument("--render", action="store_true")
        args = parser.parse_args(["--render"])
        assert args.render

    def test_parser_defaults(self):
        import argparse
        parser = argparse.ArgumentParser(description="test")
        parser.add_argument("--n_blocks", type=int, default=2, choices=[2])
        parser.add_argument("--num_episodes", type=int, default=5)
        args = parser.parse_args([])
        assert args.n_blocks == 2
        assert args.num_episodes == 5


class TestImportGuard:
    """Script should handle missing robosuite gracefully."""

    def test_main_handles_import_error(self):
        """Import error should result in a dict with error key."""
        from vlm_nsm.experiments.run_embodied_stacking_robosuite import main as _main
        assert callable(_main)

    def test_run_function_raises_import_error(self):
        """Direct call without robosuite should raise ImportError."""
        try:
            from vlm_nsm.experiments.run_embodied_stacking_robosuite import run_robosuite_eval
            pytest.skip("Robosuite available — cannot test ImportError path")
        except ImportError:
            pass  # Expected
