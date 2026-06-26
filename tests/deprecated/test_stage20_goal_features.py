import numpy as np
import pytest

from vlm_nsm.data.stage20_goal_features import (
    compute_goal_delta,
    compute_goal_error,
    compute_placement_bin,
    compute_progress_scalar,
    compute_reached_goal_delta,
    compute_relative_pose,
    compute_stalled_steps,
)


def test_compute_relative_pose_returns_xyz_difference():
    out = compute_relative_pose([1, 2, 3], [0.5, 1, 1.5])
    np.testing.assert_allclose(out, [0.5, 1.0, 1.5])


def test_compute_relative_pose_tolerates_orientation_fields():
    out = compute_relative_pose(
        {"x": 1, "y": 2, "z": 3, "quat": [0, 0, 0, 1]},
        [0, 1, 2, 0, 0, 0, 1],
    )
    np.testing.assert_allclose(out, [1, 1, 1])


@pytest.mark.parametrize(
    ("top", "expected"),
    [
        ([0.0, 0.0, 0.0], "center"),
        ([0.0, -0.1, 0.0], "front"),
        ([0.0, 0.1, 0.0], "back"),
        ([-0.1, 0.0, 0.0], "left"),
        ([0.1, 0.0, 0.0], "right"),
        ([0.1, -0.1, 0.0], "front-right"),
        ([-0.1, 0.1, 0.0], "back-left"),
    ],
)
def test_compute_placement_bin(top, expected):
    assert compute_placement_bin(top, [0, 0, 0]) == expected


def test_compute_goal_delta_reach_pick_points_gripper_to_target():
    out = compute_goal_delta(
        "reach_pick",
        {"gripper_pose": [0, 0, 0]},
        {"target_gripper_pose": [1, 2, 3]},
    )
    np.testing.assert_allclose(out, [1, 2, 3])


def test_compute_goal_delta_lift_returns_target_minus_current_z():
    out = compute_goal_delta(
        "lift",
        {"top_pose": [0, 0, 0.8]},
        {"target_top_z": 1.0},
    )
    np.testing.assert_allclose(out, [0.2])


def test_compute_goal_delta_place_returns_target_minus_top_pose():
    out = compute_goal_delta(
        "place",
        {"top_pose": [0.1, 0.2, 0.8]},
        {"target_top_pose": [0.0, 0.0, 0.9]},
    )
    np.testing.assert_allclose(out, [-0.1, -0.2, 0.1])


def test_compute_reached_goal_delta_returns_current_minus_start():
    out = compute_reached_goal_delta(
        {"gripper_pose": [0, 0, 0], "gripper_open": 1.0},
        {"gripper_pose": [1, 0, 0.5], "gripper_open": 0.5},
    )
    np.testing.assert_allclose(out["gripper_pose"], [1, 0, 0.5])
    np.testing.assert_allclose(out["gripper_open"], -0.5)


def test_compute_goal_error_is_finite_norm():
    assert compute_goal_error([3, 4, np.nan]) == 5.0


def test_compute_progress_scalar_increases_as_error_decreases():
    p1 = compute_progress_scalar(10, 8)
    p2 = compute_progress_scalar(10, 2)
    assert p2 > p1


def test_compute_progress_scalar_clips_to_unit_interval():
    assert compute_progress_scalar(10, -1) == 1.0
    assert compute_progress_scalar(10, 20) == 0.0


def test_compute_stalled_steps_counts_low_change_tail():
    assert compute_stalled_steps([1.0, 0.8, 0.7001, 0.7000, 0.6999], tolerance=1e-3) == 3


def test_unsupported_phase_raises_value_error():
    with pytest.raises(ValueError):
        compute_goal_delta("teleport", {}, {})


def test_missing_optional_orientation_does_not_crash():
    out = compute_goal_delta(
        "retreat",
        {"gripper_pose": {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"target_gripper_pose": {"x": 0.0, "y": 0.0, "z": 0.2, "quat": [0, 0, 0, 1]}},
    )
    np.testing.assert_allclose(out, [0, 0, 0.2])
