"""Tests for TerminationFailureMonitor and StepSignal."""

from vlm_nsm.core.data_types import StepSignal
from vlm_nsm.embodied.termination_failure_monitor import (
    TerminationFailureMonitor,
)


def test_step_signal_defaults():
    signal = StepSignal()
    assert signal.done is False
    assert signal.success is False
    assert signal.failed is False
    assert signal.timeout is False
    assert signal.failure_type is None


def test_step_signal_failure_type():
    signal = StepSignal(done=True, failed=True, failure_type="grasp_failed")
    assert signal.done is True
    assert signal.failed is True
    assert signal.failure_type == "grasp_failed"


def test_monitor_timeout():
    monitor = TerminationFailureMonitor(step_timeout=10)
    signal = monitor.check("reach_pick", {}, 10)
    assert signal.timeout is True
    assert signal.failed is True
    assert signal.done is True
    assert signal.failure_type == "timeout"


def test_monitor_step_success():
    monitor = TerminationFailureMonitor(step_timeout=200)
    signal = monitor.check("reach_pick", {"gripper_has_object": False}, 1)
    assert signal.success is True
    assert signal.failed is False
    assert signal.done is False


def test_monitor_grasp_failure():
    """Simulate grasp that didn't lift the object."""
    monitor = TerminationFailureMonitor(step_timeout=200)
    # Record initial z at grasp time
    grasp_obs = {
        "grasped_object": "cube1",
        "cube1_pos": [0.0, 0.0, 0.02],
    }
    monitor.check("grasp", grasp_obs, 1)
    # Lift with same z → grasp_failed
    lift_obs = {
        "grasped_object": "cube1",
        "cube1_pos": [0.0, 0.0, 0.02],  # Same z, didn't lift
    }
    signal = monitor.check("lift", lift_obs, 50)
    assert signal.failed is True
    assert signal.failure_type == "grasp_failed"


def test_monitor_grasp_success():
    """Simulate successful lift."""
    monitor = TerminationFailureMonitor(step_timeout=200)
    grasp_obs = {
        "grasped_object": "cube1",
        "cube1_pos": [0.0, 0.0, 0.02],
    }
    monitor.check("grasp", grasp_obs, 1)
    lift_obs = {
        "grasped_object": "cube1",
        "cube1_pos": [0.0, 0.0, 0.10],  # Higher z → lifted
    }
    signal = monitor.check("lift", lift_obs, 50)
    assert signal.success is True
    assert signal.failed is False


def test_monitor_place_failure():
    """Simulate gripper still holding after release."""
    monitor = TerminationFailureMonitor(step_timeout=200)
    obs = {"gripper_has_object": True}
    signal = monitor.check("release", obs, 50)
    assert signal.failed is True
    assert signal.failure_type == "place_failed"
