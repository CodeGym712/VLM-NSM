"""Tests for timeout_after_goal detection logic (no robosuite dependency)."""
from __future__ import annotations


def test_timeout_after_goal_classification():
    """Verify timeout_after_goal vs timeout_before_goal classification logic.

    The pipeline classifies:
      - timeout_before_goal: episode hit horizon, goal NOT achieved
      - timeout_after_goal:  episode hit horizon, goal WAS achieved
      - None: episode did not hit horizon
    """
    # Simulate the classification logic from _run_robosuite_episode
    def classify(horizon: int, total_steps: int, goal_ok: bool, goal_was_achieved: bool, early_stopped: bool):
        hit_horizon = total_steps >= horizon - 5
        if hit_horizon and not early_stopped:
            if goal_was_achieved or goal_ok:
                return "timeout_after_goal"
            else:
                return "timeout_before_goal"
        return None

    assert classify(1000, 995, True, True, False) == "timeout_after_goal"
    assert classify(1000, 998, False, False, False) == "timeout_before_goal"
    assert classify(1000, 500, True, True, False) is None  # no horizon hit
    assert classify(1000, 990, True, True, True) is None  # early stopped, not timeou

