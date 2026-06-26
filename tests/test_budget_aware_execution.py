"""Tests for budget-aware execution helpers (no robosuite dependency)."""
from __future__ import annotations

from vlm_nsm.experiments.run_embodied_stacking_robosuite import (
    _remaining_budget,
    _MIN_BUDGET_REACH,
    _MIN_BUDGET_GRASP,
    _MIN_BUDGET_LIFT,
    _MIN_BUDGET_REACH_PLACE,
    _MIN_BUDGET_RELEASE,
    _MIN_BUDGET_RETREAT,
)


def test_remaining_budget_positive():
    assert _remaining_budget(100, 1000) == 900


def test_remaining_budget_exact():
    assert _remaining_budget(1000, 1000) == 0


def test_remaining_budget_exceeded():
    assert _remaining_budget(1100, 1000) == 0


def test_remaining_budget_zero_horizon():
    assert _remaining_budget(0, 0) == 0


def test_min_budget_constants_positive():
    assert _MIN_BUDGET_REACH > 0
    assert _MIN_BUDGET_GRASP > 0
    assert _MIN_BUDGET_LIFT > 0
    assert _MIN_BUDGET_REACH_PLACE > 0
    assert _MIN_BUDGET_RELEASE > 0
    assert _MIN_BUDGET_RETREAT > 0


def test_min_budget_release_at_least_release_steps():
    """release budget must cover RELEASE_OPEN_STEPS + margin."""
    from vlm_nsm.experiments.run_embodied_stacking_robosuite import _RELEASE_OPEN_STEPS
    assert _MIN_BUDGET_RELEASE >= _RELEASE_OPEN_STEPS + 5
