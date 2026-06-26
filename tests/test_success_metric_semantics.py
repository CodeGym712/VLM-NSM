"""Tests for strict vs semantic success metric semantics (no robosuite dependency)."""
from __future__ import annotations

from vlm_nsm.core.metrics import reality_gap


def test_strict_vs_semantic_concept():
    """Verify the concept: strict counts only clean success, semantic counts goal achievement."""
    # Scenario: 20 episodes, 18 goal success, but 2 of those had timeout_after_goal
    # strict success = 16 (clean execution + goal)
    # semantic success = 18 (goal achieved regardless)
    n_symbolic = 20
    n_strict = 16
    n_semantic = 18

    strict_rg = reality_gap(n_symbolic, n_strict)
    semantic_rg = reality_gap(n_symbolic, n_semantic)

    assert strict_rg == 20.0  # (20-16)/20 * 100
    assert semantic_rg == 10.0  # (20-18)/20 * 100
    assert strict_rg > semantic_rg  # strict should always be >= semantic gap


def test_strict_equals_semantic_when_no_timeout_after_goal():
    """When no timeout_after_goal occurs, strict and semantic are equal."""
    n_symbolic = 10
    n_strict = 9
    n_semantic = 9
    assert reality_gap(n_symbolic, n_strict) == reality_gap(n_symbolic, n_semantic)


def test_semantic_perfect_no_gap():
    """If all episodes achieve goal semantically, semantic RG is 0%."""
    assert reality_gap(20, 20) == 0.0


def test_strict_perfect_no_gap():
    """If all episodes achieve goal with clean execution, strict RG is 0%."""
    assert reality_gap(20, 20) == 0.0


def test_all_fail_strict():
    """If no episode has clean execution+goal, strict RG is 100%."""
    assert reality_gap(10, 0) == 100.0


def test_reality_gap_zero_symbolic():
    """Edge case: no symbolic successes returns None."""
    assert reality_gap(0, 0) is None
