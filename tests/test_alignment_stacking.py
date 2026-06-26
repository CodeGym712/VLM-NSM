"""Test dynamic stacking completeness in AlignmentLayer.

Verifies that both orderings (cube1→cube2 and cube2→cube1) pass,
and that various invalid chains are correctly rejected.
"""
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import Predicate


TYPE_TABLE = {"cube1": "block", "cube2": "block", "table": "table"}


def _check(goals, n_blocks=2):
    """Helper: run stacking completeness check and return (ok, missing)."""
    ok, missing, expected = AlignmentLayer()._check_stacking_completeness(
        goals, TYPE_TABLE, n_blocks
    )
    return ok, missing, expected


def test_standard_ordering():
    """on(cube1, cube2), on(cube2, table) — original fixed order."""
    goals = [
        Predicate("on", ["cube1", "cube2"]),
        Predicate("on", ["cube2", "table"]),
    ]
    ok, missing, _ = _check(goals)
    assert ok is True
    assert missing == []


def test_reverse_ordering():
    """on(cube2, cube1), on(cube1, table) — reversed order."""
    goals = [
        Predicate("on", ["cube2", "cube1"]),
        Predicate("on", ["cube1", "table"]),
    ]
    ok, missing, _ = _check(goals)
    assert ok is True
    assert missing == []


def test_three_block_cube1_top():
    """cube1 on cube2, cube2 on cube3, cube3 on table."""
    tt = {"cube1": "block", "cube2": "block", "cube3": "block", "table": "table"}
    goals = [
        Predicate("on", ["cube1", "cube2"]),
        Predicate("on", ["cube2", "cube3"]),
        Predicate("on", ["cube3", "table"]),
    ]
    ok, missing, _ = AlignmentLayer()._check_stacking_completeness(goals, tt, 3)
    assert ok is True
    assert missing == []


def test_three_block_cube3_top():
    """cube3 on cube2, cube2 on cube1, cube1 on table."""
    tt = {"cube1": "block", "cube2": "block", "cube3": "block", "table": "table"}
    goals = [
        Predicate("on", ["cube3", "cube2"]),
        Predicate("on", ["cube2", "cube1"]),
        Predicate("on", ["cube1", "table"]),
    ]
    ok, missing, _ = AlignmentLayer()._check_stacking_completeness(goals, tt, 3)
    assert ok is True
    assert missing == []


def test_only_one_block():
    """on(cube1, table) → valid single-block stack."""
    goals = [Predicate("on", ["cube1", "table"])]
    ok, missing, _ = _check(goals, n_blocks=1)
    assert ok is True
    assert missing == []


def test_missing_on_table_raises():
    """No block on table → incomplete."""
    goals = [Predicate("on", ["cube1", "cube2"])]
    ok, missing, _ = _check(goals)
    assert ok is False


def test_disconnected_block_raises():
    """cube1 on table, cube2 not in chain."""
    goals = [Predicate("on", ["cube1", "table"])]
    ok, missing, _ = _check(goals, n_blocks=2)
    assert ok is False


def test_duplicate_support_raises():
    """cube1 on both cube2 and table — impossible."""
    goals = [
        Predicate("on", ["cube1", "cube2"]),
        Predicate("on", ["cube1", "table"]),
    ]
    ok, missing, _ = _check(goals)
    assert ok is False


def test_two_blocks_on_table_raises():
    """Two stacks — incomplete single stack."""
    goals = [
        Predicate("on", ["cube1", "table"]),
        Predicate("on", ["cube2", "table"]),
    ]
    ok, missing, _ = _check(goals)
    assert ok is False


def test_cycle_raises():
    """cube1 on cube2, cube2 on cube1 — cycle."""
    goals = [
        Predicate("on", ["cube1", "cube2"]),
        Predicate("on", ["cube2", "cube1"]),
    ]
    ok, missing, _ = _check(goals)
    assert ok is False


def test_non_on_predicates_ignored():
    """clear, free-gripper etc. should not affect stacking check."""
    goals = [
        Predicate("on", ["cube1", "cube2"]),
        Predicate("on", ["cube2", "table"]),
        Predicate("clear", ["cube1"]),
        Predicate("free-gripper", []),
    ]
    ok, missing, _ = _check(goals)
    assert ok is True
    assert missing == []


def test_no_blocks_returns_true():
    ok, missing, _ = AlignmentLayer()._check_stacking_completeness(
        [], {}, None
    )
    assert ok is True
