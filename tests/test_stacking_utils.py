"""Test target_stack_from_predicates for parser-driven stacking planning."""
import pytest
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates


def test_basic_two_block():
    """on(cube1, cube2), on(cube2, table) → [cube2, cube1]."""
    result = target_stack_from_predicates(
        [("on", "cube1", "cube2"), ("on", "cube2", "table")],
        blocks=["cube1", "cube2"],
    )
    assert result == ["cube2", "cube1"]


def test_reverse_ordering():
    """on(cube2, cube1), on(cube1, table) → [cube1, cube2].

    Dynamic chain validation must support any block ordering.
    """
    result = target_stack_from_predicates(
        [("on", "cube2", "cube1"), ("on", "cube1", "table")],
        blocks=["cube1", "cube2"],
    )
    assert result == ["cube1", "cube2"]


def test_three_blocks():
    result = target_stack_from_predicates(
        [
            ("on", "cube1", "cube2"),
            ("on", "cube2", "cube3"),
            ("on", "cube3", "table"),
        ],
        blocks=["cube1", "cube2", "cube3"],
    )
    assert result == ["cube3", "cube2", "cube1"]


def test_three_blocks_reverse():
    """cube3 on cube2, cube2 on cube1, cube1 on table."""
    result = target_stack_from_predicates(
        [
            ("on", "cube3", "cube2"),
            ("on", "cube2", "cube1"),
            ("on", "cube1", "table"),
        ],
        blocks=["cube1", "cube2", "cube3"],
    )
    assert result == ["cube1", "cube2", "cube3"]


def test_unknown_object_raises():
    with pytest.raises(ValueError, match="Unknown"):
        target_stack_from_predicates(
            [("on", "cube1", "cube2")],
            blocks=["cube1"],
        )


def test_unknown_upper_block_raises():
    with pytest.raises(ValueError, match="Unknown object"):
        target_stack_from_predicates(
            [("on", "cube3", "cube1")],
            blocks=["cube1", "cube2"],
        )


def test_unknown_support_raises():
    with pytest.raises(ValueError, match="Unknown support"):
        target_stack_from_predicates(
            [("on", "cube1", "peg1")],
            blocks=["cube1", "cube2"],
        )


def test_duplicate_support_raises():
    with pytest.raises(ValueError, match="Duplicate support"):
        target_stack_from_predicates(
            [("on", "cube1", "cube2"), ("on", "cube1", "table")],
            blocks=["cube1", "cube2"],
        )


def test_cycle_raises():
    with pytest.raises(ValueError, match="Cycle"):
        target_stack_from_predicates(
            [("on", "cube1", "cube2"), ("on", "cube2", "cube1")],
            blocks=["cube1", "cube2"],
        )


def test_no_block_on_table_raises():
    with pytest.raises(ValueError, match="No block is on table"):
        target_stack_from_predicates(
            [("on", "cube1", "cube2")],
            blocks=["cube1", "cube2"],
        )


def test_multiple_on_table_raises():
    with pytest.raises(ValueError, match="Multiple blocks on table"):
        target_stack_from_predicates(
            [("on", "cube1", "table"), ("on", "cube2", "table")],
            blocks=["cube1", "cube2"],
        )


def test_disconnected_chain_raises():
    with pytest.raises(ValueError, match="not in chain"):
        target_stack_from_predicates(
            [("on", "cube1", "table")],
            blocks=["cube1", "cube2"],
        )


def test_no_on_predicates_raises():
    with pytest.raises(ValueError, match="No block is on table"):
        target_stack_from_predicates(
            [],
            blocks=["cube1", "cube2"],
        )


def test_require_table_support_false():
    """Without table requirement, single on relation builds chain."""
    result = target_stack_from_predicates(
        [("on", "cube1", "cube2")],
        blocks=["cube1", "cube2"],
        require_table_support=False,
    )
    assert result == ["cube2", "cube1"]


def test_only_on_predicates_processed():
    """Non-on predicates are ignored."""
    result = target_stack_from_predicates(
        [
            ("on", "cube1", "cube2"),
            ("on", "cube2", "table"),
            ("clear", "cube1"),
            ("free-gripper",),
        ],
        blocks=["cube1", "cube2"],
    )
    assert result == ["cube2", "cube1"]
