"""Test alignment modes with mock/synthetic goals."""

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import Predicate
from vlm_nsm.experiments.run_language_variation import _predicates_from_goal_json


def _hanoi_signature():
    return {
        "on": [["disk", "disk|peg"]],
        "clear": [["disk|peg"]],
        "smaller": [["disk", "disk|peg"]],
        "free-gripper": [[]],
        "grasped": [["disk"]],
    }


def _hanoi_types():
    return {"cube1": "disk", "cube2": "disk", "cube3": "disk",
            "peg1": "peg", "peg2": "peg", "peg3": "peg"}


def test_no_alignment_skips_validation():
    """no_alignment mode should accept raw predicates without checks."""
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["on", "cube1", "cube2"],
            ["on", "cube2", "cube3"],
            ["on", "cube3", "peg3"],
        ],
    }
    preds = _predicates_from_goal_json(goal_json)
    assert len(preds) == 3
    assert all(isinstance(p, Predicate) for p in preds)


def test_no_alignment_invalid_object_no_error():
    """no_alignment should not reject invalid objects."""
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [["on", "cube1", "nonexistent_object"]],
    }
    preds = _predicates_from_goal_json(goal_json)
    assert len(preds) == 1


def test_alignment_mode_full_accepts_valid():
    aligner = AlignmentLayer()
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {"smallest disk": "cube1"},
        "goal_predicates": [
            ["on", "cube1", "cube2"],
            ["on", "cube2", "cube3"],
            ["on", "cube3", "peg3"],
        ],
    }
    result = aligner.align_and_validate(
        goal_json, list(_hanoi_types().keys()),
        ["on", "clear", "smaller", "free-gripper", "grasped"],
        _hanoi_types(), _hanoi_signature(),
        task_type="hanoi", target_peg="peg3", n_blocks=3,
    )
    assert result.success
    assert not result.repaired


def test_alignment_mode_no_alias_repair():
    """above should NOT be repaired when using empty aliases."""
    aligner = AlignmentLayer(aliases={})
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["above", "cube1", "cube2"],
        ],
    }
    result = aligner.align_and_validate(
        goal_json, list(_hanoi_types().keys()),
        ["on", "clear", "smaller", "free-gripper", "grasped"],
        _hanoi_types(), _hanoi_signature(),
        task_type="hanoi", target_peg="peg3", n_blocks=3,
    )
    # "above" is not a valid predicate and should fail
    assert not result.success
    assert result.error == "invalid_predicate"


def test_alignment_mode_no_completeness():
    """Without completeness check, partial goal can pass alignment."""
    aligner = AlignmentLayer()
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["on", "cube1", "cube2"],
        ],
    }
    # Pass n_blocks=None and empty target_peg to skip completeness check
    result = aligner.align_and_validate(
        goal_json, list(_hanoi_types().keys()),
        ["on", "clear", "smaller", "free-gripper", "grasped"],
        _hanoi_types(), _hanoi_signature(),
        task_type="hanoi", target_peg="", n_blocks=None,
    )
    assert result.success


def test_alignment_mode_no_repair_still_validates():
    """no_repair mode (empty aliases, no autocomplete) should validate objects."""
    aligner = AlignmentLayer(aliases={})
    goal_json = {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["on", "cube1", "invalid_obj"],
        ],
    }
    result = aligner.align_and_validate(
        goal_json, list(_hanoi_types().keys()),
        ["on", "clear", "smaller", "free-gripper", "grasped"],
        _hanoi_types(), _hanoi_signature(),
        task_type="hanoi", target_peg="peg3", n_blocks=3,
    )
    assert not result.success
    assert result.error == "invalid_object"
