from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.experiments.run_alignment_ablation import (
    SYNTHETIC_GOALS,
    _hanoi_objects,
    _hanoi_types,
    _hanoi_signature,
    _stacking_objects,
    _stacking_types,
    _stacking_signature,
    run_single_alignment,
)

VALID_PREDICATES = ["on", "clear", "smaller", "free-gripper", "grasped"]


def test_invalid_object_detected():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [["on", "blockA", "cube2"]],
    })
    assert not result["object_validity"]
    assert result["error_type"] == "invalid_object"


def test_invalid_predicate_detected():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [["stacked_on", "cube1", "cube2"]],
    })
    assert not result["predicate_validity"]
    assert result["error_type"] == "invalid_predicate"


def test_type_error_detected():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [["on", "peg1", "cube1"]],
    })
    assert not result["type_check_pass"]


def test_incomplete_hanoi_goal():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [["on", "cube1", "cube2"]],
    })
    assert not result["goal_completeness"]
    assert result["error_type"] == "incomplete_goal"


def test_incomplete_stacking_goal():
    result = run_single_alignment("stacking", 3, {
        "task_type": "stacking",
        "object_bindings": {},
        "goal_predicates": [["on", "cube1", "cube2"]],
    })
    assert not result["goal_completeness"]
    assert result["error_type"] == "incomplete_goal"


def test_alias_repair():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["above", "cube1", "cube2"],
            ["on", "cube2", "cube3"],
            ["on", "cube3", "peg3"],
        ],
    })
    assert result["repaired"]
    assert "predicate_alias" in result["repair_types"]
    assert result["planner_success"]
    assert result["goal_success"]


def test_valid_hanoi_passes():
    result = run_single_alignment("hanoi", 3, {
        "task_type": "hanoi",
        "object_bindings": {},
        "goal_predicates": [
            ["on", "cube1", "cube2"],
            ["on", "cube2", "cube3"],
            ["on", "cube3", "peg3"],
        ],
    })
    assert result["object_validity"]
    assert result["predicate_validity"]
    assert result["type_check_pass"]
    assert result["goal_completeness"]
    assert result["planner_success"]
    assert result["goal_success"]
