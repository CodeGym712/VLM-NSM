from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.experiments.run_hanoi_symbolic import predicate_signature


def base():
    objects = ["cube1", "cube2", "cube3", "peg1", "peg2", "peg3"]
    types = {"cube1": "disk", "cube2": "disk", "cube3": "disk", "peg1": "peg", "peg2": "peg", "peg3": "peg"}
    preds = ["on", "clear", "smaller", "free-gripper", "grasped"]
    return objects, types, preds


def test_aliases_and_repair():
    objects, types, preds = base()
    goal_json = {"task_type": "hanoi", "object_bindings": {}, "goal_predicates": [["above", "cube1", "cube2"], ["on", "cube2", "cube3"], ["on", "cube3", "peg3"]]}
    result = AlignmentLayer().align_and_validate(goal_json, objects, preds, types, predicate_signature(), "hanoi", target_peg="peg3", n_blocks=3)
    assert result.success
    assert result.repaired
    assert result.goal_predicates[0].name == "on"


def test_handempty_alias():
    objects, types, preds = base()
    goal_json = {"task_type": "hanoi", "object_bindings": {}, "goal_predicates": [["handempty"], ["on", "cube1", "cube2"], ["on", "cube2", "cube3"], ["on", "cube3", "peg3"]]}
    result = AlignmentLayer().align_and_validate(goal_json, objects, preds, types, predicate_signature(), "hanoi", target_peg="peg3", n_blocks=3)
    assert result.success
    assert any(p.name == "free-gripper" for p in result.goal_predicates)


def test_invalid_object_rejected():
    objects, types, preds = base()
    goal_json = {"task_type": "hanoi", "object_bindings": {}, "goal_predicates": [["on", "cube9", "cube2"]]}
    result = AlignmentLayer().align_and_validate(goal_json, objects, preds, types, predicate_signature(), "hanoi", target_peg="peg3", n_blocks=3)
    assert not result.success
    assert result.error == "invalid_object"


def test_invalid_predicate_rejected():
    objects, types, preds = base()
    goal_json = {"task_type": "hanoi", "object_bindings": {}, "goal_predicates": [["touching", "cube1", "cube2"]]}
    result = AlignmentLayer().align_and_validate(goal_json, objects, preds, types, predicate_signature(), "hanoi", target_peg="peg3", n_blocks=3)
    assert not result.success
    assert result.error == "invalid_predicate"


def test_incomplete_goal_detected():
    objects, types, preds = base()
    goal_json = {"task_type": "hanoi", "object_bindings": {}, "goal_predicates": [["on", "cube1", "cube2"]]}
    result = AlignmentLayer().align_and_validate(goal_json, objects, preds, types, predicate_signature(), "hanoi", target_peg="peg3", n_blocks=3)
    assert not result.success
    assert result.error == "incomplete_goal"
