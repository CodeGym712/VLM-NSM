from vlm_nsm.core.data_types import AlignmentResult, Predicate
from vlm_nsm.experiments.eval_online_qwen_parser import parser_target_stack_from_alignment


def test_parser_target_stack_from_alignment_uses_aligned_predicate_order():
    align_result = AlignmentResult(
        success=True,
        goal_predicates=[
            Predicate("on", ["cube1", "cube2"]),
            Predicate("on", ["cube2", "table"]),
        ],
    )

    assert parser_target_stack_from_alignment(align_result, ["cube1", "cube2"]) == [
        "cube2",
        "cube1",
    ]


def test_parser_target_stack_from_alignment_rejects_disconnected_goal():
    align_result = AlignmentResult(
        success=True,
        goal_predicates=[
            Predicate("on", ["cube1", "table"]),
            Predicate("on", ["cube2", "table"]),
        ],
    )

    try:
        parser_target_stack_from_alignment(align_result, ["cube1", "cube2"])
    except ValueError as exc:
        assert "Multiple blocks on table" in str(exc)
    else:
        raise AssertionError("expected disconnected stack to be rejected")
