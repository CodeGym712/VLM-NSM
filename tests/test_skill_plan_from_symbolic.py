import pytest

from vlm_nsm.core.data_types import Action, PlanResult
from vlm_nsm.skills.skill_plan_from_symbolic import skill_plan_from_symbolic


def test_single_move_to_pick_place():
    plan = skill_plan_from_symbolic([Action("move", ["cube1", "table", "cube2"])])

    assert len(plan.commands) == 1
    command = plan.commands[0]
    assert command.skill_name == "pick_place"
    assert command.object_id == "cube1"
    assert command.source == "table"
    assert command.target_object_id == "cube2"
    assert command.relation == "on"


def test_multi_move_to_multi_skill_plan():
    symbolic = [
        Action("move", ["cube2", "table", "cube3"]),
        Action("move", ["cube1", "table", "cube2"]),
    ]

    plan = skill_plan_from_symbolic(symbolic)

    assert [c.object_id for c in plan.commands] == ["cube2", "cube1"]
    assert [c.target_object_id for c in plan.commands] == ["cube3", "cube2"]
    assert plan.metadata["num_commands"] == 2


def test_plan_result_actions_are_supported():
    result = PlanResult(
        success=True,
        actions=[Action("move", ["cube1", "table", "cube2"])],
        raw_output="",
    )

    plan = skill_plan_from_symbolic(result)

    assert plan.commands[0].object_id == "cube1"
    assert plan.to_dict()["source_plan"] == [
        {"name": "move", "args": ["cube1", "table", "cube2"]}
    ]


def test_dict_and_tuple_action_forms_are_supported():
    dict_plan = skill_plan_from_symbolic([
        {"name": "move", "args": ["cube1", "table", "cube2"]}
    ])
    tuple_plan = skill_plan_from_symbolic([
        ("move", "cube1", "table", "cube2")
    ])

    assert dict_plan.commands[0].target_object_id == "cube2"
    assert tuple_plan.commands[0].target_object_id == "cube2"


def test_two_arg_move_to_table_is_valid():
    """Fix C3: move(obj, target) with 2 args is valid — places obj on target."""
    plan = skill_plan_from_symbolic([Action("move", ["cube1", "cube2"])])

    assert len(plan.commands) == 1
    command = plan.commands[0]
    assert command.object_id == "cube1"
    assert command.target_object_id == "cube2"
    assert command.relation == "on"


def test_move_to_table_action_supported():
    """Fix C3: move_to_table(x) from BFS SymbolicPlanner is supported."""
    plan = skill_plan_from_symbolic([Action("move_to_table", ["cube1"])])

    assert len(plan.commands) == 1
    assert plan.commands[0].object_id == "cube1"
    assert plan.commands[0].target_object_id == "table"
    assert plan.commands[0].relation == "table"


def test_move_on_action_supported():
    """Fix C3: move_on(x,y) from BFS SymbolicPlanner is supported."""
    plan = skill_plan_from_symbolic([Action("move_on", ["cube1", "cube2"])])

    assert len(plan.commands) == 1
    assert plan.commands[0].object_id == "cube1"
    assert plan.commands[0].target_object_id == "cube2"
    assert plan.commands[0].relation == "on"


def test_unsupported_action_raises():
    with pytest.raises(ValueError, match="Unsupported symbolic action"):
        skill_plan_from_symbolic([Action("fly", ["cube1"])])
