import json

from vlm_nsm.skills.skill_types import (
    SkillCommand,
    SkillExecutionResult,
    SkillPlan,
)


def test_skill_command_json_round_trip():
    command = SkillCommand(
        skill_name="pick_place",
        object_id="cube1",
        target_object_id="cube2",
        relation="on",
        source="table",
        metadata={"phase": "teacher"},
    )

    restored = SkillCommand.from_json(command.to_json())
    assert restored == command
    assert json.loads(command.to_json())["object_id"] == "cube1"


def test_skill_plan_json_round_trip():
    plan = SkillPlan(
        commands=[
            SkillCommand("pick_place", object_id="cube2", target_object_id="cube3", relation="on"),
            SkillCommand("pick_place", object_id="cube1", target_object_id="cube2", relation="on"),
        ],
        source_plan=[
            {"name": "move", "args": ["cube2", "table", "cube3"]},
            {"name": "move", "args": ["cube1", "table", "cube2"]},
        ],
        metadata={"task": "three_block_stack"},
    )

    restored = SkillPlan.from_json(plan.to_json())
    assert len(restored.commands) == 2
    assert restored.commands[0].object_id == "cube2"
    assert restored.commands[1].target_object_id == "cube2"
    assert restored.metadata["task"] == "three_block_stack"


def test_skill_execution_result_json_round_trip():
    result = SkillExecutionResult(
        success=True,
        skill_command=SkillCommand("pick_place", object_id="cube1", target_object_id="cube2"),
        steps=42,
        metrics={"bc_steps": 10},
    )

    restored = SkillExecutionResult.from_json(result.to_json())
    assert restored.success is True
    assert restored.steps == 42
    assert restored.metrics["bc_steps"] == 10
