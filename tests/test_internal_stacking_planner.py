from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.planners.internal_stacking_planner import InternalStackingPlanner


def test_internal_stacking_planner_outputs_expected_moves():
    result = InternalStackingPlanner().solve(
        cubes=["cube1", "cube2", "cube3"],
        target_stack=["cube3", "cube2", "cube1"],
    )
    assert result.success, result.error
    assert [(action.name, action.args) for action in result.actions] == [
        ("move", ["cube2", "table", "cube3"]),
        ("move", ["cube1", "table", "cube2"]),
    ]


def test_planner_adapter_internal_stacking():
    result = PlannerAdapter("internal_stacking").solve(
        cubes=["cube1", "cube2", "cube3"],
        target_stack=["cube3", "cube2", "cube1"],
    )
    assert result.success
    assert len(result.actions) == 2
