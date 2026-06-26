from vlm_nsm.baselines.fixed_goal_nsm import (
    fixed_hanoi_goal,
    fixed_stacking_goal,
    run_fixed_goal_baseline,
)


def test_fixed_hanoi_goal_3():
    goals = fixed_hanoi_goal(3)
    assert len(goals) == 3
    assert goals[0].name == "on" and goals[0].args == ["cube1", "cube2"]
    assert goals[1].name == "on" and goals[1].args == ["cube2", "cube3"]
    assert goals[2].name == "on" and goals[2].args == ["cube3", "peg3"]


def test_fixed_stacking_goal_3():
    goals = fixed_stacking_goal(3)
    assert len(goals) == 3
    assert goals[0].name == "on" and goals[0].args == ["cube1", "cube2"]
    assert goals[1].name == "on" and goals[1].args == ["cube2", "cube3"]
    assert goals[2].name == "on" and goals[2].args == ["cube3", "table"]


def test_fixed_hanoi_baseline_runs():
    result = run_fixed_goal_baseline("hanoi", 3)
    assert result["planner_success"]
    assert result["execution_success"]
    assert result["goal_success"]
    assert result["action_count"] == 14


def test_fixed_stacking_baseline_runs():
    result = run_fixed_goal_baseline("stacking", 3)
    assert result["planner_success"]
    assert result["execution_success"]
    assert result["goal_success"]
    assert result["action_count"] == 2
