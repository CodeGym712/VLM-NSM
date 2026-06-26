from vlm_nsm.experiments.run_stacking_symbolic import run_stacking_pipeline


def test_stacking_3_symbolic():
    result = run_stacking_pipeline(n_blocks=3, write_result=False)
    assert result["planner_success"], result.get("planner_error")
    assert result["logical_plan_length"] == 2
    assert result["atomic_action_length"] == 2
    assert result["optimal_logical_length"] == 2
    assert result["plan_optimality_ratio"] == 1.0
    assert result["execution_success"]
    assert result["goal_success"]
    assert result["error_type"] == "none"


def test_stacking_4_symbolic():
    result = run_stacking_pipeline(n_blocks=4, write_result=False)
    assert result["planner_success"]
    assert result["logical_plan_length"] == 3
    assert result["atomic_action_length"] == 3
    assert result["optimal_logical_length"] == 3
    assert result["plan_optimality_ratio"] == 1.0
    assert result["execution_success"]
    assert result["goal_success"]
    assert result["error_type"] == "none"


def test_stacking_5_symbolic():
    result = run_stacking_pipeline(n_blocks=5, write_result=False)
    assert result["planner_success"]
    assert result["logical_plan_length"] == 4
    assert result["atomic_action_length"] == 4
    assert result["optimal_logical_length"] == 4
    assert result["plan_optimality_ratio"] == 1.0
    assert result["execution_success"]
    assert result["goal_success"]
    assert result["error_type"] == "none"
