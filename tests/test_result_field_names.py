from vlm_nsm.experiments.run_hanoi_symbolic import run_hanoi_pipeline
from vlm_nsm.experiments.run_stacking_symbolic import run_stacking_pipeline


def test_hanoi_result_fields():
    result = run_hanoi_pipeline(n_blocks=3, write_result=False)
    assert "atomic_action_length" in result
    assert "logical_plan_length" in result
    assert "optimal_logical_length" in result
    assert "plan_optimality_ratio" in result
    assert result["atomic_action_length"] == 14
    assert result["logical_plan_length"] == 7
    assert result["optimal_logical_length"] == 7
    assert result["plan_optimality_ratio"] == 1.0


def test_hanoi_4_result_fields():
    result = run_hanoi_pipeline(n_blocks=4, write_result=False)
    assert result["atomic_action_length"] == 30
    assert result["logical_plan_length"] == 15
    assert result["optimal_logical_length"] == 15
    assert result["plan_optimality_ratio"] == 1.0


def test_hanoi_5_result_fields():
    result = run_hanoi_pipeline(n_blocks=5, write_result=False)
    assert result["atomic_action_length"] == 62
    assert result["logical_plan_length"] == 31
    assert result["optimal_logical_length"] == 31
    assert result["plan_optimality_ratio"] == 1.0


def test_stacking_result_fields():
    result = run_stacking_pipeline(n_blocks=3, write_result=False)
    assert "atomic_action_length" in result
    assert "logical_plan_length" in result
    assert "optimal_logical_length" in result
    assert "plan_optimality_ratio" in result
    assert result["atomic_action_length"] == 2
    assert result["logical_plan_length"] == 2
    assert result["optimal_logical_length"] == 2
    assert result["plan_optimality_ratio"] == 1.0
