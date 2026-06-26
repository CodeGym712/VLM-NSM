from vlm_nsm.baselines.llm_direct_plan import (
    run_direct_plan,
    oracle_hanoi_plan,
    oracle_stacking_plan,
    get_all_corrupted_plans,
)


def test_oracle_hanoi_plan_succeeds():
    actions = oracle_hanoi_plan(3)
    result = run_direct_plan("hanoi", 3, actions)
    assert result["execution_success"], result.get("error_type")
    assert result["goal_success"]
    assert result["plan_completeness"]


def test_oracle_stacking_plan_succeeds():
    actions = oracle_stacking_plan(3)
    result = run_direct_plan("stacking", 3, actions)
    assert result["execution_success"], result.get("error_type")
    assert result["goal_success"]
    assert result["plan_completeness"]


def test_corrupted_missing_action_fails_goal():
    corrupted = get_all_corrupted_plans(3, "stacking")
    actions = corrupted["missing_action"]
    result = run_direct_plan("stacking", 3, actions)
    assert not result["goal_success"]


def test_corrupted_invalid_object_detected():
    corrupted = get_all_corrupted_plans(3, "stacking")
    actions = corrupted["invalid_object"]
    result = run_direct_plan("stacking", 3, actions)
    assert not result["execution_success"]


def test_corrupted_invalid_action_format_detected():
    corrupted = get_all_corrupted_plans(3, "stacking")
    actions = corrupted["invalid_action_format"]
    result = run_direct_plan("stacking", 3, actions)
    assert not result["execution_success"]


def test_corrupted_precondition_violation_detected():
    corrupted = get_all_corrupted_plans(3, "hanoi")
    actions = corrupted["precondition_violation"]
    result = run_direct_plan("hanoi", 3, actions)
    assert not result["execution_success"]


def test_corrupted_wrong_order_detected():
    corrupted = get_all_corrupted_plans(3, "stacking")
    actions = corrupted["wrong_order"]
    result = run_direct_plan("stacking", 3, actions)
    assert not result["execution_success"]
