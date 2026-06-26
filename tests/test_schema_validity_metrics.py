"""Test schema_validity logic in run_language_variation."""

from vlm_nsm.experiments.run_language_variation import _check_schema_validity


def test_schema_validity_valid_goal():
    goal = {"goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "peg3"]]}
    assert _check_schema_validity(goal) is True


def test_schema_validity_empty_predicates():
    goal = {"goal_predicates": []}
    assert _check_schema_validity(goal) is False


def test_schema_validity_missing_goal_predicates():
    goal = {"task_type": "hanoi"}
    assert _check_schema_validity(goal) is False


def test_schema_validity_none_goal_json():
    assert _check_schema_validity(None) is False


def test_schema_validity_not_list():
    goal = {"goal_predicates": "not_a_list"}
    assert _check_schema_validity(goal) is False


def test_schema_validity_predicate_not_list():
    goal = {"goal_predicates": ["not_a_list"]}
    assert _check_schema_validity(goal) is False


def test_schema_validity_predicate_empty():
    goal = {"goal_predicates": [[]]}
    assert _check_schema_validity(goal) is False


def test_schema_validity_predicate_name_not_str():
    goal = {"goal_predicates": [[123, "cube1"]]}
    assert _check_schema_validity(goal) is False


def test_schema_validity_single_predicate():
    goal = {"goal_predicates": [["on", "cube1", "peg3"]]}
    assert _check_schema_validity(goal) is True
