"""Test JSON extraction utility for online Qwen parser."""
import json
import pytest
from vlm_nsm.core.goal_parser import extract_json_object


def test_pure_json():
    result = extract_json_object('{"a": 1, "b": 2}')
    assert result == {"a": 1, "b": 2}


def test_fenced_json():
    text = '```json\n{"task": "stacking", "goal_predicates": []}\n```'
    result = extract_json_object(text)
    assert result == {"task": "stacking", "goal_predicates": []}


def test_text_before_json():
    text = 'Here is the result:\n{"task": "stacking"}\nEnd.'
    result = extract_json_object(text)
    assert result == {"task": "stacking"}


def test_fenced_without_marker():
    text = '```\n{"task": "hanoi"}\n```'
    result = extract_json_object(text)
    assert result == {"task": "hanoi"}


def test_nested_json():
    text = '{"task": "stacking", "predicates": [["on", "a", "b"]]}'
    result = extract_json_object(text)
    assert result == {"task": "stacking", "predicates": [["on", "a", "b"]]}


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("not json at all")


def test_unmatched_braces_raises():
    with pytest.raises(ValueError, match="incomplete JSON"):
        extract_json_object('{"unclosed": "object"')


def test_whitespace_handling():
    text = '  \n  {"key": "value"}  \n  '
    result = extract_json_object(text)
    assert result == {"key": "value"}


def test_multiple_json_objects():
    text = '{"first": 1} trailing {"second": 2}'
    result = extract_json_object(text)
    assert result == {"first": 1}


def test_array_not_object():
    with pytest.raises(ValueError):
        extract_json_object("[1, 2, 3]")


def test_online_qwen_typical_output():
    text = '{"task": "stacking", "goal_predicates": [["on", "cube1", "cube2"], ["on", "cube2", "table"]]}'
    result = extract_json_object(text)
    assert result["task"] == "stacking"
    assert len(result["goal_predicates"]) == 2
