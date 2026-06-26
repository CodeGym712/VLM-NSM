import pytest

from vlm_nsm.core.goal_parser import extract_json_object, parse_json_from_text


def test_extract_pure_json():
    text = '{"task_type": "hanoi", "goal_predicates": []}'
    assert extract_json_object(text)["task_type"] == "hanoi"
    assert parse_json_from_text(text)["task_type"] == "hanoi"


def test_extract_markdown_json_block():
    text = '```json\n{"task_type": "hanoi", "goal_predicates": [["on", "cube1", "cube2"]]}\n```'
    assert extract_json_object(text)["goal_predicates"][0] == ["on", "cube1", "cube2"]


def test_extract_json_with_explanation_text():
    text = 'Here is the goal:\n{"task_type": "hanoi", "object_bindings": {}, "goal_predicates": []}\nDone.'
    assert extract_json_object(text)["object_bindings"] == {}


def test_extract_json_with_code_block_no_lang():
    text = '```\n{"task_type": "stacking", "goal_predicates": []}\n```'
    assert extract_json_object(text)["task_type"] == "stacking"


def test_extract_json_with_extra_whitespace():
    text = '  \n  {"task_type": "hanoi", "goal_predicates": []}  \n'
    assert extract_json_object(text)["task_type"] == "hanoi"


def test_extract_json_from_markdown_with_text_around():
    text = """I will parse the goal for you.
```json
{"task_type": "hanoi", "goal_predicates": [["on", "cube1", "cube2"]]}
```
This is the final goal."""
    result = extract_json_object(text)
    assert result["goal_predicates"][0] == ["on", "cube1", "cube2"]


def test_extract_nested_json():
    text = '{"a": {"b": ["c", "d"]}, "e": 1}'
    assert extract_json_object(text)["a"]["b"] == ["c", "d"]


def test_invalid_json_raises():
    with pytest.raises(Exception):
        extract_json_object("not json at all")


def test_incomplete_json_raises():
    with pytest.raises(Exception):
        extract_json_object('{"task_type": "hanoi"')


def test_extract_json_from_multi_json():
    """Should extract the first JSON object when multiple are present."""
    text = '{"first": true}{"second": true}'
    assert extract_json_object(text)["first"] is True
