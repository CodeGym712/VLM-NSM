"""Test online Qwen prompt content — no real API calls."""
from vlm_nsm.language.online_qwen_parser import STACKING_PROMPT


def test_prompt_contains_json_only_instruction():
    assert "Output ONLY valid JSON" in STACKING_PROMPT or "Output ONLY" in STACKING_PROMPT


def test_prompt_contains_valid_predicates():
    assert "on(object, object_or_table)" in STACKING_PROMPT


def test_prompt_contains_available_objects():
    assert "cube1" in STACKING_PROMPT and "cube2" in STACKING_PROMPT


def test_prompt_contains_table():
    assert "table" in STACKING_PROMPT


def test_prompt_no_invent_object_ids():
    assert "Do NOT invent object IDs" in STACKING_PROMPT


def test_prompt_contains_few_shot_examples():
    assert "Examples:" in STACKING_PROMPT
    assert "Instruction: \"Put the red cube" in STACKING_PROMPT
    assert "goal_predicates" in STACKING_PROMPT


def test_prompt_contains_output_schema():
    assert "goal_predicates" in STACKING_PROMPT
    assert "task_type" in STACKING_PROMPT
    assert "object_bindings" in STACKING_PROMPT


def test_prompt_contains_bottom_support_predicate():
    assert '["on", "cube2", "table"]' in STACKING_PROMPT


def test_prompt_mentions_completeness():
    text_lower = STACKING_PROMPT.lower()
    assert "bottom" in text_lower or "support" in text_lower


def test_prompt_has_no_explanation_requirement():
    assert "No explanation" in STACKING_PROMPT or "no explanation" in STACKING_PROMPT


def test_prompt_has_no_markdown():
    assert "no markdown" in STACKING_PROMPT or "no code fences" in STACKING_PROMPT
