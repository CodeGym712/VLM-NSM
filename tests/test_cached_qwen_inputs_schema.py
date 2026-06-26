"""Verify cached Qwen input file schema and parser."""

import json
from pathlib import Path

INPUT_PATH = Path("generated/results/stage_12/cached_qwen_e2e_inputs.json")


def test_cached_qwen_inputs_exist():
    assert INPUT_PATH.exists(), f"{INPUT_PATH} not found"


def test_cached_qwen_inputs_valid_json():
    data = json.loads(INPUT_PATH.read_text())
    assert isinstance(data, dict), "Expected JSON object"


def test_cached_qwen_inputs_all_instructions_have_goal_json():
    data = json.loads(INPUT_PATH.read_text())
    for inst, goal in data.items():
        assert isinstance(inst, str) and len(inst) > 0, f"Invalid instruction key: {inst}"
        assert "task_type" in goal, f"Missing task_type for: {inst[:50]}"
        assert "object_bindings" in goal, f"Missing object_bindings for: {inst[:50]}"
        assert "goal_predicates" in goal, f"Missing goal_predicates for: {inst[:50]}"


def test_cached_qwen_inputs_at_least_4_entries():
    data = json.loads(INPUT_PATH.read_text())
    assert len(data) >= 4, f"Only {len(data)} entries (need ≥4)"


def test_cached_qwen_parser_loads_all():
    from vlm_nsm.experiments.run_e2e_language_to_sim import CachedQwenGoalParser
    parser = CachedQwenGoalParser(str(INPUT_PATH))
    data = json.loads(INPUT_PATH.read_text())
    for inst in data:
        result = parser.parse(instruction=inst, task_type="stacking", objects=["cube1", "cube2"], n_blocks=2)
        assert result.success, f"Parse failed for: {inst[:50]}"
        assert result.goal_json is not None


def test_cached_qwen_parser_unknown_instruction():
    from vlm_nsm.experiments.run_e2e_language_to_sim import CachedQwenGoalParser
    parser = CachedQwenGoalParser(str(INPUT_PATH))
    result = parser.parse(instruction="nonexistent instruction", task_type="stacking", objects=["cube1"], n_blocks=1)
    assert not result.success
    assert result.error is not None
