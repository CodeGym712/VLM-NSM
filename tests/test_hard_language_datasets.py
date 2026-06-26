"""Test hard language variation datasets exist and have correct format."""

from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import load_instructions


def test_hanoi_hard_dataset_exists():
    path = PROJECT_ROOT / "data" / "language_variations" / "hanoi_hard_zh_en.jsonl"
    assert path.exists(), f"Missing {path}"
    samples = load_instructions(str(path))
    assert len(samples) >= 20, f"Only {len(samples)} samples, need >= 20"


def test_hanoi_hard_dataset_fields():
    path = PROJECT_ROOT / "data" / "language_variations" / "hanoi_hard_zh_en.jsonl"
    samples = load_instructions(str(path))
    required = {"id", "task", "n_blocks", "instruction", "category", "difficulty", "expected_goal"}
    for s in samples:
        missing = required - set(s.keys())
        assert not missing, f"{s.get('id', '?')} missing fields: {missing}"
        assert s["task"] == "hanoi", f"{s['id']} wrong task"
        assert s["difficulty"] == "hard", f"{s['id']} wrong difficulty"
        assert isinstance(s["expected_goal"], list), f"{s['id']} expected_goal not list"


def test_stacking_hard_dataset_exists():
    path = PROJECT_ROOT / "data" / "language_variations" / "stacking_hard_zh_en.jsonl"
    assert path.exists(), f"Missing {path}"
    samples = load_instructions(str(path))
    assert len(samples) >= 20, f"Only {len(samples)} samples, need >= 20"


def test_stacking_hard_dataset_fields():
    path = PROJECT_ROOT / "data" / "language_variations" / "stacking_hard_zh_en.jsonl"
    samples = load_instructions(str(path))
    required = {"id", "task", "n_blocks", "instruction", "category", "difficulty", "expected_goal"}
    for s in samples:
        missing = required - set(s.keys())
        assert not missing, f"{s.get('id', '?')} missing fields: {missing}"
        assert s["task"] == "stacking", f"{s['id']} wrong task"
        assert s["difficulty"] == "hard", f"{s['id']} wrong difficulty"


def test_hanoi_hard_expected_goal_chain():
    """Verify that each hard sample's expected_goal is a valid chain ending at peg3."""
    path = PROJECT_ROOT / "data" / "language_variations" / "hanoi_hard_zh_en.jsonl"
    samples = load_instructions(str(path))
    for s in samples:
        goal = s["expected_goal"]
        assert len(goal) == 3, f"{s['id']}: expected 3 predicates, got {len(goal)}"
        assert goal[0] == ["on", "cube1", "cube2"], f"{s['id']}: {goal[0]}"
        assert goal[1] == ["on", "cube2", "cube3"], f"{s['id']}: {goal[1]}"
        assert goal[2] == ["on", "cube3", "peg3"], f"{s['id']}: {goal[2]}"


def test_stacking_hard_expected_goal_chain():
    """Verify that each hard sample's expected_goal ends at table."""
    path = PROJECT_ROOT / "data" / "language_variations" / "stacking_hard_zh_en.jsonl"
    samples = load_instructions(str(path))
    for s in samples:
        goal = s["expected_goal"]
        assert len(goal) == 3, f"{s['id']}: expected 3 predicates, got {len(goal)}"
        assert goal[0] == ["on", "cube1", "cube2"], f"{s['id']}: {goal[0]}"
        assert goal[1] == ["on", "cube2", "cube3"], f"{s['id']}: {goal[1]}"
        assert goal[2] == ["on", "cube3", "table"], f"{s['id']}: {goal[2]}"
