"""Test medium prompt files exist and have expected properties."""

from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import _resolve_prompt_path


def test_medium_prompt_exists_hanoi():
    path = PROJECT_ROOT / "prompts" / "hanoi_goal_parser_medium.txt"
    assert path.exists(), f"Missing {path}"
    content = path.read_text()
    assert "{instruction}" in content
    assert "{objects}" in content
    assert "{valid_predicates}" in content
    assert "{n_blocks}" in content
    assert "{target_peg}" in content
    assert "goal_predicates" in content


def test_medium_prompt_exists_stacking():
    path = PROJECT_ROOT / "prompts" / "stacking_goal_parser_medium.txt"
    assert path.exists(), f"Missing {path}"
    content = path.read_text()
    assert "{instruction}" in content
    assert "{objects}" in content
    assert "{valid_predicates}" in content
    assert "{n_blocks}" in content
    assert "goal_predicates" in content


def test_medium_prompt_length_between_weak_and_full():
    for task in ("hanoi", "stacking"):
        full_path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser.txt"
        medium_path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_medium.txt"
        weak_path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_weak.txt"

        full_len = len(full_path.read_text(encoding="utf-8"))
        medium_len = len(medium_path.read_text(encoding="utf-8"))
        weak_len = len(weak_path.read_text(encoding="utf-8"))

        assert weak_len < medium_len < full_len, (
            f"Medium prompt ({medium_len} chars) should be between "
            f"weak ({weak_len}) and full ({full_len}) for {task}"
        )


def test_medium_prompt_contains_json_schema():
    for task in ("hanoi", "stacking"):
        path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_medium.txt"
        content = path.read_text()
        assert "goal_predicates" in content


def test_medium_prompt_no_canonical_example():
    """Medium prompt should NOT contain complete canonical example chain."""
    for task in ("hanoi", "stacking"):
        path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_medium.txt"
        content = path.read_text()
        # Medium prompt should not have the full PDDL-style example chain
        # that the full prompt has (like ["on", "cube1", "cube2"])
        assert "do not output pddl" in content.lower()


def test_medium_prompt_no_pddl_instruction():
    """Medium prompt should not mention PDDL restrictions."""
    for task in ("hanoi", "stacking"):
        path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_medium.txt"
        content = path.read_text().lower()
        # The medium prompt should say not to output PDDL
        # but should NOT contain full canonical example chains
        assert "do not output pddl" in content or "no pddl" in content


def test_resolve_prompt_medium():
    path = _resolve_prompt_path("hanoi", "medium", None)
    assert path is not None
    assert path.endswith("hanoi_goal_parser_medium.txt")


def test_resolve_prompt_stacking_medium():
    path = _resolve_prompt_path("stacking", "medium", None)
    assert path is not None
    assert path.endswith("stacking_goal_parser_medium.txt")
