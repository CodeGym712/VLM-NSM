"""Test prompt variant selection and existence."""

from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import _resolve_prompt_path


def test_full_prompt_exists():
    for task in ("hanoi", "stacking"):
        path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser.txt"
        assert path.exists(), f"Missing {path}"


def test_weak_prompt_exists():
    for task in ("hanoi", "stacking"):
        path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_weak.txt"
        assert path.exists(), f"Missing {path}"


def test_weak_prompt_shorter_than_full():
    for task in ("hanoi", "stacking"):
        full_path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser.txt"
        weak_path = PROJECT_ROOT / "prompts" / f"{task}_goal_parser_weak.txt"
        full_len = len(full_path.read_text(encoding="utf-8"))
        weak_len = len(weak_path.read_text(encoding="utf-8"))
        assert weak_len < full_len, (
            f"Weak prompt ({weak_len} chars) should be shorter than "
            f"full prompt ({full_len} chars) for {task}"
        )


def test_resolve_prompt_full():
    path = _resolve_prompt_path("hanoi", "full", None)
    assert path is not None
    assert path.endswith("hanoi_goal_parser.txt")


def test_resolve_prompt_medium():
    path = _resolve_prompt_path("hanoi", "medium", None)
    assert path is not None
    assert path.endswith("hanoi_goal_parser_medium.txt")


def test_resolve_prompt_stacking_medium():
    path = _resolve_prompt_path("stacking", "medium", None)
    assert path is not None
    assert path.endswith("stacking_goal_parser_medium.txt")


def test_resolve_prompt_weak():
    path = _resolve_prompt_path("hanoi", "weak", None)
    assert path is not None
    assert path.endswith("hanoi_goal_parser_weak.txt")


def test_resolve_prompt_override():
    path = _resolve_prompt_path("hanoi", "full", "/custom/prompt.txt")
    assert path == "/custom/prompt.txt"


def test_resolve_prompt_stacking_weak():
    path = _resolve_prompt_path("stacking", "weak", None)
    assert path is not None
    assert path.endswith("stacking_goal_parser_weak.txt")
