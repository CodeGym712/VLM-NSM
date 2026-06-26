from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT


def test_hanoi_prompt_exists():
    path = PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt"
    assert path.exists(), f"missing prompt: {path}"
    content = path.read_text()
    assert "{instruction}" in content
    assert "{objects}" in content
    assert "{valid_predicates}" in content
    assert "{n_blocks}" in content
    assert "{target_peg}" in content
    assert "goal_predicates" in content


def test_stacking_prompt_exists():
    path = PROJECT_ROOT / "prompts" / "stacking_goal_parser.txt"
    assert path.exists(), f"missing prompt: {path}"
    content = path.read_text()
    assert "{instruction}" in content
    assert "{objects}" in content
    assert "{valid_predicates}" in content
    assert "{n_blocks}" in content
    assert "goal_predicates" in content


def test_hanoi_prompt_instructs_no_pddl_output():
    """The prompt should instruct the model not to output PDDL or action plans."""
    path = PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt"
    content = path.read_text().lower()
    assert "do not output pddl" in content or "no pddl" in content


def test_stacking_prompt_instructs_no_pddl_output():
    path = PROJECT_ROOT / "prompts" / "stacking_goal_parser.txt"
    content = path.read_text().lower()
    assert "do not output pddl" in content or "no pddl" in content


def test_hanoi_prompt_requires_json_only():
    path = PROJECT_ROOT / "prompts" / "hanoi_goal_parser.txt"
    content = path.read_text()
    assert "Only output" in content or "only" in content.lower()
    assert "JSON" in content


def test_language_variation_datasets_exist():
    for task in ["hanoi", "stacking"]:
        path = PROJECT_ROOT / "data" / "language_variations" / f"{task}_zh_en.jsonl"
        assert path.exists(), f"missing dataset: {path}"
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 7, f"{path} should have at least 7 samples"
