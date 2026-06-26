"""Test hard language variation runs with mock parser."""

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import (
    load_instructions,
    run_single_sample,
)

RAW_OUTPUT_DIR = PROJECT_ROOT / "generated" / "logs" / "qwen_outputs"
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_hanoi_hard_mock_all_pass():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_hard_zh_en.jsonl")
    )
    assert len(samples) >= 20

    for sample in samples:
        result = run_single_sample(
            sample=sample,
            task="hanoi",
            n_blocks=3,
            parser_name="mock",
            model_path=None,
            prompt_path=None,
            temperature=0.0,
            max_new_tokens=128,
            raw_output_dir=RAW_OUTPUT_DIR,
            difficulty="hard",
            prompt_variant="full",
            alignment_mode="full",
        )
        assert result["json_parse_success"], f"{sample['id']}: parse failed"
        assert result["alignment_success"], f"{sample['id']}: alignment failed"
        assert result["planner_success"], f"{sample['id']}: planner failed"
        assert result["execution_success"], f"{sample['id']}: execution failed"
        assert result["goal_success"], f"{sample['id']}: goal not satisfied"
        # Check new fields are present
        assert "difficulty" in result, f"{sample['id']}: missing difficulty"
        assert "prompt_variant" in result, f"{sample['id']}: missing prompt_variant"
        assert "alignment_mode" in result, f"{sample['id']}: missing alignment_mode"
        assert "schema_validity" in result, f"{sample['id']}: missing schema_validity"
        assert "alignment_applicable" in result, f"{sample['id']}: missing alignment_applicable"
        assert result["schema_validity"], f"{sample['id']}: schema should be valid"


def test_stacking_hard_mock_all_pass():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "stacking_hard_zh_en.jsonl")
    )
    assert len(samples) >= 20

    for sample in samples:
        result = run_single_sample(
            sample=sample,
            task="stacking",
            n_blocks=3,
            parser_name="mock",
            model_path=None,
            prompt_path=None,
            temperature=0.0,
            max_new_tokens=128,
            raw_output_dir=RAW_OUTPUT_DIR,
            difficulty="hard",
            prompt_variant="full",
            alignment_mode="full",
        )
        assert result["json_parse_success"], f"{sample['id']}: parse failed"
        assert result["alignment_success"], f"{sample['id']}: alignment failed"
        assert result["planner_success"], f"{sample['id']}: planner failed"
        assert result["execution_success"], f"{sample['id']}: execution failed"
        assert result["goal_success"], f"{sample['id']}: goal not satisfied"
        assert result["difficulty"] == "hard"
        assert result["prompt_variant"] == "full"
        assert result["alignment_mode"] == "full"
        assert result["schema_validity"], f"{sample['id']}: schema should be valid"
        assert result["alignment_applicable"], f"{sample['id']}: alignment should be applicable"


def test_hanoi_all_difficulty_merge(tmp_path):
    """Test that difficulty=all merges easy + hard datasets."""
    import sys
    from vlm_nsm.experiments.run_language_variation import main as lv_main

    old_argv = sys.argv
    sys.argv = [
        "run_language_variation.py",
        "--task", "hanoi",
        "--n_blocks", "3",
        "--parser", "mock",
        "--difficulty", "all",
        "--output_dir", str(tmp_path),
    ]
    try:
        lv_main()
    finally:
        sys.argv = old_argv

    import json
    json_out = tmp_path / "language_variation_hanoi_all_mock_full_full.json"
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert len(data) >= 28  # 8 easy + 22 hard
