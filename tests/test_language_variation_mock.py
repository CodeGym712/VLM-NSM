from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import run_single_sample, load_instructions

RAW_OUTPUT_DIR = PROJECT_ROOT / "generated" / "logs" / "qwen_outputs"
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_language_variation_mock_hanoi():
    samples = load_instructions(str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl"))
    assert len(samples) >= 7

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
        )
        assert result["json_parse_success"], f"{sample['id']}: parse failed"
        assert result["alignment_success"], f"{sample['id']}: alignment failed"
        assert result["planner_success"], f"{sample['id']}: planner failed"
        assert result["execution_success"], f"{sample['id']}: execution failed"
        assert result["goal_success"], f"{sample['id']}: goal not satisfied"
        assert result["schema_validity"], f"{sample['id']}: schema should be valid"
        assert result["alignment_applicable"], f"{sample['id']}: alignment_applicable"


def test_language_variation_mock_stacking():
    samples = load_instructions(str(PROJECT_ROOT / "data" / "language_variations" / "stacking_zh_en.jsonl"))
    assert len(samples) >= 7

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
        )
        assert result["json_parse_success"], f"{sample['id']}: parse failed"
        assert result["alignment_success"], f"{sample['id']}: alignment failed"
        assert result["planner_success"], f"{sample['id']}: planner failed"
        assert result["execution_success"], f"{sample['id']}: execution failed"
        assert result["goal_success"], f"{sample['id']}: goal not satisfied"
        assert result["schema_validity"], f"{sample['id']}: schema should be valid"
        assert result["alignment_applicable"], f"{sample['id']}: alignment_applicable"


def test_language_variation_mock_entrypoint(tmp_path):
    """Test that the CLI entrypoint works with parser=mock."""
    import json
    from vlm_nsm.experiments.run_language_variation import main as lv_main

    # Simulate CLI args via argparse
    import sys
    old_argv = sys.argv
    instruction_file = str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    sys.argv = [
        "run_language_variation.py",
        "--task", "hanoi",
        "--n_blocks", "3",
        "--parser", "mock",
        "--instruction_file", instruction_file,
        "--output_dir", str(tmp_path),
    ]
    try:
        lv_main()
    finally:
        sys.argv = old_argv

    # Check output files
    json_out = tmp_path / "language_variation_hanoi_easy_mock_full_full.json"
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert len(data) >= 7
    assert all(r["goal_success"] for r in data)
