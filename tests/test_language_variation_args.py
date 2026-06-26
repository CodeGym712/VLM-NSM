"""Test the new debug parameters for run_language_variation.py."""

import sys
from pathlib import Path

from vlm_nsm.core.utils import PROJECT_ROOT
from vlm_nsm.experiments.run_language_variation import load_instructions


def test_max_samples_filter():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    )
    total = len(samples)

    # Simulate --max_samples 2
    filtered = list(samples)
    if total > 2:
        filtered = filtered[:2]
    assert len(filtered) == min(total, 2)


def test_limit_categories_filter():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    )
    allowed = {"zh_standard", "en_standard"}
    filtered = [s for s in samples if s.get("category", "") in allowed]
    assert len(filtered) >= 2
    for s in filtered:
        assert s["category"] in allowed


def test_sample_ids_filter():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    )
    allowed = {"hanoi_zh_standard_001", "hanoi_en_standard_005"}
    filtered = [s for s in samples if s["id"] in allowed]
    assert len(filtered) == 2
    for s in filtered:
        assert s["id"] in allowed


def test_model_name_tagging():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    )
    model_name = "qwen25_14b"
    for s in samples:
        s["model_name"] = model_name
    for s in samples:
        assert s["model_name"] == "qwen25_14b"


def test_prompt_version_tagging():
    samples = load_instructions(
        str(PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl")
    )
    prompt_version = "v1"
    for s in samples:
        s["prompt_version"] = prompt_version
    for s in samples:
        assert s["prompt_version"] == "v1"


def test_mock_entrypoint_with_max_samples(tmp_path):
    """Test CLI entrypoint with --max_samples flag."""
    from vlm_nsm.experiments.run_language_variation import main as lv_main

    old_argv = sys.argv
    instruction_file = str(
        PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl"
    )
    sys.argv = [
        "run_language_variation.py",
        "--task", "hanoi",
        "--n_blocks", "3",
        "--parser", "mock",
        "--instruction_file", instruction_file,
        "--output_dir", str(tmp_path),
        "--max_samples", "2",
    ]
    try:
        lv_main()
    finally:
        sys.argv = old_argv

    json_out = tmp_path / "language_variation_hanoi_easy_mock_full_full.json"
    assert json_out.exists()
    import json
    data = json.loads(json_out.read_text())
    assert len(data) <= 2


def test_mock_entrypoint_with_limit_categories(tmp_path):
    """Test CLI entrypoint with --limit_categories flag."""
    from vlm_nsm.experiments.run_language_variation import main as lv_main

    old_argv = sys.argv
    instruction_file = str(
        PROJECT_ROOT / "data" / "language_variations" / "hanoi_zh_en.jsonl"
    )
    sys.argv = [
        "run_language_variation.py",
        "--task", "hanoi",
        "--n_blocks", "3",
        "--parser", "mock",
        "--instruction_file", instruction_file,
        "--output_dir", str(tmp_path),
        "--limit_categories", "zh_standard,en_standard",
    ]
    try:
        lv_main()
    finally:
        sys.argv = old_argv

    json_out = tmp_path / "language_variation_hanoi_easy_mock_full_full.json"
    assert json_out.exists()
    import json
    data = json.loads(json_out.read_text())
    assert len(data) >= 2
    for r in data:
        assert r["category"] in ("zh_standard", "en_standard")


def test_mock_entrypoint_with_model_name(tmp_path):
    """Test CLI entrypoint with --model_name flag."""
    from vlm_nsm.experiments.run_language_variation import main as lv_main

    old_argv = sys.argv
    instruction_file = str(
        PROJECT_ROOT / "data" / "language_variations" / "stacking_zh_en.jsonl"
    )
    sys.argv = [
        "run_language_variation.py",
        "--task", "stacking",
        "--n_blocks", "3",
        "--parser", "mock",
        "--instruction_file", instruction_file,
        "--output_dir", str(tmp_path),
        "--model_name", "my_test_model",
    ]
    try:
        lv_main()
    finally:
        sys.argv = old_argv

    json_out = tmp_path / "language_variation_stacking_easy_my_test_model_full_full.json"
    assert json_out.exists(), f"Expected {json_out}"
    import json
    data = json.loads(json_out.read_text())
    assert len(data) >= 1
    for r in data:
        assert r["model_name"] == "my_test_model"
