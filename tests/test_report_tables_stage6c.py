"""Test report tables from Stage 6C with mock data.

Tests the four report tables with filtering logic:
1. model_comparison: difficulty=hard, prompt_variant in (full,medium), alignment_mode=full
2. prompt_comparison: model_name=qwen25_32b, difficulty=hard, alignment_mode=full
3. alignment_benefit: model_name=qwen25_32b, difficulty=hard, prompt_variant=medium
4. by_category: all results grouped by category
"""

import csv
import json

from vlm_nsm.experiments.export_report_tables import (
    export_model_comparison,
    export_prompt_comparison,
    export_alignment_benefit,
    export_by_category,
)


def _make_result(
    model_name="qwen25_32b",
    task="hanoi",
    difficulty="hard",
    category="alias_heavy",
    prompt_variant="full",
    alignment_mode="full",
    goal_success=True,
    json_parse_success=True,
    schema_validity=True,
    alignment_success=True,
    alignment_applicable=True,
    repaired=False,
    planner_success=True,
    execution_success=True,
    inference_time=1.0,
    output_tokens=50,
):
    return {
        "model_name": model_name,
        "task": task,
        "difficulty": difficulty,
        "category": category,
        "prompt_variant": prompt_variant,
        "alignment_mode": alignment_mode,
        "goal_success": goal_success,
        "json_parse_success": json_parse_success,
        "schema_validity": schema_validity,
        "alignment_success": alignment_success,
        "alignment_applicable": alignment_applicable,
        "repaired": repaired,
        "planner_success": planner_success,
        "execution_success": execution_success,
        "inference_time": inference_time,
        "output_tokens": output_tokens,
    }


def test_model_comparison_table(tmp_path):
    """Model comparison filters: hard, full/medium prompts, full alignment."""
    results = [
        # Should be included
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full"),
        _make_result(model_name="qwen25_14b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        # Should be excluded
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="easy",
                      prompt_variant="full", alignment_mode="full"),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="weak", alignment_mode="full"),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="no_alignment"),
    ]
    out = tmp_path / "report_table_model_comparison.csv"
    export_model_comparison(results, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 2 models x 1 task = 2 rows (grouped by model_name + task)
    assert len(rows) >= 2
    # Verify schema_validity_rate column exists
    assert "schema_validity_rate" in rows[0]


def test_prompt_comparison_table(tmp_path):
    """Prompt comparison filters: qwen25_32b, hard, full alignment."""
    results = [
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full"),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="weak", alignment_mode="full"),
        # Should be excluded (wrong model)
        _make_result(model_name="qwen25_14b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        # Should be excluded (not full alignment)
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="no_alignment"),
    ]
    out = tmp_path / "report_table_prompt_comparison.csv"
    export_prompt_comparison(results, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 3 prompt variants
    prompt_variants = {r["prompt_variant"] for r in rows}
    assert prompt_variants == {"full", "medium", "weak"}


def test_alignment_benefit_table(tmp_path):
    """Alignment benefit filters: qwen25_32b, hard, medium prompt."""
    results = [
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full",
                      goal_success=True, alignment_success=True),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="no_alignment",
                      goal_success=False, alignment_success=None,
                      alignment_applicable=False),
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="no_repair",
                      goal_success=True, alignment_success=True),
        # Should be excluded (not medium prompt)
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        # Should be excluded (not qwen25_32b)
        _make_result(model_name="qwen25_14b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full"),
    ]
    out = tmp_path / "report_table_alignment_benefit.csv"
    export_alignment_benefit(results, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 3 alignment modes
    # no_alignment should show N/A for alignment success
    for r in rows:
        if r["alignment_mode"] == "no_alignment":
            assert r["alignment_success_rate"] == "N/A"


def test_by_category_table(tmp_path):
    """By-category table includes all results grouped by category."""
    results = [
        _make_result(task="hanoi", category="alias_heavy", goal_success=True),
        _make_result(task="hanoi", category="alias_heavy", goal_success=False),
        _make_result(task="hanoi", category="underspecified", goal_success=True),
        _make_result(task="stacking", category="alias_heavy", goal_success=True),
    ]
    out = tmp_path / "report_table_by_category.csv"
    export_by_category(results, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 3 category+task combos

    # Check schema_validity_rate is present
    assert "schema_validity_rate" in rows[0]


def test_alignment_benefit_with_none_alignment(tmp_path):
    """no_alignment should report N/A for alignment_success_rate."""
    results = [
        _make_result(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="no_alignment",
                      alignment_success=None, alignment_applicable=False,
                      goal_success=False),
    ]
    out = tmp_path / "report_table_alignment_benefit.csv"
    export_alignment_benefit(results, out)
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["alignment_success_rate"] == "N/A"
    assert rows[0]["alignment_mode"] == "no_alignment"
