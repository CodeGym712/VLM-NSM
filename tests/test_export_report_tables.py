"""Test export_report_tables with mock data (no real Qwen results needed)."""

import csv
import json
from pathlib import Path

from vlm_nsm.experiments.export_report_tables import (
    export_model_comparison,
    export_prompt_comparison,
    export_alignment_benefit,
    export_by_category,
    load_raw_results,
    load_json,
)


def _make_sample(
    model_name="mock_model",
    task="hanoi",
    difficulty="easy",
    category="zh_standard",
    prompt_variant="full",
    alignment_mode="full",
    goal_success=True,
    json_parse_success=True,
    schema_validity=True,
    alignment_success=True,
    alignment_applicable=True,
    repaired=False,
    planner_success=True,
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
        "inference_time": inference_time,
        "output_tokens": output_tokens,
    }


def test_export_model_comparison(tmp_path):
    # Model comparison filters: difficulty=hard, prompt_variant in (full,medium), alignment_mode=full
    samples = [
        _make_sample(model_name="qwen25_14b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full"),
    ]
    out = tmp_path / "report_table_model_comparison.csv"
    export_model_comparison(samples, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_export_prompt_comparison(tmp_path):
    # Prompt comparison filters: model_name=qwen25_32b, difficulty=hard, alignment_mode=full
    samples = [
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="full", alignment_mode="full"),
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="weak", alignment_mode="full"),
    ]
    out = tmp_path / "report_table_prompt_comparison.csv"
    export_prompt_comparison(samples, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["prompt_variant"] in ("full", "weak")


def test_export_alignment_benefit(tmp_path):
    # Alignment benefit filters: model_name=qwen25_32b, difficulty=hard, prompt_variant=medium
    samples = [
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="full",
                      goal_success=True),
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      prompt_variant="medium", alignment_mode="no_alignment",
                      alignment_success=None, alignment_applicable=False,
                      goal_success=False),
    ]
    out = tmp_path / "report_table_alignment_benefit.csv"
    export_alignment_benefit(samples, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # Check alignment mode recorded
    modes = {r["alignment_mode"] for r in rows}
    assert "full" in modes
    assert "no_alignment" in modes


def test_export_by_category(tmp_path):
    samples = [
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      category="alias_heavy", goal_success=True, repaired=True),
        _make_sample(model_name="qwen25_32b", task="hanoi", difficulty="hard",
                      category="underspecified", goal_success=False),
    ]
    out = tmp_path / "report_table_by_category.csv"
    export_by_category(samples, out)
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_load_raw_results(tmp_path):
    samples = [
        _make_sample(model_name="qwen25_32b", task="hanoi"),
        _make_sample(model_name="qwen25_32b", task="stacking"),
    ]
    fpath = tmp_path / "language_variation_hanoi_hard_qwen25_32b_full_full.json"
    fpath.write_text(json.dumps(samples), encoding="utf-8")

    loaded = load_raw_results(tmp_path)
    assert len(loaded) == 2


def test_load_raw_results_skips_summary(tmp_path):
    samples = [_make_sample()]
    fpath = tmp_path / "language_variation_hanoi_hard_mock_full_full.json"
    fpath.write_text(json.dumps(samples), encoding="utf-8")
    summary_path = tmp_path / "language_variation_summary.json"
    summary_path.write_text(json.dumps(samples), encoding="utf-8")

    loaded = load_raw_results(tmp_path)
    assert len(loaded) == 1  # summary skipped


def test_load_json_not_found(tmp_path):
    assert load_json(tmp_path / "nonexistent.json") is None
