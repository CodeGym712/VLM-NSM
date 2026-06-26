"""Test compare_scene_grounding_results comparison logic.

No DashScope, Robosuite, or GPU required — uses synthetic in-memory data.
"""

import json
import pytest
from pathlib import Path

from vlm_nsm.experiments.compare_scene_grounding_results import (
    _summarize_results,
    _per_category,
    _load_categories,
    _find_failure_examples,
    _classify_invalid_object,
)


def _make_entry(instruction: str, overrides: dict | None = None) -> dict:
    """Create a minimal result entry with defaults."""
    entry = {
        "instruction": instruction,
        "parse_success": True,
        "schema_valid": True,
        "alignment_success": True,
        "target_stack_success": True,
        "planner_success": True,
        "object_grounding_valid": None,
        "grounding_errors": None,
        "expected_bindings": None,
        "predicted_bindings": {},
        "error_type": None,
    }
    if overrides:
        entry.update(overrides)
    return entry


class TestCompareSceneGroundingResults:
    def test_both_perfect(self):
        """All entries succeed in both — metrics should match."""
        entries = [
            _make_entry("Put the red cube on the green cube."),
            _make_entry("Stack the green cube on the red cube."),
        ]
        cat_map = {}
        wm = _summarize_results(entries, "without_scene", cat_map)
        om = _summarize_results(entries, "with_oracle", cat_map)
        assert wm["total"] == 2
        assert wm["fully_successful"] == 2
        assert om["fully_successful"] == 2

    def test_oracle_improves_grounding(self):
        """Oracle scene should fix grounding where without has failures."""
        without_entries = [
            _make_entry("Put red on green", overrides={
                "object_grounding_valid": False,
                "grounding_errors": ["Wrong binding for 'top block'"],
            }),
        ]
        with_entries = [
            _make_entry("Put red on green", overrides={
                "object_grounding_valid": True,
                "grounding_errors": [],
            }),
        ]
        cat_map = {}
        wm = _summarize_results(without_entries, "without_scene", cat_map)
        om = _summarize_results(with_entries, "with_oracle", cat_map)
        assert wm["object_grounding_valid"] == 0
        assert wm["object_grounding_fail"] == 1
        assert om["object_grounding_valid"] == 1
        assert om["object_grounding_fail"] == 0

    def test_classify_invalid_object(self):
        """_classify_invalid_object should count correctly."""
        cat_map = {
            "Put the blue cube on the green cube.": "adversarial_invalid_object",
            "Put the red cube on the green cube.": "color_grounding",
        }
        entries = [
            _make_entry("Put the blue cube on the green cube."),
            _make_entry("Put the red cube on the green cube."),
        ]
        count = _classify_invalid_object(entries, cat_map)
        assert count == 1

    def test_per_category_breakdown(self):
        """Per-category object grounding counts should be correct."""
        cat_map = {
            "Put red on green": "color_grounding",
            "Put green on red": "color_grounding",
            "Put left on right": "positional_grounding",
        }
        entries = [
            _make_entry("Put red on green", overrides={
                "object_grounding_valid": True,
            }),
            _make_entry("Put green on red", overrides={
                "object_grounding_valid": False,
            }),
            _make_entry("Put left on right", overrides={
                "object_grounding_valid": True,
            }),
        ]
        result = _per_category(entries, cat_map)
        assert result["color_grounding"]["total"] == 2
        assert result["color_grounding"]["object_grounding_valid"] == 1
        assert result["color_grounding"]["object_grounding_fail"] == 1
        assert result["positional_grounding"]["total"] == 1
        assert result["positional_grounding"]["object_grounding_valid"] == 1

    def test_summarize_counts_fully_successful(self):
        """fully_successful requires all sub-metrics to pass."""
        entries = [
            _make_entry("Instruction 1"),  # all defaults pass
            _make_entry("Instruction 2", overrides={"parse_success": False}),
            _make_entry("Instruction 3", overrides={"planner_success": False}),
        ]
        cat_map = {}
        result = _summarize_results(entries, "test", cat_map)
        assert result["total"] == 3
        assert result["parse_ok"] == 2
        assert result["fully_successful"] == 1

    def test_grounding_errors_totaled(self):
        """Total grounding error count should sum across entries."""
        entries = [
            _make_entry("A", overrides={"grounding_errors": ["err1"]}),
            _make_entry("B", overrides={"grounding_errors": ["err2", "err3"]}),
            _make_entry("C", overrides={"grounding_errors": None}),
        ]
        cat_map = {}
        result = _summarize_results(entries, "test", cat_map)
        assert result["total_grounding_errors"] == 3

    def test_failure_examples_extracted(self):
        """_find_failure_examples should extract planner and grounding failures."""
        entries = [
            _make_entry("Good one"),  # success
            _make_entry("Planner fail", overrides={"planner_success": False}),
            _make_entry("Grounding fail", overrides={"object_grounding_valid": False,
                                                      "grounding_errors": ["err"]}),
        ]
        examples = _find_failure_examples(entries, max_examples=5)
        assert len(examples) == 2

    def test_failure_examples_maxed(self):
        """_find_failure_examples should respect max_examples."""
        entries = [
            _make_entry(f"Fail {i}", overrides={"planner_success": False})
            for i in range(10)
        ]
        examples = _find_failure_examples(entries, max_examples=3)
        assert len(examples) == 3

    def test_category_file_loading(self, tmp_path):
        """Category mapping from input JSON should work."""
        data = [
            {"instruction": "Put red on green", "category": "color_grounding"},
            {"instruction": "Put left on right", "category": "positional_grounding"},
        ]
        p = tmp_path / "input.json"
        p.write_text(json.dumps(data))
        cat_map = _load_categories(str(p))
        assert cat_map["Put red on green"] == "color_grounding"
        assert cat_map["Put left on right"] == "positional_grounding"
        assert len(cat_map) == 2

    def test_category_file_none(self):
        """_load_categories(None) returns empty dict."""
        assert _load_categories(None) == {}

    def test_category_file_not_found(self):
        """_load_categories on nonexistent path returns empty dict."""
        assert _load_categories("/nonexistent/path.json") == {}

    def test_invalid_object_in_summarize_results(self):
        """_summarize_results should count invalid_object_grounded correctly."""
        cat_map = {
            "Put blue on green": "adversarial_invalid_object",
        }
        entries = [
            _make_entry("Put blue on green", overrides={
                "object_grounding_valid": True,
                "expected_bindings": {},
            }),
        ]
        result = _summarize_results(entries, "test", cat_map)
        assert result["invalid_object_grounded"] == 1
        assert result["empty_expected_bindings"] == 1

    def test_failure_taxonomy(self):
        """Failure taxonomy should categorize by failure type."""
        entries = [
            _make_entry("Parse fail", overrides={"parse_success": False}),
            _make_entry("Schema fail", overrides={"schema_valid": False}),
            _make_entry("Align fail", overrides={"alignment_success": False}),
            _make_entry("Target stack fail", overrides={"target_stack_success": False}),
            _make_entry("Planner fail", overrides={"planner_success": False}),
            _make_entry("Grounding fail", overrides={"object_grounding_valid": False}),
            _make_entry("All good"),  # skipped in taxonomy
        ]
        cat_map = {}
        result = _summarize_results(entries, "test", cat_map)
        tax = result["failure_taxonomy"]
        assert tax.get("parse_error") == 1
        assert tax.get("schema_invalid") == 1
        assert tax.get("alignment_failed") == 1
        assert tax.get("target_stack_error") == 1
        assert tax.get("planner_error") == 1
        assert tax.get("grounding_failed") == 1
        assert sum(tax.values()) == 6

    def test_ungroundable_detected_metric(self):
        """Ungroundable entries should be counted."""
        entries = [
            _make_entry("Yellow cube", overrides={
                "status": "ungroundable", "category": "adversarial_invalid_object",
                "object_grounding_valid": None,
            }),
            _make_entry("Normal instruction"),
        ]
        cat_map = {"Yellow cube": "adversarial_invalid_object"}
        result = _summarize_results(entries, "test", cat_map)
        assert result["ungroundable_detected"] == 1
        assert result["adversarial_rejection_success"] == 1
        assert result["false_rejection"] == 0

    def test_false_rejection_metric(self):
        """Non-adversarial instruction rejected should count as false rejection."""
        entries = [
            _make_entry("Put red on green", overrides={
                "status": "ungroundable", "category": "color_grounding",
                "object_grounding_valid": None,
            }),
        ]
        cat_map = {"Put red on green": "color_grounding"}
        result = _summarize_results(entries, "test", cat_map)
        assert result["ungroundable_detected"] == 1
        assert result["adversarial_rejection_success"] == 0
        assert result["false_rejection"] == 1
        assert result["false_rejection_rate"] == 1.0

    def test_ungroundable_in_failure_taxonomy(self):
        """Ungroundable entries should appear as 'ungroundable_object' in taxonomy."""
        entries = [
            _make_entry("Yellow cube", overrides={
                "status": "ungroundable", "category": "adversarial_invalid_object",
                "object_grounding_valid": None,
            }),
        ]
        cat_map = {"Yellow cube": "adversarial_invalid_object"}
        result = _summarize_results(entries, "test", cat_map)
        assert result["failure_taxonomy"].get("ungroundable_object") == 1

    def test_ungroundable_in_failure_examples(self):
        """Ungroundable entries should appear in failure examples."""
        entries = [
            _make_entry("Normal", overrides={"planner_success": True}),
            _make_entry("Rejected", overrides={
                "status": "ungroundable", "category": "adversarial_invalid_object",
                "object_grounding_valid": None,
            }),
        ]
        cat_map = {"Rejected": "adversarial_invalid_object"}
        from vlm_nsm.experiments.compare_scene_grounding_results import _find_failure_examples
        examples = _find_failure_examples(entries)
        assert len(examples) == 1
        assert examples[0]["instruction"] == "Rejected"

    def test_per_category_ungroundable(self):
        """Per-category should include ungroundable count."""
        cat_map = {"Reject yellow": "adversarial_invalid_object"}
        entries = [
            _make_entry("Reject yellow", overrides={
                "status": "ungroundable", "object_grounding_valid": None,
            }),
        ]
        result = _per_category(entries, cat_map)
        assert result["adversarial_invalid_object"]["ungroundable_detected"] == 1
