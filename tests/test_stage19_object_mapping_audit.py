"""Tests for Stage 19-D1 object mapping audit."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vlm_nsm.experiments.stage19_object_mapping_audit import (
    run_object_mapping_audit,
)


class TestObjectMappingAudit:
    def test_all_episodes_consistent_with_default_mapping(self):
        suite = {
            "description": "Test suite",
            "available_objects": ["cube1", "cube2"],
            "color_map": {"cube1": "red", "cube2": "green"},
            "episodes": [
                {
                    "episode_id": "test_red_on_green",
                    "goal_id": "red_on_green",
                    "expected_top": "cube1",
                    "expected_bottom": "cube2",
                    "expected_execution": "execute",
                },
                {
                    "episode_id": "test_green_on_red",
                    "goal_id": "green_on_red",
                    "expected_top": "cube2",
                    "expected_bottom": "cube1",
                    "expected_execution": "execute",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.json"
            suite_path.write_text(json.dumps(suite))
            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            result = run_object_mapping_audit(str(suite_path), str(output_path), str(md_path))

            assert result["mapping_inconsistent_count"] == 0
            assert result["mapping_consistent_count"] == 2
            assert output_path.exists()
            assert md_path.exists()

    def test_same_object_top_bottom_detected_as_inconsistent(self):
        suite = {
            "description": "Test suite",
            "color_map": {},
            "episodes": [
                {
                    "episode_id": "test_same_obj",
                    "goal_id": "invalid",
                    "expected_top": "cube1",
                    "expected_bottom": "cube1",
                    "expected_execution": "reject",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.json"
            suite_path.write_text(json.dumps(suite))
            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            result = run_object_mapping_audit(str(suite_path), str(output_path), str(md_path))

            assert result["mapping_inconsistent_count"] == 1
            assert len(result["mapping_issues"]) == 1
            assert "same object" in result["mapping_issues"][0].lower()

    def test_output_files_created(self):
        suite = {
            "description": "Test",
            "color_map": {},
            "episodes": [
                {
                    "episode_id": "ep1",
                    "goal_id": "g1",
                    "expected_top": "cube1",
                    "expected_bottom": "cube2",
                    "expected_execution": "execute",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "suite.json"
            suite_path.write_text(json.dumps(suite))
            output_path = Path(tmpdir) / "audit.json"
            md_path = Path(tmpdir) / "audit.md"

            run_object_mapping_audit(str(suite_path), str(output_path), str(md_path))

            assert output_path.exists()
            assert md_path.exists()

            data = json.loads(output_path.read_text())
            assert "static_mappings" in data
            assert "CUBE_NAME_MAP" in data["static_mappings"]
            assert "episodes" in data
