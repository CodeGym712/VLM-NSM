"""Test DatasetComponent and DatasetManifest dataclass schema.

Uses only the dataclass API — no file I/O, no DashScope, no Robosuite.
"""

import json
import pytest
from datetime import datetime

from vlm_nsm.data.dataset_manifest import DatasetComponent, DatasetManifest


class TestDatasetComponent:
    def test_minimal_component(self):
        c = DatasetComponent(
            name="test_component",
            version="0.1",
            component_type="parse_eval",
            source_paths=["path/to/file.json"],
            sample_count=10,
            task_types=["stacking"],
        )
        assert c.name == "test_component"
        assert c.sample_count == 10
        assert c.parser_modes is None  # optional field

    def test_component_with_all_fields(self):
        c = DatasetComponent(
            name="full_component",
            version="1.0",
            component_type="e2e_demo",
            source_paths=["a.json", "b.json"],
            sample_count=5,
            task_types=["stacking"],
            parser_modes=["online_qwen"],
            metrics={"success": 5, "total": 5},
            notes="All demos succeeded",
        )
        assert c.parser_modes == ["online_qwen"]
        assert c.metrics["success"] == 5

    def test_to_dict_excludes_none(self):
        c = DatasetComponent(
            name="minimal",
            version="0.1",
            component_type="test",
            source_paths=["x.json"],
            sample_count=0,
            task_types=["stacking"],
        )
        d = c.to_dict()
        assert "parser_modes" not in d  # None, excluded
        assert "notes" not in d  # None, excluded
        assert d["name"] == "minimal"
        assert d["sample_count"] == 0
        # metrics is empty dict (not None) — should be excluded
        assert "metrics" not in d

    def test_to_dict_includes_non_none(self):
        c = DatasetComponent(
            name="with_notes",
            version="0.1",
            component_type="test",
            source_paths=["x.json"],
            sample_count=1,
            task_types=["stacking"],
            notes="Some notes",
            metrics={"key": "val"},
        )
        d = c.to_dict()
        assert d["notes"] == "Some notes"
        assert d["metrics"] == {"key": "val"}


class TestDatasetManifest:
    def test_create_manifest(self):
        c = DatasetComponent(
            name="comp1",
            version="0.1",
            component_type="parse_eval",
            source_paths=["data.json"],
            sample_count=10,
            task_types=["stacking"],
        )
        manifest = DatasetManifest.create(
            name="Test Manifest",
            version="0.1",
            stage="stage_test",
            components=[c],
        )
        assert manifest.dataset_name == "Test Manifest"
        assert manifest.dataset_version == "0.1"
        assert manifest.project_stage == "stage_test"
        assert len(manifest.components) == 1

    def test_create_manifest_with_limitations(self):
        c = DatasetComponent(
            name="comp1",
            version="0.1",
            component_type="test",
            source_paths=["f.json"],
            sample_count=0,
            task_types=["stacking"],
        )
        manifest = DatasetManifest.create(
            name="With Limits",
            version="0.1",
            stage="test",
            components=[c],
            limitations=["Limitation A", "Limitation B"],
            global_metrics={"total": 1},
        )
        assert len(manifest.limitations) == 2
        assert manifest.global_metrics["total"] == 1

    def test_created_at_timestamp(self):
        c = DatasetComponent(
            name="c",
            version="0.1",
            component_type="t",
            source_paths=["f"],
            sample_count=0,
            task_types=["s"],
        )
        m = DatasetManifest.create("N", "1", "s", [c])
        # Verify ISO 8601 format
        assert "T" in m.created_at
        # Should parse as valid datetime
        dt = datetime.fromisoformat(m.created_at.replace("Z", "+00:00"))
        assert dt.year == 2026

    def test_to_dict_structure(self):
        c = DatasetComponent(
            name="c1",
            version="0.1",
            component_type="parse_eval",
            source_paths=["p.json"],
            sample_count=5,
            task_types=["stacking"],
            metrics={"ok": 5},
        )
        manifest = DatasetManifest.create(
            name="M", version="1", stage="s", components=[c],
            limitations=["L1"],
        )
        d = manifest.to_dict()
        assert d["dataset_name"] == "M"
        assert "created_at" in d
        assert len(d["components"]) == 1
        assert d["limitations"] == ["L1"]
        comp = d["components"][0]
        assert comp["name"] == "c1"
        assert comp["metrics"] == {"ok": 5}

    def test_write_and_load_json(self, tmp_path):
        c = DatasetComponent(
            name="roundtrip",
            version="0.1",
            component_type="test",
            source_paths=["f.json"],
            sample_count=3,
            task_types=["stacking"],
        )
        manifest = DatasetManifest.create("RT", "1", "s", [c])
        path = str(tmp_path / "manifest.json")
        manifest.write_json(path)
        loaded = DatasetManifest.load_json(path)
        assert loaded.dataset_name == "RT"
        assert len(loaded.components) == 1
        assert loaded.components[0].name == "roundtrip"
        assert loaded.components[0].sample_count == 3

    def test_json_serializable(self, tmp_path):
        c = DatasetComponent(
            name="jsontest",
            version="0.1",
            component_type="test",
            source_paths=["f.json"],
            sample_count=0,
            task_types=["stacking"],
        )
        m = DatasetManifest.create("J", "1", "s", [c])
        path = str(tmp_path / "manifest.json")
        m.write_json(path)
        raw = json.loads(open(path).read())
        assert raw["dataset_name"] == "J"
        assert len(raw["components"]) == 1
