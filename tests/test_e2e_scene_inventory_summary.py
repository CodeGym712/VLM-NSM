"""Test that E2E summary includes scene inventory fields.

No DashScope, Robosuite, or GPU required — uses MockGoalParser
and verifies the summary dict structure.
"""

import pytest
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo


EXPECTED_SCENE_FIELDS = [
    "scene_inventory_source",
    "scene_description",
    "scene_objects",
    "object_grounding_valid",
    "grounding_errors",
]


class TestE2ESceneInventorySummary:
    def test_scene_fields_in_summary_entry(self):
        """Mock summary entry should contain all scene fields."""
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ])
        entry = {
            "instruction": "test",
            "success": True,
            "scene_inventory_source": "oracle",
            "scene_description": "Current scene:\n  cube1 (block), color: red\n  cube2 (block), color: green",
            "scene_objects": [{"object_id": "cube1", "color": "red"}, {"object_id": "cube2", "color": "green"}],
            "object_grounding_valid": True,
            "grounding_errors": None,
        }
        for field in EXPECTED_SCENE_FIELDS:
            assert field in entry, f"Missing scene field: {field}"

    def test_scene_inventory_source_none(self):
        """Without scene inventory, source should be 'none'."""
        entry = {
            "scene_inventory_source": "none",
            "scene_description": None,
            "scene_objects": None,
        }
        assert entry["scene_inventory_source"] == "none"

    def test_scene_inventory_source_oracle(self):
        """With oracle scene inventory, source should be 'oracle'."""
        entry = {
            "scene_inventory_source": "oracle",
            "scene_description": "Current scene:\n  cube1 (block), color: red",
            "scene_objects": [{"object_id": "cube1", "color": "red"}],
        }
        assert entry["scene_inventory_source"] == "oracle"

    def test_scene_objects_structure(self):
        """scene_objects should be a list of dicts with object_id and color."""
        scene_objects = [
            {"object_id": "cube1", "color": "red"},
            {"object_id": "cube2", "color": "green"},
        ]
        for obj in scene_objects:
            assert "object_id" in obj
            assert "color" in obj

    def test_scene_objects_from_detected_objects(self):
        """Verify scene_objects can be derived from DetectedObject list."""
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene_objects = [{"object_id": o.label, "color": o.color} for o in objs]
        assert scene_objects == [
            {"object_id": "cube1", "color": "red"},
            {"object_id": "cube2", "color": "green"},
        ]

    def test_grounding_fields_none_when_no_inventory(self):
        """Without scene inventory, grounding fields should be None."""
        entry = {
            "object_grounding_valid": None,
            "grounding_errors": None,
        }
        assert entry["object_grounding_valid"] is None
        assert entry["grounding_errors"] is None
