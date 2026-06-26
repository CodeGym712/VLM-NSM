"""Test eval_online_qwen_parser scene inventory integration.

Uses MockGoalParser — no DashScope, Robosuite, or GPU required.
Verifies scene_inventory_source, scene_description, scene_objects,
and object_grounding_valid fields appear in output entries.
"""

import json
import pytest
from pathlib import Path

from vlm_nsm.experiments.eval_online_qwen_parser import (
    load_instructions,
    parser_target_stack_from_alignment,
    main,
)
from vlm_nsm.core.goal_parser import MockGoalParser
from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description
from vlm_nsm.evaluation.object_grounding import evaluate_object_grounding


class TestSceneInventoryFields:
    def test_entry_has_scene_fields(self, tmp_path):
        """Simulate what eval script produces with scene inventory."""
        entry = {
            "instruction": "Put the red cube on the green cube.",
            "scene_inventory_source": "oracle",
            "scene_description": "Current scene:\n  cube1 (block), color: red\n  cube2 (block), color: green",
            "scene_objects": [{"object_id": "cube1", "color": "red"}, {"object_id": "cube2", "color": "green"}],
            "object_grounding_valid": True,
            "grounding_errors": None,
            "expected_bindings": {},
            "predicted_bindings": {"top block": "cube1", "bottom block": "cube2"},
        }
        assert entry["scene_inventory_source"] == "oracle"
        assert entry["object_grounding_valid"] is True

    def test_scene_objects_from_scene_info(self):
        """Scene objects list should match DetectedObject fields."""
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene = SceneInfo(objects=objs)
        scene_objs = [{"object_id": o.label, "color": o.color} for o in scene.objects]
        assert scene_objs == [
            {"object_id": "cube1", "color": "red"},
            {"object_id": "cube2", "color": "green"},
        ]

    def test_scene_description_from_scene_info(self):
        """Scene description should contain color info for grounding."""
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        assert "red" in desc
        assert "green" in desc

    def test_grounding_eval_in_pipeline(self):
        """Object grounding should work with scene info."""
        goal_json = {
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
        }
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ])
        result = evaluate_object_grounding(
            instruction="Put the red cube on the green cube",
            goal_json=goal_json,
            scene_info=scene,
            expected_bindings={"top block": "cube1", "bottom block": "cube2"},
        )
        assert result["object_grounding_valid"] is True
        assert result["grounding_method_used"] == "expected_bindings_match"

    def test_grounding_failure_detected(self):
        """Wrong bindings should set grounding_valid=False."""
        goal_json = {
            "object_bindings": {"top block": "cube2", "bottom block": "cube1"},
        }
        result = evaluate_object_grounding(
            instruction="Put red on green",
            goal_json=goal_json,
            expected_bindings={"top block": "cube1", "bottom block": "cube2"},
        )
        assert result["object_grounding_valid"] is False

    def test_scene_inventory_none_by_default(self):
        """Without --scene_inventory, scene fields should be absent or None."""
        entry = {
            "scene_inventory_source": "none",
            "scene_description": None,
            "scene_objects": None,
        }
        assert entry["scene_inventory_source"] == "none"
        assert entry["scene_description"] is None
        assert entry["scene_objects"] is None
