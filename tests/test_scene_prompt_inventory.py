"""Test scene prompt inventory — build_scene_description with and without colors/poses.

No DashScope, Robosuite, or GPU required.
"""

import pytest
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


class TestScenePromptInventory:
    def test_build_scene_no_objects(self):
        scene = SceneInfo()
        desc = build_scene_description(scene)
        assert "Current scene:" in desc
        assert "Gripper is empty" in desc

    def test_build_scene_with_objects_no_colors(self):
        objs = [
            DetectedObject(label="cube1", category="block"),
            DetectedObject(label="cube2", category="block"),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        assert "cube1 (block)" in desc
        assert "cube2 (block)" in desc
        assert "color:" not in desc

    def test_build_scene_with_colors(self):
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        assert "color: red" in desc
        assert "color: green" in desc

    def test_build_scene_with_poses(self):
        objs = [
            DetectedObject(label="cube1", category="block", pose={"x": 0.1, "y": 0.2, "z": 0.8}),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        assert "0.10" in desc
        assert "0.20" in desc
        assert "0.80" in desc

    def test_build_scene_with_on_relations(self):
        scene = SceneInfo(on_relations=[("cube1", "cube2")])
        desc = build_scene_description(scene)
        assert "Stacking relations" in desc
        assert "cube1 is on cube2" in desc

    def test_build_scene_with_grasped_object(self):
        scene = SceneInfo(free_gripper=False, grasped_object="cube1")
        desc = build_scene_description(scene)
        assert "Grasping: cube1" in desc
        assert "Gripper is empty" not in desc

    def test_build_scene_with_table(self):
        objs = [
            DetectedObject(label="table", category="table", pose={"x": 0.0, "y": 0.0, "z": 0.0}),
            DetectedObject(label="cube1", category="block", color="red"),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        assert "table (table)" in desc
        assert "cube1 (block)" in desc

    def test_scene_prompt_inventory_format(self):
        """The prompt should list available objects for grounding."""
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene = SceneInfo(objects=objs)
        desc = build_scene_description(scene)
        # Should contain both objects with color info for grounding
        assert "cube1" in desc
        assert "cube2" in desc
        assert "red" in desc
        assert "green" in desc
