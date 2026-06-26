"""Test oracle perception color map and DetectedObject fields.

No DashScope, Robosuite, or GPU required.
"""

import pytest
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo, DEFAULT_COLOR_MAP


class TestDetectedObject:
    def test_minimal_creation(self):
        obj = DetectedObject(label="cube1", category="block")
        assert obj.label == "cube1"
        assert obj.category == "block"
        assert obj.color is None
        assert obj.confidence == 1.0

    def test_object_id_property(self):
        obj = DetectedObject(label="cube1", category="block", color="red")
        assert obj.object_id == "cube1"

    def test_object_id_setter(self):
        obj = DetectedObject(label="cube1", category="block")
        obj.object_id = "cube2"
        assert obj.label == "cube2"
        assert obj.object_id == "cube2"

    def test_full_fields(self):
        obj = DetectedObject(
            label="cube1",
            category="block",
            pose={"x": 0.1, "y": 0.2, "z": 0.8},
            color="red",
            confidence=0.95,
            attributes={"material": "plastic"},
        )
        assert obj.pose["x"] == 0.1
        assert obj.color == "red"
        assert obj.attributes["material"] == "plastic"


class TestSceneInfo:
    def test_empty_scene(self):
        scene = SceneInfo()
        assert len(scene.objects) == 0
        assert scene.free_gripper is True
        assert scene.source == "oracle"
        assert scene.camera_name is None

    def test_with_objects(self):
        objs = [
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ]
        scene = SceneInfo(objects=objs, source="mock", camera_name="front")
        assert len(scene.get_objects_by_category("block")) == 2
        assert scene.get_object_labels() == ["cube1", "cube2"]
        assert scene.source == "mock"
        assert scene.camera_name == "front"

    def test_on_relations(self):
        scene = SceneInfo(on_relations=[("cube1", "cube2")])
        assert len(scene.on_relations) == 1
        assert scene.on_relations[0] == ("cube1", "cube2")

    def test_gripper_state(self):
        scene = SceneInfo(free_gripper=False, grasped_object="cube1")
        assert scene.free_gripper is False
        assert scene.grasped_object == "cube1"


class TestColorMap:
    def test_default_color_map(self):
        assert DEFAULT_COLOR_MAP["cube1"] == "red"
        assert DEFAULT_COLOR_MAP["cube2"] == "green"
        assert DEFAULT_COLOR_MAP["cube3"] == "blue"

    def test_color_map_length(self):
        assert len(DEFAULT_COLOR_MAP) >= 3

    def test_color_map_lookup_all(self):
        """All standard cube IDs should have colors."""
        for i in range(1, 6):
            key = f"cube{i}"
            assert key in DEFAULT_COLOR_MAP, f"Missing {key}"
            assert isinstance(DEFAULT_COLOR_MAP[key], str)


class TestPerceiveSceneColors:
    """Test that perceive_scene() fills colors from DEFAULT_COLOR_MAP."""

    def test_perceive_scene_fills_colors(self):
        from vlm_nsm.perception.oracle_perception import perceive_scene

        class MockEnv:
            cube_names = ["cube1", "cube2"]
            def get_object_pose(self, name):
                return {"x": 0.1, "y": 0.0, "z": 0.8} if "cube" in name else {"x": 0.0, "y": 0.0, "z": 0.0}
            def get_all_object_poses(self):
                return {}
            def get_gripper_state(self):
                return {"has_object": False}

        scene = perceive_scene(MockEnv())
        cubes = [o for o in scene.objects if o.category == "block"]
        assert len(cubes) == 2

        for cn in MockEnv.cube_names:
            obj = next(o for o in cubes if o.label == cn)
            assert obj.color == DEFAULT_COLOR_MAP[cn], (
                f"{cn} should have color {DEFAULT_COLOR_MAP[cn]}, got {obj.color}"
            )

    def test_perceive_scene_color_in_scene_prompt(self):
        """Scene prompt built from perceive_scene output should include colors."""
        from vlm_nsm.perception.oracle_perception import perceive_scene
        from vlm_nsm.perception.scene_prompt import build_scene_description

        class MockEnv:
            cube_names = ["cube1", "cube2"]
            def get_object_pose(self, name):
                return {"x": 0.1, "y": 0.0, "z": 0.8} if "cube" in name else {"x": 0.0, "y": 0.0, "z": 0.0}
            def get_all_object_poses(self):
                return {}
            def get_gripper_state(self):
                return {"has_object": False}

        scene = perceive_scene(MockEnv())
        desc = build_scene_description(scene)
        assert "color: red" in desc, f"Expected 'color: red' in scene description:\n{desc}"
        assert "color: green" in desc, f"Expected 'color: green' in scene description:\n{desc}"
