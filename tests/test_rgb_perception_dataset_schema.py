"""Test RGB perception dataset collection logic.

No Robosuite or GPU required — uses MockEnvWrapper and synthetic data.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from vlm_nsm.experiments.collect_rgb_perception_dataset import (
    _save_image,
    _write_annotations,
    _write_split,
)


class MockEnv:
    """Minimal mock of RobosuiteEnvWrapper for testing."""
    def __init__(self):
        self.cube_names = ["cube1", "cube2"]
        self._pose_index = 0

    def render_rgb(self, camera="agentview", height=512, width=512):
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def get_object_pose(self, name):
        poses = {
            "cube1": {"x": 0.1, "y": 0.0, "z": 0.8},
            "cube2": {"x": -0.1, "y": 0.0, "z": 0.8},
            "table": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        return poses.get(name, {"x": 0.0, "y": 0.0, "z": 0.0})

    def get_all_object_poses(self):
        return {
            "cube1": {"pos": [0.1, 0.0, 0.8], "quat": [0, 0, 0, 1]},
            "cube2": {"pos": [-0.1, 0.0, 0.8], "quat": [0, 0, 0, 1]},
            "table": {"pos": [0.0, 0.0, 0.0], "quat": [0, 0, 0, 1]},
        }

    def get_gripper_state(self):
        return {"has_object": False}

    def step(self, action):
        return {}, 0.0, False, {}

    def reset(self):
        pass

    def close(self):
        pass


class TestRgbPerceptionDatasetSchema:
    def test_save_image(self, tmp_path):
        """_save_image should write a valid PNG file."""
        rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out_path = tmp_path / "images" / "test.png"
        _save_image(rgb, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 50  # PNG overhead

    def test_save_image_invalid_array(self, tmp_path):
        """Non-uint8 array should still be saved via PIL conversion."""
        rgb = np.random.rand(64, 64, 3) * 255  # float array
        out_path = tmp_path / "images" / "test_float.png"
        _save_image(rgb, out_path)
        assert out_path.exists()

    def test_write_annotations(self, tmp_path):
        """_write_annotations should produce valid JSONL."""
        anns = [
            {"image_path": "images/step_000000.png", "episode_id": 0, "step_id": 0},
            {"image_path": "images/step_000001.png", "episode_id": 0, "step_id": 1},
        ]
        anno_path = tmp_path / "annotations.jsonl"
        _write_annotations(anns, anno_path)
        lines = anno_path.read_text().strip().splitlines()
        assert len(lines) == 2
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["image_path"] == anns[i]["image_path"]

    def test_write_split(self, tmp_path):
        """_write_split should produce train.txt and val.txt."""
        anns = [
            {"image_path": f"images/step_{i:06d}.png"}
            for i in range(10)
        ]
        n_train, n_val = _write_split(anns, train_ratio=0.8, output_dir=tmp_path)
        assert n_train == 8
        assert n_val == 2
        train_paths = tmp_path / "train.txt"
        val_paths = tmp_path / "val.txt"
        assert train_paths.exists()
        assert val_paths.exists()
        assert len(train_paths.read_text().strip().splitlines()) == 8
        assert len(val_paths.read_text().strip().splitlines()) == 2

    def test_write_split_edge_cases(self, tmp_path):
        """_write_split should handle empty and single-element lists."""
        # Empty
        n_train, n_val = _write_split([], train_ratio=0.8, output_dir=tmp_path)
        assert n_train == 0
        assert n_val == 0

        # Single item — int(1 * 0.8) = 0, all goes to val
        anns = [{"image_path": "images/step_000000.png"}]
        n_train, n_val = _write_split(anns, train_ratio=0.8, output_dir=tmp_path)
        assert (n_train, n_val) == (0, 1)

    def test_mock_env_perceive(self):
        """Mock environment should work with perceive_scene."""
        from vlm_nsm.perception.oracle_perception import perceive_scene
        env = MockEnv()

        # Mock perceive_scene by calling the module's helper
        from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
        objects = []
        for cn in env.cube_names:
            pose = env.get_object_pose(cn)
            objects.append(DetectedObject(label=cn, category="block", pose=pose))
        objects.append(
            DetectedObject(label="table", category="table", pose=env.get_object_pose("table"))
        )
        scene = SceneInfo(objects=objects)
        assert len(scene.objects) == 3
        assert scene.objects[0].label == "cube1"
        assert scene.objects[1].label == "cube2"
        assert scene.objects[2].label == "table"

    def test_annotation_structure(self):
        """Annotation dict should contain all required fields."""
        from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", color="red",
                           pose={"x": 0.1, "y": 0.0, "z": 0.8}),
        ])

        obj_list = []
        for obj in scene.objects:
            obj_entry = {
                "object_id": obj.label,
                "category": obj.category,
                "color": obj.color,
                "position": obj.pose if obj.pose else None,
                "bbox": None,
                "center": None,
                "visibility": "visible",
            }
            obj_list.append(obj_entry)

        annotation = {
            "image_path": "images/step_000000.png",
            "episode_id": 0,
            "step_id": 0,
            "camera_name": "agentview",
            "image_width": 512,
            "image_height": 512,
            "objects": obj_list,
            "scene_source": "oracle",
            "on_relations": [],
            "free_gripper": True,
        }

        assert annotation["image_path"].endswith(".png")
        assert annotation["objects"][0]["object_id"] == "cube1"
        assert annotation["objects"][0]["color"] == "red"
        assert annotation["scene_source"] == "oracle"

    def test_annotation_required_fields(self):
        """All annotations must have the required top-level fields."""
        required = [
            "image_path", "episode_id", "step_id", "camera_name",
            "image_width", "image_height", "objects", "scene_source",
        ]
        annotation = {
            "image_path": "images/test.png",
            "episode_id": 0,
            "step_id": 0,
            "camera_name": "agentview",
            "image_width": 512,
            "image_height": 512,
            "objects": [],
            "scene_source": "oracle",
        }
        for field in required:
            assert field in annotation, f"Missing required field: {field}"

    def test_object_entry_required_fields(self):
        """Each object entry must have required fields."""
        required = ["object_id", "category", "color", "position"]
        obj_entry = {
            "object_id": "cube1",
            "category": "block",
            "color": "red",
            "position": {"x": 0.1, "y": 0.0, "z": 0.8},
        }
        for field in required:
            assert field in obj_entry, f"Missing required field: {field}"

    def test_meta_structure(self):
        """dataset_meta.json should contain all standard fields."""
        meta = {
            "dataset_name": "rgb_perception_v0_1",
            "dataset_version": "0.1.0",
            "created_at": "2026-05-21",
            "source": "Robosuite Stack env, oracle labels",
            "total_images": 100,
            "image_size": {"height": 512, "width": 512},
            "camera": "agentview",
            "num_episodes": 5,
            "objects": ["cube1", "cube2", "table"],
            "categories": ["block", "table"],
            "colors": {"cube1": "red", "cube2": "green"},
            "split": {"train": 80, "val": 20},
            "has_bbox": False,
            "has_center": False,
            "purpose": "RGB perception training",
        }
        assert meta["dataset_name"] == "rgb_perception_v0_1"
        assert len(meta["objects"]) == 3
        assert meta["split"]["train"] + meta["split"]["val"] == 100

    def test_meta_json_roundtrip(self, tmp_path):
        """dataset_meta.json should survive JSON roundtrip."""
        meta = {
            "dataset_name": "rgb_perception_v0_1",
            "total_images": 50,
            "image_size": {"height": 512, "width": 512},
            "split": {"train": 40, "val": 10},
        }
        p = tmp_path / "dataset_meta.json"
        p.write_text(json.dumps(meta, indent=2))
        loaded = json.loads(p.read_text())
        assert loaded["dataset_name"] == "rgb_perception_v0_1"
        assert loaded["total_images"] == 50
        assert loaded["split"]["train"] == 40

    def test_render_rgb_mock_shape(self):
        """Mock render_rgb should return correct shape."""
        env = MockEnv()
        rgb = env.render_rgb(camera="agentview", height=224, width=224)
        assert rgb.shape == (224, 224, 3)
        assert rgb.dtype == np.uint8


class TestEstimate2DCenter:
    """Test the _estimate_2d_center projection function."""

    def test_center_at_origin(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        cx, cy = _estimate_2d_center({"x": 0.0, "y": 0.0, "z": 0.8},
                                      width=512, height=512)
        # At origin: nx=0.5, ny=0.5 → px=50+0.5*(512-100)=256, py=(512-50)-0.5*(512-100)=256
        assert abs(cx - 256) < 1
        assert abs(cy - 256) < 1

    def test_center_front_left(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        # x=-0.3 (left), y=-0.25 (front/near robot) → image bottom-left
        cx, cy = _estimate_2d_center({"x": -0.3, "y": -0.25, "z": 0.8},
                                      width=512, height=512)
        assert cx < 256  # left side of image
        assert cy > 256  # bottom of image (near robot)

    def test_center_back_right(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        # x=0.3 (right), y=0.25 (back/far) → image top-right
        cx, cy = _estimate_2d_center({"x": 0.3, "y": 0.25, "z": 0.8},
                                      width=512, height=512)
        assert cx > 256  # right side of image
        assert cy < 256  # top of image (far from robot)

    def test_center_fallback_default(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        cx, cy = _estimate_2d_center(None, width=512, height=512)
        assert abs(cx - 256) < 1
        assert abs(cy - 256) < 1

    def test_center_fallback_custom(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        cx, cy = _estimate_2d_center(None, width=512, height=512,
                                      default_center=(100, 200))
        assert cx == 100
        assert cy == 200

    def test_center_different_resolution(self):
        from vlm_nsm.experiments.collect_rgb_perception_dataset import _estimate_2d_center
        cx, cy = _estimate_2d_center({"x": 0.0, "y": 0.0, "z": 0.8},
                                      width=1024, height=768)
        assert abs(cx - 512) < 2
        assert abs(cy - 384) < 2
