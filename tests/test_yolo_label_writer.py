"""Test YOLO label writer functions.

No Robosuite or GPU required — pure unit tests.
"""

import json
import pytest
from pathlib import Path

from vlm_nsm.perception.data_types import DetectedObject
from vlm_nsm.perception.bbox_utils import BBox
from vlm_nsm.perception.yolo_labels import (
    class_id_for_object,
    write_yolo_label,
    write_yolo_yaml,
)


class TestYoloLabelWriter:
    def test_class_id_color_cube_red(self):
        obj = DetectedObject(label="cube1", category="block", color="red")
        assert class_id_for_object(obj, class_mode="color_cube") == 0

    def test_class_id_color_cube_green(self):
        obj = DetectedObject(label="cube2", category="block", color="green")
        assert class_id_for_object(obj, class_mode="color_cube") == 1

    def test_class_id_color_cube_blue(self):
        obj = DetectedObject(label="cube3", category="block", color="blue")
        assert class_id_for_object(obj, class_mode="color_cube") == 2

    def test_class_id_color_cube_no_color(self):
        obj = DetectedObject(label="cube1", category="block", color=None)
        assert class_id_for_object(obj, class_mode="color_cube") is None

    def test_class_id_cube_mode(self):
        obj = DetectedObject(label="cube1", category="block", color="red")
        assert class_id_for_object(obj, class_mode="cube") == 0

    def test_class_id_cube_mode_table(self):
        obj = DetectedObject(label="table", category="table")
        assert class_id_for_object(obj, class_mode="cube") is None

    def test_write_yolo_label_basic(self, tmp_path):
        label_path = tmp_path / "labels" / "img_000001.txt"
        objs = [
            (DetectedObject(label="cube1", category="block", color="red"),
             BBox(x1=100, y1=50, x2=200, y2=150)),
            (DetectedObject(label="cube2", category="block", color="green"),
             BBox(x1=300, y1=50, x2=400, y2=150)),
        ]
        warnings = write_yolo_label(
            label_path, objs, image_width=640, image_height=480, class_mode="color_cube",
        )
        assert len(warnings) == 0
        assert label_path.exists()
        content = label_path.read_text().strip()
        lines = content.splitlines()
        assert len(lines) == 2
        # First line should be red cube (class 0) with normalized coords
        parts = lines[0].split()
        assert parts[0] == "0"  # red_cube class
        assert 0 <= float(parts[1]) <= 1
        assert 0 <= float(parts[2]) <= 1

    def test_write_yolo_label_cube_mode(self, tmp_path):
        label_path = tmp_path / "img_000001.txt"
        objs = [
            (DetectedObject(label="cube1", category="block", color="red"),
             BBox(x1=100, y1=50, x2=200, y2=150)),
        ]
        warnings = write_yolo_label(
            label_path, objs, image_width=640, image_height=480, class_mode="cube",
        )
        assert len(warnings) == 0
        content = label_path.read_text().strip()
        assert content.startswith("0 ")  # single cube class

    def test_write_yolo_label_skips_table(self, tmp_path):
        label_path = tmp_path / "img.txt"
        objs = [
            (DetectedObject(label="table", category="table", color=None),
             BBox(x1=0, y1=0, x2=640, y2=480)),
        ]
        warnings = write_yolo_label(
            label_path, objs, image_width=640, image_height=480, class_mode="color_cube",
        )
        # Table should be skipped
        assert len(warnings) > 0
        assert "Skipping" in warnings[0]
        # File should still exist but be empty
        assert label_path.exists()
        assert label_path.read_text().strip() == ""

    def test_write_yolo_label_handles_missing_color(self, tmp_path):
        label_path = tmp_path / "img.txt"
        objs = [
            (DetectedObject(label="cube1", category="block", color=None),
             BBox(x1=100, y1=50, x2=200, y2=150)),
        ]
        warnings = write_yolo_label(
            label_path, objs, image_width=640, image_height=480, class_mode="color_cube",
        )
        assert len(warnings) > 0
        assert "Skipping" in warnings[0]

    def test_write_yolo_label_invalid_bbox(self, tmp_path):
        label_path = tmp_path / "img.txt"
        objs = [
            (DetectedObject(label="cube1", category="block", color="red"),
             BBox(x1=100, y1=50, x2=50, y2=150)),  # x2 < x1
        ]
        warnings = write_yolo_label(
            label_path, objs, image_width=640, image_height=480, class_mode="color_cube",
        )
        assert len(warnings) > 0

    def test_yolo_yaml_color_cube(self, tmp_path):
        yaml_path = write_yolo_yaml(tmp_path, class_mode="color_cube")
        content = Path(yaml_path).read_text()
        assert "nc: 3" in content
        assert "red_cube" in content
        assert "green_cube" in content
        assert "blue_cube" in content
        assert "train: images/train" in content

    def test_yolo_yaml_cube(self, tmp_path):
        yaml_path = write_yolo_yaml(tmp_path, class_mode="cube")
        content = Path(yaml_path).read_text()
        assert "nc: 1" in content
        assert "cube" in content
        assert "red_cube" not in content

    def test_yolo_yaml_with_test(self, tmp_path):
        yaml_path = write_yolo_yaml(tmp_path, class_mode="color_cube", test_rel="images/test")
        content = Path(yaml_path).read_text()
        assert "test: images/test" in content

    def test_yolo_label_normalization_values(self, tmp_path):
        """Verify YOLO coords are properly normalized."""
        label_path = tmp_path / "norm.txt"
        objs = [
            (DetectedObject(label="cube1", category="block", color="red"),
             BBox(x1=0, y1=0, x2=640, y2=480)),  # full image (exclusive convention)
        ]
        write_yolo_label(label_path, objs, image_width=640, image_height=480, class_mode="color_cube")
        content = label_path.read_text().strip()
        parts = content.split()
        # cx = (0 + 640) / (2 * 640) = 0.5
        assert abs(float(parts[1]) - 0.5) < 1e-6
        # cy = (0 + 480) / (2 * 480) = 0.5
        assert abs(float(parts[2]) - 0.5) < 1e-6
        # w = (640 - 0) / 640 = 1.0
        assert abs(float(parts[3]) - 1.0) < 1e-6
        # h = (480 - 0) / 480 = 1.0
        assert abs(float(parts[4]) - 1.0) < 1e-6
