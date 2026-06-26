"""Test RGBDetectorPerception spatial metadata output.

No GPU required — uses mock ultralytics YOLO results.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vlm_nsm.perception.detector_perception import RGBDetectorPerception


class MockBox:
    """Mock a single ultralytics Box object."""

    def __init__(self, cls_id: int, conf: float, xyxy: list[float]):
        self.cls = [cls_id]
        self.conf = [conf]
        # Wrap so that xyxy[0].tolist() returns the raw list
        self.xyxy = [_MockTensor(xyxy)]


class _MockTensor:
    """Minimal mock of a torch/tensor with .tolist()."""

    def __init__(self, data: list[float]):
        self._data = data

    def tolist(self) -> list[float]:
        return self._data


class MockBoxes:
    """Mock ultralytics Results.boxes."""

    def __init__(self, boxes: list[MockBox]):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def tolist(self):
        return [b.xyxy[0] for b in self._boxes]


class MockResults:
    """Mock a single ultralytics Results object."""

    def __init__(self, boxes: list[MockBox]):
        self.boxes = MockBoxes(boxes)


class MockYOLOModel:
    """Mock ultralytics YOLO returning synthetic detection results."""

    def __init__(self, boxes: list[MockBox]):
        self._boxes = boxes
        self.ckpt_path = "mock.pt"

    def __call__(self, rgb_image, conf=0.5, verbose=False):
        return [MockResults(self._boxes)]


def _make_detector(boxes: list[MockBox]) -> RGBDetectorPerception:
    model = MockYOLOModel(boxes)
    detector = RGBDetectorPerception.__new__(RGBDetectorPerception)
    detector.model = model
    detector.conf_threshold = 0.15
    detector.class_map = {
        0: ("cube1", "red"),
        1: ("cube2", "green"),
        2: ("cube3", "blue"),
    }
    return detector


class TestRGBDetectorPerceptionSpatial:
    def test_center_norm_added(self):
        """Each detected object gets center_norm and center_px."""
        boxes = [
            MockBox(cls_id=0, conf=0.9, xyxy=[50, 100, 150, 200]),
            MockBox(cls_id=1, conf=0.8, xyxy=[300, 120, 450, 220]),
        ]
        detector = _make_detector(boxes)
        scene = detector.perceive(
            rgb_image=np.zeros((512, 512, 3), dtype=np.uint8),
            width=512,
            height=512,
        )
        assert len(scene.objects) == 2

        obj0 = scene.objects[0]
        assert obj0.label == "cube1"
        assert obj0.color == "red"
        assert obj0.pose is None
        assert obj0.attributes["center_norm"] == {"x": 100.0 / 512, "y": 150.0 / 512}
        assert obj0.attributes["center_px"] == {"x": 100.0, "y": 150.0}
        assert obj0.attributes["position_source"] == "unknown_rgb_detector_3d"
        assert obj0.attributes["bbox_source"] == "detector"
        assert obj0.confidence == 0.9

        obj1 = scene.objects[1]
        assert obj1.label == "cube2"
        assert obj1.color == "green"
        assert obj1.pose is None
        assert obj1.attributes["center_norm"] == {"x": 375.0 / 512, "y": 170.0 / 512}
        assert obj1.attributes["center_px"] == {"x": 375.0, "y": 170.0}

    def test_pose_is_none_not_zero(self):
        """pose must be None, not (0,0,0)."""
        boxes = [MockBox(cls_id=0, conf=0.9, xyxy=[10, 20, 100, 200])]
        detector = _make_detector(boxes)
        scene = detector.perceive(
            rgb_image=np.zeros((256, 256, 3), dtype=np.uint8),
            width=256,
            height=256,
        )
        assert scene.objects[0].pose is None

    def test_empty_detection(self):
        """No boxes returns empty SceneInfo."""
        detector = _make_detector([])
        scene = detector.perceive(
            rgb_image=np.zeros((512, 512, 3), dtype=np.uint8),
            width=512,
            height=512,
        )
        assert len(scene.objects) == 0

    def test_different_image_size(self):
        """center_norm should be in [0,1] for any image size."""
        boxes = [MockBox(cls_id=2, conf=0.5, xyxy=[100, 200, 300, 400])]
        detector = _make_detector(boxes)
        scene = detector.perceive(
            rgb_image=np.zeros((800, 600, 3), dtype=np.uint8),
            width=600,
            height=800,
        )
        cn = scene.objects[0].attributes["center_norm"]
        assert 0 <= cn["x"] <= 1
        assert 0 <= cn["y"] <= 1
        assert cn["x"] == 200.0 / 600
        assert cn["y"] == 300.0 / 800

    def test_scene_source_and_gripper(self):
        """SceneInfo source must be rgb_detector, gripper free."""
        boxes = [MockBox(cls_id=0, conf=0.9, xyxy=[0, 0, 50, 50])]
        detector = _make_detector(boxes)
        scene = detector.perceive(
            rgb_image=np.zeros((512, 512, 3), dtype=np.uint8),
            width=512,
            height=512,
        )
        assert scene.source == "rgb_detector"
        assert scene.free_gripper is True
        assert scene.grasped_object is None

    def test_confidence_propagated(self):
        """Detector confidence appears in DetectedObject.confidence."""
        boxes = [MockBox(cls_id=1, conf=0.723, xyxy=[10, 10, 60, 80])]
        detector = _make_detector(boxes)
        scene = detector.perceive(
            rgb_image=np.zeros((512, 512, 3), dtype=np.uint8),
            width=512,
            height=512,
        )
        assert scene.objects[0].confidence == 0.723
