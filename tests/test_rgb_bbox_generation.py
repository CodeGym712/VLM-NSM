"""Test bbox utility functions.

No Robosuite or GPU required — pure unit tests.
"""

import pytest
from vlm_nsm.perception.bbox_utils import (
    BBox,
    clamp_bbox,
    bbox_to_center,
    bbox_to_yolo,
    yolo_to_bbox,
    approximate_bbox_from_center,
    validate_bbox,
)


class TestBBoxUtils:
    def test_bbox_to_center(self):
        bbox = BBox(x1=100, y1=50, x2=200, y2=150)
        cx, cy = bbox_to_center(bbox)
        assert cx == 150.0
        assert cy == 100.0

    def test_bbox_to_yolo_and_back(self):
        bbox = BBox(x1=100, y1=50, x2=300, y2=150)
        cx, cy, w, h = bbox_to_yolo(bbox, width=640, height=480)
        # Reconstruct
        restored = yolo_to_bbox(cx, cy, w, h, width=640, height=480)
        assert abs(restored.x1 - 100) < 1e-9
        assert abs(restored.y1 - 50) < 1e-9
        assert abs(restored.x2 - 300) < 1e-9
        assert abs(restored.y2 - 150) < 1e-9

    def test_yolo_normalization(self):
        """YOLO values should be normalized to [0, 1]."""
        bbox = BBox(x1=160, y1=120, x2=480, y2=360)
        cx, cy, w, h = bbox_to_yolo(bbox, width=640, height=480)
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1

    def test_clamp_bbox_inside(self):
        """Clamping should not modify already-valid bbox."""
        bbox = BBox(x1=10, y1=10, x2=100, y2=100)
        clamped = clamp_bbox(bbox, width=640, height=480)
        assert clamped.x1 == 10
        assert clamped.y1 == 10
        assert clamped.x2 == 100
        assert clamped.y2 == 100

    def test_clamp_bbox_overflow(self):
        """Clamping should cap at image boundaries."""
        bbox = BBox(x1=-10, y1=-20, x2=700, y2=500)
        clamped = clamp_bbox(bbox, width=640, height=480)
        assert clamped.x1 == 0
        assert clamped.y1 == 0
        assert clamped.x2 == 639
        assert clamped.y2 == 479

    def test_approximate_bbox_from_center(self):
        bbox = approximate_bbox_from_center(
            center_px=(320, 240),
            box_size=(100, 80),
            width=640,
            height=480,
        )
        assert bbox.x1 == 270  # 320 - 50
        assert bbox.y1 == 200  # 240 - 40
        assert bbox.x2 == 370  # 320 + 50
        assert bbox.y2 == 280  # 240 + 40

    def test_approximate_bbox_clamps(self):
        """Approximate bbox at edge should clamp."""
        bbox = approximate_bbox_from_center(
            center_px=(5, 5),
            box_size=(100, 100),
            width=640,
            height=480,
        )
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert bbox.x2 > 0
        assert bbox.y2 > 0

    def test_validate_bbox_valid(self):
        bbox = BBox(x1=100, y1=50, x2=200, y2=150)
        valid, reason = validate_bbox(bbox, width=640, height=480)
        assert valid is True
        assert reason == "valid"

    def test_validate_bbox_negative(self):
        bbox = BBox(x1=-10, y1=50, x2=200, y2=150)
        valid, reason = validate_bbox(bbox, width=640, height=480)
        assert valid is False
        assert "negative" in reason

    def test_validate_bbox_zero_width(self):
        bbox = BBox(x1=100, y1=50, x2=100, y2=150)
        valid, reason = validate_bbox(bbox, width=640, height=480)
        assert valid is False

    def test_validate_bbox_exceeds_image(self):
        bbox = BBox(x1=100, y1=50, x2=700, y2=150)
        valid, reason = validate_bbox(bbox, width=640, height=480)
        assert valid is False

    def test_center_symmetry(self):
        """Center of symmetric bbox should be at midpoint."""
        bbox = BBox(x1=0, y1=0, x2=640, y2=480)
        cx, cy = bbox_to_center(bbox)
        assert cx == 320.0
        assert cy == 240.0

    def test_yolo_roundtrip_small(self):
        """Very small bbox should survive YOLO roundtrip."""
        bbox = BBox(x1=300, y1=200, x2=310, y2=210)
        cx, cy, w, h = bbox_to_yolo(bbox, width=640, height=480)
        restored = yolo_to_bbox(cx, cy, w, h, width=640, height=480)
        assert abs(restored.x1 - 300) < 1e-6
        assert abs(restored.y1 - 200) < 1e-6
