"""Bounding box utility functions for RGB perception dataset generation.

Provides coordinate clamping, center/YOLO format conversions, and
approximate bbox generation from projected center points.
All bbox values are in pixel coordinates (x, y) with origin at top-left.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: float  # left
    y1: float  # top
    x2: float  # right
    y2: float  # bottom


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Clamp bbox coordinates to image boundaries."""
    return BBox(
        x1=max(0.0, min(bbox.x1, width - 1)),
        y1=max(0.0, min(bbox.y1, height - 1)),
        x2=max(0.0, min(bbox.x2, width - 1)),
        y2=max(0.0, min(bbox.y2, height - 1)),
    )


def bbox_to_center(bbox: BBox) -> tuple[float, float]:
    """Convert bbox to (cx, cy) center coordinates in pixels."""
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0
    return (cx, cy)


def bbox_to_yolo(bbox: BBox, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert pixel bbox to normalized YOLO format: (cx, cy, w, h).

    All values normalized to [0, 1] relative to image dimensions.
    """
    cx = (bbox.x1 + bbox.x2) / (2.0 * width)
    cy = (bbox.y1 + bbox.y2) / (2.0 * height)
    w = (bbox.x2 - bbox.x1) / width
    h = (bbox.y2 - bbox.y1) / height
    return (cx, cy, w, h)


def yolo_to_bbox(
    cx_norm: float, cy_norm: float, w_norm: float, h_norm: float,
    width: int, height: int,
) -> BBox:
    """Convert normalized YOLO format back to pixel bbox."""
    x1 = (cx_norm - w_norm / 2.0) * width
    y1 = (cy_norm - h_norm / 2.0) * height
    x2 = (cx_norm + w_norm / 2.0) * width
    y2 = (cy_norm + h_norm / 2.0) * height
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def approximate_bbox_from_center(
    center_px: tuple[float, float],
    box_size: tuple[float, float],
    width: int,
    height: int,
) -> BBox:
    """Generate a bbox from a center point and fixed box dimensions.

    Useful when only object center is known (e.g. from projection)
    and a fixed-size box is needed as approximation.

    Args:
        center_px: (cx, cy) center in pixel coordinates.
        box_size: (w, h) of the box in pixels.
        width: Image width for clamping.
        height: Image height for clamping.

    Returns:
        Clamped BBox.
    """
    bw, bh = box_size
    cx, cy = center_px
    bbox = BBox(
        x1=cx - bw / 2.0,
        y1=cy - bh / 2.0,
        x2=cx + bw / 2.0,
        y2=cy + bh / 2.0,
    )
    return clamp_bbox(bbox, width, height)


def validate_bbox(bbox: BBox, width: int, height: int) -> tuple[bool, str]:
    """Validate that a bbox is reasonable.

    Returns:
        (is_valid, reason) tuple.
    """
    if bbox.x2 <= bbox.x1:
        return False, "x2 <= x1 (zero or negative width)"
    if bbox.y2 <= bbox.y1:
        return False, "y2 <= y1 (zero or negative height)"
    if bbox.x1 < 0 or bbox.y1 < 0:
        return False, "negative coordinates"
    if bbox.x2 > width or bbox.y2 > height:
        return False, f"coordinates exceed image dimensions ({width}x{height})"
    return True, "valid"
