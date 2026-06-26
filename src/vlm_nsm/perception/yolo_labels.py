"""YOLO label generation utilities for RGB perception dataset.

Supports two class modes:
- color_cube: 0=red_cube, 1=green_cube, 2=blue_cube
- cube: 0=cube (all blocks treated as single class)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vlm_nsm.perception.bbox_utils import BBox, bbox_to_yolo, validate_bbox
from vlm_nsm.perception.data_types import DEFAULT_COLOR_MAP, DetectedObject

ClassMode = Literal["color_cube", "cube"]

COLOR_CUBE_CLASSES: dict[str, int] = {
    "red": 0,
    "green": 1,
    "blue": 2,
}

CUBE_CLASS = 0  # single class for all cubes


def class_id_for_object(
    obj: DetectedObject,
    class_mode: ClassMode = "color_cube",
) -> int | None:
    """Determine YOLO class ID for a detected object.

    Args:
        obj: Detected object.
        class_mode: 'color_cube' for color-specific classes, 'cube' for single class.

    Returns:
        Integer class ID, or None if object has no color (color_cube mode).
    """
    if class_mode == "cube":
        if obj.category == "block":
            return CUBE_CLASS
        return None  # skip non-block objects
    if class_mode == "color_cube":
        if obj.category != "block":
            return None
        if not obj.color:
            return None
        return COLOR_CUBE_CLASSES.get(obj.color.lower())
    return None


def write_yolo_label(
    label_path: str | Path,
    objects: list[tuple[DetectedObject, BBox]],
    image_width: int,
    image_height: int,
    class_mode: ClassMode = "color_cube",
) -> list[str]:
    """Write a YOLO-format label file for one image.

    Each line: <class_id> <cx> <cy> <w> <h>
    All coordinates normalized to [0, 1].

    Args:
        label_path: Output .txt file path.
        objects: List of (DetectedObject, BBox) pairs in the image.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        class_mode: Class mapping mode.

    Returns:
        List of warning messages (e.g. skipped objects).
    """
    warnings: list[str] = []
    lines: list[str] = []

    for obj, bbox in objects:
        class_id = class_id_for_object(obj, class_mode)
        if class_id is None:
            warnings.append(
                f"Skipping {obj.label} (category={obj.category}, color={obj.color}): "
                f"no class mapping for mode={class_mode}"
            )
            continue

        valid, reason = validate_bbox(bbox, image_width, image_height)
        if not valid:
            clamped_bbox = BBox(
                x1=max(0, bbox.x1), y1=max(0, bbox.y1),
                x2=min(image_width - 1, bbox.x2), y2=min(image_height - 1, bbox.y2),
            )
            valid2, _ = validate_bbox(clamped_bbox, image_width, image_height)
            if not valid2:
                warnings.append(
                    f"Skipping {obj.label}: bbox invalid even after clamp ({reason})"
                )
                continue
            bbox = clamped_bbox
            warnings.append(f"Clamped bbox for {obj.label}: {reason}")

        cx, cy, w, h = bbox_to_yolo(bbox, image_width, image_height)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return warnings


def write_yolo_yaml(
    dataset_root: str | Path,
    class_mode: ClassMode = "color_cube",
    train_rel: str = "images/train",
    val_rel: str = "images/val",
    test_rel: str | None = None,
) -> str:
    """Write a YOLO dataset YAML configuration file.

    Args:
        dataset_root: Root directory of the dataset.
        class_mode: 'color_cube' or 'cube'.
        train_rel: Relative path to training images.
        val_rel: Relative path to validation images.
        test_rel: Optional relative path to test images.

    Returns:
        Path to the written YAML file as string.
    """
    if class_mode == "color_cube":
        names = {0: "red_cube", 1: "green_cube", 2: "blue_cube"}
        nc = 3
    else:
        names = {0: "cube"}
        nc = 1

    root = Path(dataset_root).resolve()
    lines = [
        f"# YOLO dataset config for {class_mode} mode",
        f"path: {root.as_posix()}",
        f"train: {train_rel}",
        f"val: {val_rel}",
    ]
    if test_rel:
        lines.append(f"test: {test_rel}")
    lines += [
        "",
        f"nc: {nc}",
        f"names: {names}",
    ]

    yaml_path = root / "yolo.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    return str(yaml_path)
