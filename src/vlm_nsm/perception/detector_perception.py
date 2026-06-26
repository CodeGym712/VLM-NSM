"""Vision-based perception using a YOLO detector.

Replaces oracle_perception with actual RGB object detection
for scene grounding evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.bbox_utils import BBox


# color_cube mode: 0=red_cube, 1=green_cube, 2=blue_cube
COLOR_CUBE_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("cube1", "red"),
    1: ("cube2", "green"),
    2: ("cube3", "blue"),
}


class RGBDetectorPerception:
    """Perception module using YOLOv8 object detection.

    Runs inference on RGB images and returns SceneInfo with
    detected blocks, their colors, and bounding boxes.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        class_map: dict[int, tuple[str, str]] | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "RGBDetectorPerception requires 'ultralytics' package"
            ) from exc

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.class_map = class_map or COLOR_CUBE_CLASS_MAP

    def perceive(
        self,
        rgb_image: Any,
        width: int,
        height: int,
    ) -> SceneInfo:
        """Detect objects in an RGB image.

        Args:
            rgb_image: numpy array (H, W, 3) or PIL Image.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            SceneInfo with detected objects.
        """
        results = self.model(rgb_image, conf=self.conf_threshold, verbose=False)[0]

        objects: list[DetectedObject] = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                label, color = self.class_map.get(cls_id, (f"object_{cls_id}", None))
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                objects.append(DetectedObject(
                    label=label,
                    category="block",
                    color=color,
                    pose=None,  # no 3D pose from RGB-only detector
                    confidence=conf,
                    attributes={
                        "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2},
                        "center_px": {"x": center_x, "y": center_y},
                        "center_norm": {"x": center_x / width, "y": center_y / height},
                        "bbox_source": "detector",
                        "position_source": "unknown_rgb_detector_3d",
                    },
                ))

        return SceneInfo(
            objects=objects,
            on_relations=[],
            free_gripper=True,
            source="rgb_detector",
        )

    @property
    def model_name(self) -> str:
        """Return the model file name for logging."""
        return Path(self.model.ckpt_path).name if hasattr(self.model, 'ckpt_path') and self.model.ckpt_path else "unknown"
