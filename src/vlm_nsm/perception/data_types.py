"""Data types for scene perception output.

These types represent the output of a scene-understanding module
(e.g. vision-based object detection) and serve as the interface
between perception and the scene-aware parser adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_COLOR_MAP: dict[str, str] = {
    "cube1": "red",
    "cube2": "green",
    "cube3": "blue",
    "cube4": "yellow",
    "cube5": "purple",
}


@dataclass
class DetectedObject:
    """An object detected in the scene."""

    label: str
    category: str  # e.g. "block", "table", "peg"
    pose: dict[str, float] | None = None  # {"x": ..., "y": ..., "z": ...}
    color: str | None = None
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def object_id(self) -> str:
        """Alias for label — used in grounding evaluation."""
        return self.label

    @object_id.setter
    def object_id(self, value: str) -> None:
        self.label = value


@dataclass
class SceneInfo:
    """Full scene description from perception."""

    objects: list[DetectedObject] = field(default_factory=list)
    on_relations: list[tuple[str, str]] = field(default_factory=list)
    free_gripper: bool = True
    grasped_object: str | None = None
    source: str = "oracle"  # "oracle", "mock", "vision"
    camera_name: str | None = None

    def get_objects_by_category(self, category: str) -> list[DetectedObject]:
        return [o for o in self.objects if o.category == category]

    def get_object_labels(self) -> list[str]:
        return [o.label for o in self.objects]
