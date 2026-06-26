"""SceneState and related dataclasses for unified observation interface."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ObjectState:
    """Perceived or oracle state of a single object."""

    name: str
    class_name: str = ""
    position: np.ndarray | None = None       # (3,) xyz
    orientation: np.ndarray | None = None     # (4,) quaternion or (3,3) rotation
    bbox: np.ndarray | None = None            # (4,) xywh or (4,2) corners
    confidence: float = 1.0
    covariance: np.ndarray | None = None      # (3,3) position covariance
    source: str = "unknown"
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_name,
            "position": self.position.tolist() if self.position is not None else None,
            "orientation": self.orientation.tolist() if self.orientation is not None else None,
            "confidence": self.confidence,
            "source": self.source,
            "available": self.available,
        }


@dataclass
class RobotState:
    """Robot proprioceptive or oracle state."""

    eef_position: np.ndarray | None = None     # (3,) xyz
    eef_orientation: np.ndarray | None = None  # quaternion or rotation matrix
    gripper_width: float | None = None
    joint_positions: np.ndarray | None = None
    source: str = "unknown"
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eef_position": self.eef_position.tolist() if self.eef_position is not None else None,
            "gripper_width": self.gripper_width,
            "source": self.source,
            "available": self.available,
        }


@dataclass
class SceneState:
    """Unified scene observation from a single ObservationSource."""

    objects: dict[str, ObjectState] = field(default_factory=dict)
    robot: RobotState = field(default_factory=RobotState)
    relations: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    frame_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_object(self, name: str) -> ObjectState:
        if name not in self.objects:
            return ObjectState(name=name, available=False, source=self.source)
        return self.objects[name]

    def object_position(self, name: str) -> np.ndarray | None:
        obj = self.get_object(name)
        return obj.position if obj.available else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "robot": self.robot.to_dict(),
            "source": self.source,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
        }


@dataclass
class ObservationLegalityReport:
    """Audit report for an ObservationSource's real-world deployability."""

    observation_source_name: str = ""
    uses_privileged_object_pose: bool = False
    uses_privileged_eef_pose: bool = False
    uses_privileged_contact: bool = False
    uses_perception_object_pose: bool = False
    uses_robot_proprioception: bool = False
    real_world_deployable: bool | str = False
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_source_name": self.observation_source_name,
            "uses_privileged_object_pose": self.uses_privileged_object_pose,
            "uses_privileged_eef_pose": self.uses_privileged_eef_pose,
            "uses_privileged_contact": self.uses_privileged_contact,
            "uses_perception_object_pose": self.uses_perception_object_pose,
            "uses_robot_proprioception": self.uses_robot_proprioception,
            "real_world_deployable": self.real_world_deployable,
            "notes": self.notes,
        }
