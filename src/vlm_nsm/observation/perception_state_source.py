"""PerceptionStateSource — adapter for Stage 18 vision pipeline outputs.

Currently a stub that documents missing fields.
Will be completed when perception→SceneState conversion is implemented.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.observation.observation_source import ObservationSource
from vlm_nsm.observation.scene_state import (
    ObjectState, RobotState, SceneState, ObservationLegalityReport,
)


class PerceptionStateSource(ObservationSource):
    """Adapter that converts Stage 18 perception outputs into SceneState.

    Currently in STUB mode: returns available=false for all objects.
    Logs loaded from perception JSONL/JSON if available.
    """

    def __init__(
        self,
        cubes: list[str] | None = None,
        perception_log_path: str | None = None,
    ):
        self._cubes = cubes or ["cube1", "cube2"]
        self._perception_data: list[dict[str, Any]] = []
        self._frame_idx = 0

        if perception_log_path and Path(perception_log_path).exists():
            try:
                raw = Path(perception_log_path).read_text().strip()
                if perception_log_path.endswith(".jsonl"):
                    self._perception_data = [json.loads(l) for l in raw.split("\n") if l.strip()]
                else:
                    self._perception_data = [json.loads(raw)]
            except Exception:
                pass

    def reset(self, env: Any) -> None:
        self._frame_idx = 0

    def get_scene_state(self, env: Any, camera_obs: Any = None) -> SceneState:
        objects: dict[str, ObjectState] = {}

        # Try to load from perception data
        perception_entry = None
        if self._frame_idx < len(self._perception_data):
            perception_entry = self._perception_data[self._frame_idx]
        self._frame_idx += 1

        for name in self._cubes:
            pos = None
            conf = 0.0
            avail = False

            if perception_entry is not None:
                detections = perception_entry.get("detections") or perception_entry.get("objects") or {}
                det = detections.get(name) if isinstance(detections, dict) else None
                if det is not None:
                    if isinstance(det, dict):
                        p = det.get("position") or det.get("xyz") or det.get("triangulated_3d")
                        if p is not None:
                            pos = np.array(p, dtype=np.float32).ravel()[:3]
                            avail = True
                        conf = float(det.get("confidence", 0.0))
                    elif isinstance(det, (list, tuple)):
                        pos = np.array(det, dtype=np.float32).ravel()[:3]
                        avail = True

            objects[name] = ObjectState(
                name=name, class_name=name, position=pos,
                confidence=conf, source="perception_stage18",
                available=avail,
                metadata={"has_perception_log": bool(perception_entry)},
            )

        # Robot: use FK from env (robot proprioception is not privileged)
        robot = RobotState(source="robot_proprioception", available=True)
        try:
            eef = env.get_eef_pose()
            robot.eef_position = np.array([eef["x"], eef["y"], eef["z"]], dtype=np.float32)
            robot.metadata["eef_source"] = "robot_fk"
        except Exception:
            robot.available = False

        return SceneState(
            objects=objects, robot=robot, source="perception_stage18",
            timestamp=time.time(),
            metadata={"perception_loaded": bool(perception_entry)},
        )

    def get_legality_report(self) -> ObservationLegalityReport:
        has_perception = len(self._perception_data) > 0
        return ObservationLegalityReport(
            observation_source_name="PerceptionStateSource",
            uses_privileged_object_pose=False,
            uses_privileged_eef_pose=False,
            uses_privileged_contact=False,
            uses_perception_object_pose=has_perception,
            uses_robot_proprioception=True,
            real_world_deployable="partial" if has_perception else "missing_perception_data",
            notes=[
                "Perception-based object pose; not oracle GT.",
                "EEF from robot FK (proprioception); not privileged.",
                f"Perception data loaded: {len(self._perception_data)} frames." if has_perception else "NO perception data loaded. Object positions unavailable.",
            ],
        )
