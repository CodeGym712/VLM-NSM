"""OracleStateSource — wraps simulator GT state behind explicit privileged interface."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vlm_nsm.observation.observation_source import ObservationSource
from vlm_nsm.observation.scene_state import (
    ObjectState, RobotState, SceneState, ObservationLegalityReport,
)


class OracleStateSource(ObservationSource):
    """Provides ground-truth object/robot state from the simulator.

    WARNING: This source is PRIVILEGED. It reads MuJoCo GT poses.
    real_world_deployable = false. Use only for oracle baselines and diagnostics.
    """

    def __init__(self, cubes: list[str] | None = None):
        self._cubes = cubes or ["cube1", "cube2"]
        self._step_count = 0

    def reset(self, env: Any) -> None:
        self._step_count = 0

    def get_scene_state(self, env: Any, camera_obs: Any = None) -> SceneState:
        self._step_count += 1
        objects: dict[str, ObjectState] = {}

        for name in self._cubes:
            try:
                pose = env.get_object_pose(name)
                pos = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
                objects[name] = ObjectState(
                    name=name, class_name=name, position=pos,
                    source="oracle_simulator", available=True,
                    metadata={"oracle": True},
                )
            except Exception:
                objects[name] = ObjectState(
                    name=name, available=False, source="oracle_simulator",
                    metadata={"oracle": True, "error": "pose_unavailable"},
                )

        robot = RobotState(source="oracle_simulator", available=True)
        try:
            eef = env.get_eef_pose()
            robot.eef_position = np.array([eef["x"], eef["y"], eef["z"]], dtype=np.float32)
        except Exception:
            robot.available = False
            robot.metadata["error"] = "eef_unavailable"

        try:
            gs = env.get_gripper_state()
            robot.gripper_width = float(gs.get("open", 0.05))
        except Exception:
            pass

        return SceneState(
            objects=objects, robot=robot, source="oracle_simulator",
            timestamp=time.time(),
            metadata={"step": self._step_count, "privileged": True},
        )

    def get_legality_report(self) -> ObservationLegalityReport:
        return ObservationLegalityReport(
            observation_source_name="OracleStateSource",
            uses_privileged_object_pose=True,
            uses_privileged_eef_pose=True,
            uses_privileged_contact=False,
            uses_perception_object_pose=False,
            uses_robot_proprioception=True,
            real_world_deployable=False,
            notes=[
                "Oracle simulator state; privileged baseline only.",
                "Object poses from MuJoCo sim.data.body_xpos.",
                "EEF pose from MuJoCo sim.data.site_xpos.",
            ],
        )
