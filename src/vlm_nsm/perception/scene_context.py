"""Scene reference frame: table geometry, robot pose, workspace bounds.

Provides a ``SceneReferenceFrame`` dataclass that encodes the spatial context
of a robosuite manipulation scene. Functions to extract from a live simulator
and persist/load as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class SceneReferenceFrame:
    """Workspace geometry and robot reference pose.

    Defaults match the robosuite Stack environment with Panda robot:
    table 0.8×0.8 m centred at origin, surface at z≈0.8 m,
    robot base at (0.5, 0, 1.35).
    """

    table_x_min: float = -0.4
    table_x_max: float = 0.4
    table_y_min: float = -0.4
    table_y_max: float = 0.4
    table_z_surface: float = 0.8
    robot_base_x: float = 0.5
    robot_base_y: float = 0.0
    robot_base_z: float = 1.35
    edge_threshold: float = 0.10
    workspace_padding: float = 0.30

    @property
    def table_center_x(self) -> float:
        return (self.table_x_min + self.table_x_max) / 2.0

    @property
    def table_center_y(self) -> float:
        return (self.table_y_min + self.table_y_max) / 2.0


def extract_reference_frame_from_env(env) -> SceneReferenceFrame:
    """Extract workspace geometry from a robosuite environment.

    Args:
        env: A ``RobosuiteEnvWrapper`` instance with an active simulation.

    Returns:
        SceneReferenceFrame populated from sim state.
    """
    sim = env.env.sim
    model = sim.model
    data = sim.data

    # Find table body and its geom extents
    table_x_min, table_x_max = -0.4, 0.4
    table_y_min, table_y_max = -0.4, 0.4
    table_z = 0.8

    for i in range(model.nbody):
        name = str(model.body_names[i]).lower()
        if "table" in name:
            pos = data.body_xpos[i]
            table_z = float(pos[2])
            # Try to read geom sizes for more accurate table bounds
            geom_start = model.body_geomadr[i]
            geom_count = model.body_geomnum[i]
            if geom_count > 0 and geom_start >= 0:
                gidx = geom_start
                gsize = model.geom_size[gidx]
                gpos = data.geom_xpos[gidx]
                half_x = float(gsize[0])
                half_y = float(gsize[1])
                table_x_min = float(gpos[0]) - half_x
                table_x_max = float(gpos[0]) + half_x
                table_y_min = float(gpos[1]) - half_y
                table_y_max = float(gpos[1]) + half_y
            break

    # Find robot base position
    robot_x, robot_y, robot_z = 0.5, 0.0, 1.35
    for i in range(model.nbody):
        name = str(model.body_names[i]).lower()
        if "robot0" in name and ("base" in name or "mount" in name):
            pos = data.body_xpos[i]
            robot_x = float(pos[0])
            robot_y = float(pos[1])
            robot_z = float(pos[2])
            break

    return SceneReferenceFrame(
        table_x_min=table_x_min,
        table_x_max=table_x_max,
        table_y_min=table_y_min,
        table_y_max=table_y_max,
        table_z_surface=table_z,
        robot_base_x=robot_x,
        robot_base_y=robot_y,
        robot_base_z=robot_z,
    )


def save_reference_frame(path: str | Path, ref_frame: SceneReferenceFrame) -> None:
    """Persist a SceneReferenceFrame as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = asdict(ref_frame)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_reference_frame(path: str | Path) -> SceneReferenceFrame:
    """Load a SceneReferenceFrame from JSON."""
    data = json.loads(Path(path).read_text())
    return SceneReferenceFrame(**data)
