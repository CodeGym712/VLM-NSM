"""Table-plane 3D pose estimation from RGB bbox centres.

For each detected object with known pixel coordinates (bbox centre), this
module projects a ray through the camera model to estimate the 3D position
on the table plane (z = table_z).

The resulting estimates are stored on each ``DetectedObject``:

- ``pose`` — dict with ``x``, ``y``, ``z`` (world coords, Z = table_z + half-height)
- ``attributes["position_source"]`` — ``"rgb_bbox_table_plane_projection"``
- ``attributes["pose_confidence"]`` — detector confidence
- ``attributes["pose_uncertainty"]`` — approximate [dx, dy, dz]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vlm_nsm.perception.camera_geometry import pixel_to_table_point
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.robosuite_camera_utils import CameraInfo


def get_projection_pixel_from_bbox(
    bbox: dict[str, float] | tuple[float, float, float, float] | list[float],
    mode: str = "center",
) -> tuple[float, float]:
    """Determine which pixel to project based on bbox geometry.

    Args:
        bbox: Bounding box as dict with keys ``x1``, ``y1``, ``x2``, ``y2``,
            or a sequence ``(x1, y1, x2, y2)``.
        mode: One of ``"center"``, ``"bottom_center"``, ``"weighted_bottom"``.

    Returns:
        ``(u, v)`` pixel coordinates for projection.

    Raises:
        ValueError: If *mode* is unknown.
    """
    # Normalise to (x1, y1, x2, y2)
    if isinstance(bbox, dict):
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
    else:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])

    u = (x1 + x2) / 2.0  # horizontal centre — always the same

    if mode == "center":
        v = (y1 + y2) / 2.0
    elif mode == "bottom_center":
        v = y2
    elif mode == "weighted_bottom":
        v = 0.25 * y1 + 0.75 * y2
    else:
        raise ValueError(
            f"Unknown projection_point mode '{mode}'. "
            f"Expected one of: center, bottom_center, weighted_bottom."
        )

    return (u, v)


def add_table_plane_pose_estimates(
    scene_info: SceneInfo,
    camera_info: CameraInfo,
    table_z: float = 0.0,
    object_height: float = 0.05,
    default_uncertainty: tuple[float, float, float] = (0.02, 0.02, 0.02),
    projection_point: str = "center",
) -> SceneInfo:
    """Estimate 3D positions for detected objects via table-plane projection.

    For each object in *scene_info*, projects a ray onto the plane
    ``z = table_z``.  The pixel to project is determined by
    *projection_point* mode, falling back to stored ``center_px`` or
    ``center_norm`` when no bbox is available.

    The resulting Z is raised by ``object_height / 2`` so the pose sits on
    top of the table rather than embedded in it.

    Objects without usable centre coordinates retain ``pose=None`` and get
    ``position_source = "projection_failed"``.

    Args:
        scene_info: Detected scene (from RGB detector).
        camera_info: Camera calibration (intrinsics + extrinsics).
        table_z: Z coordinate of the table surface in world frame.
        object_height: Approximate object height in metres (used to set
            the pose Z to ``table_z + object_height / 2``).
        default_uncertainty: (dx, dy, dz) uncertainty in metres when the
            detector does not provide its own estimate.
        projection_point: Which pixel to project. One of ``"center"``,
            ``"bottom_center"``, ``"weighted_bottom"``.

    Returns:
        The same *scene_info* with updated object poses.
    """
    for obj in scene_info.objects:
        attributes = obj.attributes

        # Determine pixel (u, v) — prefer bbox-based then fallback
        u: float | None = None
        v: float | None = None
        bbox = attributes.get("bbox")

        if bbox is not None:
            u, v = get_projection_pixel_from_bbox(bbox, mode=projection_point)
        else:
            center_px = attributes.get("center_px")
            center_norm = attributes.get("center_norm")
            if center_px is not None and "x" in center_px and "y" in center_px:
                u = float(center_px["x"])
                v = float(center_px["y"])
            elif center_norm is not None and "x" in center_norm and "y" in center_norm:
                u = float(center_norm["x"] * camera_info.width)
                v = float(center_norm["y"] * camera_info.height)

        if u is None or v is None:
            obj.pose = None
            attributes["position_source"] = "projection_failed"
            continue

        # Project to table plane
        point = pixel_to_table_point(
            u, v,
            camera_info.K,
            camera_info.T_camera_world,
            table_z,
        )

        if point is None:
            obj.pose = None
            attributes["position_source"] = "projection_failed"
            continue

        # Place object on table surface (half height up)
        z_pose = table_z + object_height / 2.0
        obj.pose = {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": z_pose,
        }
        attributes["position_source"] = "rgb_bbox_table_plane_projection"
        attributes["pose_confidence"] = obj.confidence
        attributes["pose_uncertainty"] = {
            "x": default_uncertainty[0],
            "y": default_uncertainty[1],
            "z": default_uncertainty[2],
        }
        attributes["projection_point_mode"] = projection_point
        attributes["projection_pixel"] = [float(u), float(v)]

    return scene_info
