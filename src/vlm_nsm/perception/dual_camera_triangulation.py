"""Dual-camera geometric triangulation for 3D pose estimation.

For each object detected in both camera views, projects two world-frame
rays (one per camera) and finds their closest-approach midpoint via
:func:`triangulate_rays`.  Objects only visible in one camera are kept
but left with ``pose=None`` and ``position_source="triangulation_failed"``.
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.perception.camera_geometry import (
    pixel_to_camera_ray,
    transform_ray_to_world,
    triangulate_rays,
)
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.rgb_table_projection import get_projection_pixel_from_bbox
from vlm_nsm.perception.robosuite_camera_utils import CameraInfo


def _extract_pixel(
    obj: DetectedObject,
    camera_info: CameraInfo,
    pixel_mode: str = "center",
) -> tuple[float, float] | None:
    """Determine the (u, v) pixel for an object using the standard fallback chain.

    Prefers bbox (via ``get_projection_pixel_from_bbox`` with the given
    *pixel_mode*), then ``center_px``, then ``center_norm``.
    """
    bbox = obj.attributes.get("bbox")
    if bbox is not None:
        return get_projection_pixel_from_bbox(bbox, mode=pixel_mode)

    center_px = obj.attributes.get("center_px")
    if center_px is not None and "x" in center_px and "y" in center_px:
        return (float(center_px["x"]), float(center_px["y"]))

    center_norm = obj.attributes.get("center_norm")
    if center_norm is not None and "x" in center_norm and "y" in center_norm:
        u = float(center_norm["x"] * camera_info.width)
        v = float(center_norm["y"] * camera_info.height)
        return (u, v)

    return None


def _match_objects(
    scene1: SceneInfo,
    scene2: SceneInfo,
) -> list[tuple[int, int]]:
    """Match objects across two SceneInfos by label, falling back to color.

    Returns list of ``(idx_scene1, idx_scene2)`` pairs.
    """
    label_to_idx1: dict[str, int] = {}
    for i, obj in enumerate(scene1.objects):
        label_to_idx1[obj.label] = i

    label_to_idx2: dict[str, int] = {}
    for i, obj in enumerate(scene2.objects):
        label_to_idx2[obj.label] = i

    pairs: list[tuple[int, int]] = []
    matched_i2: set[int] = set()
    matched_i1: set[int] = set()

    for label, i1 in label_to_idx1.items():
        if label in label_to_idx2:
            i2 = label_to_idx2[label]
            pairs.append((i1, i2))
            matched_i1.add(i1)
            matched_i2.add(i2)

    # Fallback: match unmatched objects by color
    unmatched_i1 = [i for i in range(len(scene1.objects)) if i not in matched_i1]
    unmatched_i2 = [i for i in range(len(scene2.objects)) if i not in matched_i2]

    for i1 in unmatched_i1:
        c1 = scene1.objects[i1].color
        if c1 is None:
            continue
        for i2 in unmatched_i2:
            c2 = scene2.objects[i2].color
            if c2 is not None and c1 == c2:
                pairs.append((i1, i2))
                unmatched_i2.remove(i2)
                break

    return pairs


def triangulate_scene(
    scene1: SceneInfo,
    scene2: SceneInfo,
    camera1_info: CameraInfo,
    camera2_info: CameraInfo,
    table_z: float = 0.0,
    object_height: float = 0.05,
    pixel_mode: str = "center",
) -> SceneInfo:
    """Merge two single-camera SceneInfos via geometric ray triangulation.

    For each object detected in *both* scenes, the two world-frame rays
    are triangulated with :func:`triangulate_rays`.  The resulting 3D
    midpoint is stored as ``obj.pose``.  Objects detected in only one
    camera are kept with ``pose=None`` and an appropriate failure reason.

    Args:
        scene1: Primary scene (e.g. agentview).
        scene2: Secondary scene (e.g. frontview).
        camera1_info: Calibration for the primary camera.
        camera2_info: Calibration for the secondary camera.
        table_z: Table-plane Z coordinate (used only for pose Z offset).
        object_height: Object height; pose Z is set to
            ``table_z + object_height / 2``.
        pixel_mode: Bbox projection mode for pixel extraction
            (``"center"`` or ``"bottom_center"``).  Default ``"center"``
            per Stage 18-F2 calibration audit results.

    Returns:
        A merged ``SceneInfo``.  Objects from *scene1* form the base;
        their ``pose`` and ``attributes`` are updated in-place.
    """
    # Build pixel lookup per scene
    pixels1: dict[int, tuple[float, float]] = {}
    for i, obj in enumerate(scene1.objects):
        px = _extract_pixel(obj, camera1_info, pixel_mode=pixel_mode)
        if px is not None:
            pixels1[i] = px

    pixels2: dict[int, tuple[float, float]] = {}
    for i, obj in enumerate(scene2.objects):
        px = _extract_pixel(obj, camera2_info, pixel_mode=pixel_mode)
        if px is not None:
            pixels2[i] = px

    # Match objects across scenes
    pairs = _match_objects(scene1, scene2)

    z_pose = table_z + object_height / 2.0
    seen_i1: set[int] = set()

    for i1, i2 in pairs:
        seen_i1.add(i1)
        obj = scene1.objects[i1]

        if i1 not in pixels1 or i2 not in pixels2:
            obj.pose = None
            obj.attributes["position_source"] = "triangulation_failed"
            obj.attributes["failure_reason"] = "no_pixel"
            continue

        u1, v1 = pixels1[i1]
        u2, v2 = pixels2[i2]

        # Ray 1 (camera 1)
        _, dir_cam1 = pixel_to_camera_ray(u1, v1, camera1_info.K)
        origin1, dir1 = transform_ray_to_world(
            np.zeros(3, dtype=np.float64), dir_cam1, camera1_info.T_camera_world,
        )

        # Ray 2 (camera 2)
        _, dir_cam2 = pixel_to_camera_ray(u2, v2, camera2_info.K)
        origin2, dir2 = transform_ray_to_world(
            np.zeros(3, dtype=np.float64), dir_cam2, camera2_info.T_camera_world,
        )

        point = triangulate_rays(origin1, dir1, origin2, dir2)

        if point is None:
            obj.pose = None
            obj.attributes["position_source"] = "triangulation_failed"
            obj.attributes["failure_reason"] = "degenerate_rays"
        else:
            # Compute residual for diagnostics
            b = np.dot(dir1, dir2)
            w = origin1 - origin2
            d = np.dot(dir1, w)
            e_val = np.dot(dir2, w)
            denom = 1.0 - b * b
            t1 = (b * e_val - d) / denom
            t2 = (e_val - b * d) / denom
            p1 = origin1 + t1 * dir1
            p2 = origin2 + t2 * dir2
            residual = float(np.linalg.norm(p1 - p2))

            obj.pose = {"x": float(point[0]), "y": float(point[1]), "z": z_pose}
            obj.confidence = min(
                obj.confidence,
                scene2.objects[i2].confidence,
            )
            obj.attributes["position_source"] = "rgb_dual_camera_triangulation"
            obj.attributes["pose_confidence"] = obj.confidence
            obj.attributes["pose_uncertainty"] = {"x": 0.01, "y": 0.01, "z": 0.02}
            obj.attributes["triangulation_residual"] = residual
            obj.attributes["triangulation_source"] = (
                f"{camera1_info.camera_name}+{camera2_info.camera_name}"
            )
            obj.attributes["camera1_pixel"] = [float(u1), float(v1)]
            obj.attributes["camera2_pixel"] = [float(u2), float(v2)]

    # Objects in scene1 not matched → mark as failed
    for i, obj in enumerate(scene1.objects):
        if i not in seen_i1:
            if obj.pose is None and obj.attributes.get("position_source") != "triangulation_failed":
                obj.attributes["position_source"] = "triangulation_failed"
                obj.attributes["failure_reason"] = "no_match"

    return scene1
