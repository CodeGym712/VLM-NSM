"""Predicate-level scene enrichment from estimated 3D poses.

Computes per-object predicates (on_table, robot-relative distance/direction,
edge proximity, pose sanity) and annotates them on the SceneInfo.

Follows the same enrichment pattern as ``compute_world_spatial_relations()``:
take SceneInfo, mutate obj.attributes in-place, return SceneInfo.
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.perception.data_types import SceneInfo
from vlm_nsm.perception.scene_context import SceneReferenceFrame


def compute_on_table_status(
    scene_info: SceneInfo,
    ref_frame: SceneReferenceFrame,
    object_height: float = 0.05,
    z_tolerance: float = 0.02,
) -> SceneInfo:
    """Mark each object as on_table (True/False/None) based on Z position.

    An object is "on table" when its centre Z is close to
    ``table_z_surface + object_height / 2`` — i.e. resting directly on the
    table surface, not stacked on another object.

    Args:
        scene_info: SceneInfo with posed objects.
        ref_frame: Workspace reference frame.
        object_height: Approximate object height in metres (default 0.05).
        z_tolerance: Max Z deviation for on_table status (default 0.02 m).

    Returns:
        The same SceneInfo with ``obj.attributes["on_table"]`` set.
    """
    expected_z = ref_frame.table_z_surface + object_height / 2.0

    for obj in scene_info.objects:
        if obj.pose is not None and "z" in obj.pose:
            dz = abs(obj.pose["z"] - expected_z)
            obj.attributes["on_table"] = dz <= z_tolerance
        else:
            obj.attributes["on_table"] = None

    return scene_info


def compute_robot_relative(
    scene_info: SceneInfo,
    ref_frame: SceneReferenceFrame,
    xy_threshold: float = 0.02,
) -> SceneInfo:
    """Add robot-relative distance and direction for each object.

    Sets ``obj.attributes["robot_distance"]`` (float, metres) and
    ``obj.attributes["robot_direction"]`` (list of str like
    ``"robot-left"``, ``"toward-robot"``).

    Coordinate convention (matching spatial_relations.py):
    - X: negative = robot-left, positive = robot-right
    - Y: negative = toward robot (front), positive = away from robot (behind)

    Args:
        scene_info: SceneInfo with posed objects.
        ref_frame: Workspace reference frame.
        xy_threshold: Minimum delta for directional labels (default 0.02 m).

    Returns:
        The same SceneInfo with robot-relative attributes set.
    """
    rx = ref_frame.robot_base_x
    ry = ref_frame.robot_base_y

    for obj in scene_info.objects:
        if obj.pose is not None and "x" in obj.pose and "y" in obj.pose:
            dx = obj.pose["x"] - rx
            dy = obj.pose["y"] - ry
            dist = float(np.hypot(dx, dy))
            obj.attributes["robot_distance"] = dist

            direction: list[str] = []
            if abs(dx) >= xy_threshold:
                direction.append("robot-left" if dx < 0 else "robot-right")
            else:
                direction.append("robot-center-x")
            if abs(dy) >= xy_threshold:
                direction.append("toward-robot" if dy < 0 else "away-from-robot")
            else:
                direction.append("robot-center-y")
            obj.attributes["robot_direction"] = direction
        else:
            obj.attributes["robot_distance"] = None
            obj.attributes["robot_direction"] = []

    return scene_info


def compute_edge_relations(
    scene_info: SceneInfo,
    ref_frame: SceneReferenceFrame,
) -> SceneInfo:
    """Mark objects near table edges.

    Sets ``obj.attributes["near_edges"]`` (list of str):
    ``"left_edge"``, ``"right_edge"``, ``"front_edge"``, ``"back_edge"``.

    "front" = negative Y (closer to robot), matching coordinate convention.

    Args:
        scene_info: SceneInfo with posed objects.
        ref_frame: Workspace reference frame with edge_threshold.

    Returns:
        The same SceneInfo with edge proximity attributes set.
    """
    t = ref_frame.edge_threshold
    x_min = ref_frame.table_x_min
    x_max = ref_frame.table_x_max
    y_min = ref_frame.table_y_min
    y_max = ref_frame.table_y_max

    for obj in scene_info.objects:
        if obj.pose is not None and "x" in obj.pose and "y" in obj.pose:
            x, y = obj.pose["x"], obj.pose["y"]
            edges: list[str] = []
            if abs(x - x_min) < t:
                edges.append("left_edge")
            elif abs(x - x_max) < t:
                edges.append("right_edge")
            if abs(y - y_min) < t:
                edges.append("front_edge")
            elif abs(y - y_max) < t:
                edges.append("back_edge")
            obj.attributes["near_edges"] = edges
        else:
            obj.attributes["near_edges"] = []

    return scene_info


def filter_pose_outliers(
    scene_info: SceneInfo,
    ref_frame: SceneReferenceFrame,
) -> SceneInfo:
    """Flag objects whose estimated pose falls outside the workspace.

    Sets ``obj.attributes["pose_in_workspace"]`` (bool) and, when False,
    ``obj.attributes["pose_suspicious"] = True`` and
    ``obj.attributes["suspicious_reason"]``.

    Args:
        scene_info: SceneInfo with posed objects.
        ref_frame: Workspace reference frame with workspace_padding.

    Returns:
        The same SceneInfo with pose sanity attributes set.
    """
    pad = ref_frame.workspace_padding
    x_min = ref_frame.table_x_min - pad
    x_max = ref_frame.table_x_max + pad
    y_min = ref_frame.table_y_min - pad
    y_max = ref_frame.table_y_max + pad

    for obj in scene_info.objects:
        if obj.pose is not None and "x" in obj.pose and "y" in obj.pose:
            x, y = obj.pose["x"], obj.pose["y"]
            in_bounds = x_min <= x <= x_max and y_min <= y <= y_max
            obj.attributes["pose_in_workspace"] = in_bounds
            if not in_bounds:
                obj.attributes["pose_suspicious"] = True
                obj.attributes["suspicious_reason"] = (
                    f"estimated position ({x:.2f}, {y:.2f}) outside "
                    f"workspace bounds [{x_min:.1f}, {x_max:.1f}] × "
                    f"[{y_min:.1f}, {y_max:.1f}]"
                )
            else:
                obj.attributes["pose_suspicious"] = False
        else:
            obj.attributes["pose_in_workspace"] = None
            obj.attributes["pose_suspicious"] = False

    return scene_info


def compute_scene_predicates(
    scene_info: SceneInfo,
    ref_frame: SceneReferenceFrame,
    object_height: float = 0.05,
    z_tolerance: float = 0.02,
    xy_threshold: float = 0.02,
) -> SceneInfo:
    """Compute all predicate-level enrichments for a scene.

    Convenience master function that calls the individual predicate
    functions in order. Follows the same pattern as
    ``compute_world_spatial_relations()``: takes SceneInfo, mutates
    in-place, returns it.

    Args:
        scene_info: SceneInfo with estimated 3D poses.
        ref_frame: Workspace reference frame.
        object_height: Approximate object height in metres.
        z_tolerance: Max Z deviation for on_table status.
        xy_threshold: Minimum delta for directional labels.

    Returns:
        The same SceneInfo enriched with predicate attributes.
    """
    compute_on_table_status(scene_info, ref_frame, object_height, z_tolerance)
    compute_robot_relative(scene_info, ref_frame, xy_threshold)
    compute_edge_relations(scene_info, ref_frame)
    filter_pose_outliers(scene_info, ref_frame)
    return scene_info
