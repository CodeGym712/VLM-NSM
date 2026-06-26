"""Image-space and world-space spatial relations from detection metadata.

Image-space relations
---------------------
For RGB-detected objects where 3D positions are unknown, provides 2D
spatial relations based on normalized bbox centres.  Relations are in
image space (image-left/image-right), **not** physical 3D space.

World-space relations
---------------------
For objects with estimated 3D positions (e.g. via table-plane projection),
provides physical left/right/front/behind/near/far relations in the
robosuite world frame (Z-up, right-hand).

Coordinate convention (world frame)
------------------------------------
- **X**: lateral; negative X ≈ left, positive X ≈ right (from robot
  perspective facing the table).
- **Y**: depth; negative Y ≈ toward robot (front/closer), positive Y ≈ away
  from robot (behind/further).
- **Z**: up (vertical, not used for horizontal relations).
- Relations are computed from XY differences; Z is ignored for
  horizontal spatial predicates.
"""

from __future__ import annotations

import numpy as np

from vlm_nsm.perception.data_types import SceneInfo


def compute_image_spatial_relations(
    scene_info: SceneInfo,
    image_width: int,
    image_height: int,
    threshold: float = 0.05,
) -> SceneInfo:
    """Enrich a SceneInfo with image-space spatial relations between objects.

    For each detected object that has ``center_norm`` in its attributes,
    compares x-coordinates pairwise. Objects with ``center_norm.x`` differing
    by at least *threshold* are assigned ``image_left_of`` / ``image_right_of``.
    Pairs with smaller differences are marked as ambiguous.

    Args:
        scene_info: SceneInfo with DetectedObjects (must have center_norm).
        image_width: Image width in pixels (unused, kept for API consistency).
        image_height: Image height in pixels (unused, kept for API consistency).
        threshold: Minimum normalized dx to assign a directional relation.
            Default 0.05 (~25 px on 512-wide image).

    Returns:
        The same SceneInfo with each object's attributes enriched with
        a ``spatial`` dict containing:
        - ``center_norm``: [x, y] normalized coordinates.
        - ``image_position``: "left", "right", or "center" (relative to
          image midline at x=0.5).
        - ``relations``: dict mapping other object label to list of relation
          strings, e.g. {"cube2": ["image_left_of"]}.
        - ``ambiguous_relations``: list of object labels where dx < threshold.
    """
    objects_with_spatial: list[tuple[int, str, dict]] = []
    for idx, obj in enumerate(scene_info.objects):
        center_norm = obj.attributes.get("center_norm")
        if center_norm is not None and "x" in center_norm:
            objects_with_spatial.append((idx, obj.label, center_norm))

    for idx, label, center in objects_with_spatial:
        relations: dict[str, list[str]] = {}
        ambiguous: list[str] = []

        for other_idx, other_label, other_center in objects_with_spatial:
            if other_label == label:
                continue

            dx = center["x"] - other_center["x"]
            if abs(dx) >= threshold:
                if dx < 0:  # this object is image-left of other
                    relations[other_label] = ["image_left_of"]
                else:  # this object is image-right of other
                    relations[other_label] = ["image_right_of"]
            else:
                ambiguous.append(other_label)

        mid_x = 0.5
        if center["x"] < mid_x - threshold:
            image_position = "left"
        elif center["x"] > mid_x + threshold:
            image_position = "right"
        else:
            image_position = "center"

        scene_info.objects[idx].attributes["spatial"] = {
            "center_norm": [center["x"], center["y"]],
            "image_position": image_position,
            "relations": relations,
            "ambiguous_relations": ambiguous,
        }

    return scene_info


def compute_world_spatial_relations(
    scene_info: SceneInfo,
    xy_threshold: float = 0.02,
    near_threshold: float = 0.15,
    far_threshold: float = 0.30,
) -> SceneInfo:
    """Enrich a SceneInfo with world-frame spatial relations from estimated 3D poses.

    Uses the coordinate convention documented at the top of this module.
    Only objects with a valid **pose** dict (containing at least ``x``,
    ``y``) are considered.

    Relations computed per object pair:

    - ``left_of`` / ``right_of``  — from world X difference.
    - ``front_of`` / ``behind``   — from world Y difference
      (front = positive Y, away from robot).
    - ``near`` / ``far``          — from Euclidean XY distance.
    - Ambiguous markers for pairs where abs(delta) < *xy_threshold*.

    Results are stored in ``obj.attributes["spatial"]["world_relations"]``
    as a dict mapping other-object label to a list of relation strings.

    Image-space relations are preserved as a fallback; this method only
    adds the ``world_relations`` key.

    Args:
        scene_info: SceneInfo with objects that have estimated poses.
        xy_threshold: Minimum absolute XY delta (metres) for a directional
            relation (left/right/front/behind). Default 0.02 m.
        near_threshold: XY distance below which objects are ``near`` each
            other. Default 0.15 m.
        far_threshold: XY distance above which objects are ``far`` from
            each other. Default 0.30 m.

    Returns:
        The same SceneInfo with world_relations added to each object's
        spatial metadata.
    """
    # Collect objects with usable poses
    posed: list[tuple[int, str, float, float]] = []  # (idx, label, x, y)
    for idx, obj in enumerate(scene_info.objects):
        pose = obj.pose
        if pose is not None and "x" in pose and "y" in pose:
            posed.append((idx, obj.label, pose["x"], pose["y"]))

    for idx, label, x, y in posed:
        world_relations: dict[str, list[str]] = {}
        ambiguous_x: list[str] = []
        ambiguous_y: list[str] = []

        for other_idx, other_label, ox, oy in posed:
            if other_label == label:
                continue

            dx = x - ox
            dy = y - oy
            dist = np.hypot(dx, dy)

            # Directional relations on X (left/right)
            if abs(dx) >= xy_threshold:
                rel = "left_of" if dx < 0 else "right_of"
                world_relations.setdefault(other_label, []).append(rel)
            else:
                ambiguous_x.append(other_label)

            # Directional relations on Y (front/behind)
            # From robot's perspective: lower Y = closer = "front_of"
            if abs(dy) >= xy_threshold:
                rel = "front_of" if dy < 0 else "behind"
                world_relations.setdefault(other_label, []).append(rel)
            else:
                ambiguous_y.append(other_label)

            # Distance relations (near/far)
            if dist < near_threshold:
                world_relations.setdefault(other_label, []).append("near")
            elif dist > far_threshold:
                world_relations.setdefault(other_label, []).append("far")

        # Ensure spatial key exists
        if "spatial" not in scene_info.objects[idx].attributes:
            scene_info.objects[idx].attributes["spatial"] = {}

        scene_info.objects[idx].attributes["spatial"]["world_relations"] = world_relations
        scene_info.objects[idx].attributes["spatial"]["ambiguous_x"] = ambiguous_x
        scene_info.objects[idx].attributes["spatial"]["ambiguous_y"] = ambiguous_y

    return scene_info
