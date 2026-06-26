from __future__ import annotations

import math


def is_on(obj_pose, target_pose, xy_threshold, z_min, z_max) -> bool:
    dx = obj_pose["x"] - target_pose["x"]
    dy = obj_pose["y"] - target_pose["y"]
    xy_distance = math.hypot(dx, dy)
    dz = obj_pose["z"] - target_pose["z"]
    return xy_distance <= xy_threshold and z_min <= dz <= z_max


def extract_on_predicates(object_poses, targets, thresholds) -> set[tuple[str, ...]]:
    on_predicates: set[tuple[str, ...]] = set()
    xy_threshold = thresholds["xy_threshold"]
    z_min = thresholds["z_min"]
    z_max = thresholds["z_max"]

    for obj_name, obj_pose in object_poses.items():
        for target_name, target_pose in targets.items():
            if obj_name == target_name:
                continue
            if is_on(obj_pose, target_pose, xy_threshold, z_min, z_max):
                on_predicates.add(("on", obj_name, target_name))
                break
    return on_predicates


def extract_clear_predicates(objects, on_predicates) -> set[tuple[str, ...]]:
    occupied = {pred[2] for pred in on_predicates if len(pred) == 3}
    return {("clear", obj) for obj in objects if obj not in occupied}


def physical_state_to_predicates(object_poses, targets, thresholds) -> set[tuple[str, ...]]:
    on_predicates = extract_on_predicates(object_poses, targets, thresholds)
    clear_predicates = extract_clear_predicates(object_poses.keys(), on_predicates)
    return on_predicates | clear_predicates
