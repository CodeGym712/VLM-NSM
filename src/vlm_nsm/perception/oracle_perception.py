"""Oracle perception — returns ground truth scene info from sim state.

This is an oracle that reads the simulation state directly rather than
using vision. Suitable for testing and offline development before
vision-based perception (Stage 15) is integrated.
"""

from __future__ import annotations

from vlm_nsm.perception.data_types import DEFAULT_COLOR_MAP, DetectedObject, SceneInfo


def perceive_scene(env) -> SceneInfo:
    """Extract ground-truth scene info from a RobosuiteEnvWrapper instance.

    Args:
        env: A ``RobosuiteEnvWrapper`` or ``MockEnvWrapper`` instance.

    Returns:
        SceneInfo populated with current sim state.
    """
    objects: list[DetectedObject] = []

    # Cube objects
    for cn in env.cube_names:
        try:
            pose = env.get_object_pose(cn)
        except (ValueError, KeyError):
            pose = None
        objects.append(DetectedObject(label=cn, category="block", pose=pose, color=DEFAULT_COLOR_MAP.get(cn)))

    # Table
    try:
        objects.append(
            DetectedObject(
                label="table", category="table", pose=env.get_object_pose("table")
            )
        )
    except (ValueError, KeyError):
        objects.append(DetectedObject(label="table", category="table"))

    # On relations via physical predicate extractor
    on_relations: list[tuple[str, str]] = []
    try:
        from vlm_nsm.core.physical_predicate_extractor import extract_on_predicates

        object_poses = env.get_all_object_poses()
        thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
        on_preds = extract_on_predicates(object_poses, object_poses, thresholds)
        on_relations = [
            (pred[1], pred[2]) for pred in on_preds if len(pred) == 3
        ]
    except Exception:
        pass

    # Gripper state
    gripper_state = {}
    try:
        gripper_state = env.get_gripper_state()
    except Exception:
        pass

    has_object = gripper_state.get("has_object", False)

    return SceneInfo(
        objects=objects,
        on_relations=on_relations,
        free_gripper=not has_object,
    )
