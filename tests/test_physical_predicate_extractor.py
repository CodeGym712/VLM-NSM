from vlm_nsm.core.physical_predicate_extractor import (
    extract_clear_predicates,
    extract_on_predicates,
    physical_state_to_predicates,
)


def test_extract_on_and_clear_predicates():
    object_poses = {
        "cube1": {"x": 0.0, "y": 0.0, "z": 0.10},
        "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        "cube3": {"x": 0.1, "y": 0.0, "z": 0.02},
    }
    targets = {
        "table": {"x": 0.0, "y": 0.0, "z": 0.0},
        "cube1": object_poses["cube1"],
        "cube2": object_poses["cube2"],
        "cube3": object_poses["cube3"],
    }
    thresholds = {"xy_threshold": 0.03, "z_min": 0.03, "z_max": 0.05}
    on_predicates = extract_on_predicates(object_poses, targets, thresholds)
    assert ("on", "cube1", "cube2") in on_predicates
    clear_predicates = extract_clear_predicates(object_poses.keys(), on_predicates)
    assert ("clear", "cube1") in clear_predicates
    assert ("clear", "cube3") in clear_predicates


def test_physical_state_to_predicates_supports_table_targets():
    object_poses = {
        "cube1": {"x": 0.0, "y": 0.0, "z": 0.06},
        "cube2": {"x": 0.0, "y": 0.0, "z": 0.02},
    }
    targets = {"table": {"x": 0.0, "y": 0.0, "z": 0.0}} | object_poses
    thresholds = {"xy_threshold": 0.03, "z_min": 0.015, "z_max": 0.05}
    predicates = physical_state_to_predicates(object_poses, targets, thresholds)
    assert ("on", "cube2", "table") in predicates
    assert ("on", "cube1", "cube2") in predicates
