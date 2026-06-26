import numpy as np

from vlm_nsm.experiments.stage19_run_symbolic_rollouts import (
    add_stage20_trajectory_fields,
    build_stage20_manifest_record,
)


def test_add_stage20_trajectory_fields_adds_metadata():
    traj = {
        "obs": np.zeros((3, 11), dtype=np.float32),
        "action": np.zeros((3, 7), dtype=np.float32),
        "phase_id": np.array([0, 1, 3], dtype=np.int32),
    }
    episode = {"goal_id": "green_on_red", "direction": "cube2_to_cube1"}
    result = {
        "resolved_top": "cube2",
        "resolved_bottom": "cube1",
        "task_success": True,
        "grasp_z_offset": -0.003,
        "place_z_offset": 0.01,
        "release_z_offset": 0.005,
        "stabilization_steps": 10,
    }

    out = add_stage20_trajectory_fields(
        traj,
        episode_index=7,
        episode_entry=episode,
        result=result,
        scripted_profile="cube2_robust",
    )

    assert out["phase_name"].tolist() == ["reach_pick", "grasp", "reach_place"]
    assert set(out["teacher_profile"].tolist()) == {"cube2_robust"}
    assert set(out["top_object"].tolist()) == {"cube2"}
    assert np.all(out["scripted_stabilization_steps"] == 10)
    assert out["phase_id_raw"].tolist() == [0, 1, 3]
    assert out["phase_label_patch_applied"].tolist() == [False, False, False]
    assert out["phase_label_changed"].tolist() == [False, False, False]


def test_add_stage20_trajectory_fields_relabels_composite_place_release_retreat():
    actions = np.zeros((6, 7), dtype=np.float32)
    actions[:2, 6] = 1.0
    actions[2:4, 6] = -1.0
    actions[4:, 6] = -1.0
    actions[4:, 2] = 0.05
    traj = {
        "obs": np.zeros((6, 11), dtype=np.float32),
        "action": actions,
        "phase_id": np.array([3, 3, 3, 3, 3, 3], dtype=np.int32),
    }
    episode = {"goal_id": "green_on_red", "direction": "cube2_to_cube1"}
    result = {"resolved_top": "cube2", "resolved_bottom": "cube1", "task_success": True}

    out = add_stage20_trajectory_fields(
        traj,
        episode_index=8,
        episode_entry=episode,
        result=result,
        scripted_profile="cube2_robust",
    )

    assert out["phase_id"].tolist() == [3, 3, 4, 4, 5, 5]
    assert out["phase_name"].tolist() == ["reach_place", "reach_place", "release", "release", "retreat", "retreat"]
    assert out["phase_id_raw"].tolist() == [3, 3, 3, 3, 3, 3]
    assert out["phase_label_patch_applied"].tolist() == [True] * 6
    assert out["phase_label_changed"].tolist() == [False, False, True, True, True, True]


def test_build_stage20_manifest_record_contains_required_fields():
    episode = {"episode_id": "ep0", "goal_id": "red_on_green", "seed": 3}
    result = {
        "resolved_top": "cube1",
        "resolved_bottom": "cube2",
        "task_success": False,
        "failure_reason": "grasp_failure",
        "steps": 12,
        "skill_failure_phase": "grasp",
        "grasp_z_offset": 0.0,
        "place_z_offset": 0.0,
        "release_z_offset": 0.0,
        "stabilization_steps": 0,
    }

    record = build_stage20_manifest_record(
        episode,
        result,
        "trajectories/ep0.npz",
        "cube2_robust",
    )

    assert record["episode_id"] == "ep0"
    assert record["object_id"] == "cube1"
    assert record["target_object_id"] == "cube2"
    assert record["teacher"] == "scripted"
    assert record["teacher_profile"] == "cube2_robust"
    assert record["trajectory_path"] == "trajectories/ep0.npz"
    assert record["failure_phase"] == "grasp"
