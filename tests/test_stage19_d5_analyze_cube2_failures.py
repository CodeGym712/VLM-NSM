import json

from vlm_nsm.experiments.stage19_d5_analyze_cube2_failures import (
    analyze_cube2_failures,
    write_markdown,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_analyze_cube2_failures_filters_green_on_red_failures(tmp_path):
    r1c = tmp_path / "r1c.jsonl"
    d2 = tmp_path / "d2.jsonl"
    d4 = tmp_path / "d4.jsonl"

    _write_jsonl(r1c, [
        {
            "episode_id": "r1c_green_fail",
            "goal_id": "green_on_red",
            "resolved_top": "cube2",
            "task_success": False,
            "failure_reason": "unknown",
            "phase_transition_trace": {"reach_pick": 100, "grasp": 250, "lift": 0},
            "initial_top_bottom_distance_xy": 0.12,
            "final_pose_error": 0.14,
            "initial_cube2_pose": {"x": 0, "y": 0, "z": 0.835},
            "final_cube2_pose": {"x": 0.1, "y": 0, "z": 0.825},
        },
        {
            "episode_id": "r1c_red_fail",
            "goal_id": "red_on_green",
            "resolved_top": "cube1",
            "task_success": False,
            "failure_reason": "place_failure",
        },
    ])
    _write_jsonl(d2, [
        {
            "episode_id": "d2_green_success",
            "goal_id": "d2_green_on_red",
            "resolved_top": "cube2",
            "task_success": True,
            "failure_reason": "none",
        },
    ])
    _write_jsonl(d4, [
        {
            "episode_id": "d4_green_fail",
            "goal_id": "green_on_red",
            "resolved_top": "cube2",
            "task_success": False,
            "failure_reason": "place_failure",
            "phase_transition_trace": {"reach_pick": 100, "grasp": 100, "lift": 100, "reach_place": 500},
        },
    ])

    analysis = analyze_cube2_failures(
        r1c_diagnostics=r1c,
        d2_episodes=d2,
        d4_diagnostics=d4,
    )

    assert analysis["failure_count"] == 2
    assert analysis["failure_reason_distribution"]["unknown"] == 1
    assert analysis["failure_reason_distribution"]["place_failure"] == 1
    assert "adjust_grasp_height" in analysis["recommendations"]
    assert "add_stabilization_steps" in analysis["recommendations"]
    assert analysis["initial_top_bottom_distance_xy"]["count"] == 1


def test_write_markdown(tmp_path):
    analysis = {
        "scope": "test",
        "total_input_rows": 2,
        "failure_count": 1,
        "failure_reason_distribution": {"unknown": 1},
        "failure_phase_distribution": {"grasp": 1},
        "missing_diagnostics": {"planned_grasp_pose": 1},
        "recommendations": ["adjust_grasp_height"],
    }
    out = tmp_path / "report.md"

    write_markdown(analysis, out)

    text = out.read_text(encoding="utf-8")
    assert "Stage 19-D5 Cube2 Failure Analysis" in text
    assert "adjust_grasp_height" in text
