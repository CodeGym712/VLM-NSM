import json

from vlm_nsm.experiments.stage19_d5_profile_report import (
    build_best_eval_report,
    build_profile_sweep_report,
)


def test_profile_report_selects_best(tmp_path):
    (tmp_path / "stage19_d5_profile_default_summary.json").write_text(json.dumps({
        "task_success_count": 10,
        "executed_episodes": 20,
        "task_success_rate": 0.5,
        "per_failure_reason_counts": {"grasp_failure": 5},
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
    }))
    (tmp_path / "stage19_d5_profile_robust_summary.json").write_text(json.dumps({
        "task_success_count": 16,
        "executed_episodes": 20,
        "task_success_rate": 0.8,
        "per_failure_reason_counts": {"place_failure": 2},
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
    }))

    out = tmp_path / "sweep.md"
    report = build_profile_sweep_report(tmp_path, out)

    assert report["best_profile"] == "cube2_robust"
    assert "cube2_robust" in out.read_text(encoding="utf-8")


def test_best_eval_report_verdict_pass(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "task_success_rate": 0.825,
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
        "per_goal_success": {
            "red_on_green": {"success": 17, "total": 20},
            "green_on_red": {"success": 16, "total": 20},
        },
        "per_failure_reason_counts": {"none": 33, "place_failure": 7},
    }))
    out = tmp_path / "report.md"

    result = build_best_eval_report(summary, out, profile="cube2_robust")

    assert result["verdict"] == "PASS"
    text = out.read_text(encoding="utf-8")
    assert "D5 `cube2_robust`" in text
    assert "82.5%" in text
