import json

from vlm_nsm.experiments.stage20_f_dataset_report import build_report, main


def _summary(windows, phases=None):
    return {
        "num_trajectories_loaded": 10,
        "num_trajectories_used": 8,
        "num_windows": windows,
        "train_windows": int(windows * 0.8),
        "val_windows": windows - int(windows * 0.8),
        "can_train_mlp_baseline": True,
        "direction_counts": {"cube1_to_cube2": windows // 2, "cube2_to_cube1": windows // 2},
        "phase_counts": phases or {"0": 5, "1": 5},
        "placement_bin_counts": {"center": windows},
        "teacher_profile_counts": {"cube2_robust": windows},
        "missing_field_counts": {},
    }


def test_build_report_compares_datasets():
    collection = {
        "total_episodes": 2,
        "executed_episodes": 2,
        "task_success_count": 2,
        "task_success_rate": 1.0,
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
        "scripted_profile": "cube2_robust",
        "per_goal_success": {"red_on_green": {"total": 1, "success": 1}},
        "per_failure_reason_counts": {"none": 2},
    }

    text = build_report(_summary(10), _summary(20, {"0": 5, "4": 5}), collection)

    assert "Stage 20-F Dataset Report" in text
    assert "phase_4_or_5_present = true" in text
    assert "cube2_robust" in text


def test_report_cli_writes_markdown(tmp_path, monkeypatch):
    v01 = tmp_path / "v01.json"
    v02 = tmp_path / "v02.json"
    coll = tmp_path / "collection.json"
    out = tmp_path / "report.md"
    v01.write_text(json.dumps(_summary(10)))
    v02.write_text(json.dumps(_summary(20)))
    coll.write_text(json.dumps({
        "total_episodes": 1,
        "executed_episodes": 1,
        "task_success_count": 1,
        "task_success_rate": 1.0,
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
    }))
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--v01_summary", str(v01),
            "--v02_summary", str(v02),
            "--collection_summary", str(coll),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "v0_1 vs v0_2" in out.read_text()
