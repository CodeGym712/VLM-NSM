import json

from vlm_nsm.experiments.stage20_g_mlp_dataset_comparison import build_report, main


def _metrics(val_loss: float, train_loss: float | None = None):
    return {
        "train_loss": train_loss if train_loss is not None else val_loss * 0.8,
        "val_loss": val_loss,
        "best_val_loss": val_loss * 0.9,
        "best_epoch": 3,
        "per_direction_val_loss": {"0": val_loss, "1": val_loss + 0.001},
        "per_phase_val_loss": {"0": val_loss * 0.5, "1": val_loss * 1.5},
        "action_dim_mse": [val_loss] * 7,
    }


def _summary(windows: int, *, phase45: bool = False):
    phase_counts = {"0": 10, "1": 20, "2": 10, "3": 40}
    if phase45:
        phase_counts["4"] = 5
    return {
        "num_trajectories_loaded": 10,
        "num_windows": windows,
        "train_windows": int(windows * 0.8),
        "val_windows": windows - int(windows * 0.8),
        "can_train_mlp_baseline": True,
        "teacher_profile_counts": {"cube2_robust": windows},
        "direction_counts": {"cube1_to_cube2": windows // 2, "cube2_to_cube1": windows // 2},
        "phase_counts": phase_counts,
        "missing_field_counts": {},
    }


def _collection():
    return {
        "total_episodes": 200,
        "task_success_count": 168,
        "task_success_rate": 0.84,
        "unsafe_goal_leak_count": 0,
        "controller_runtime_errors": 0,
    }


def test_report_generation_includes_v01_v02_and_improvement():
    text = build_report(
        v01_obs=_metrics(0.006),
        v01_goal=_metrics(0.004),
        v02_obs=_metrics(0.005),
        v02_goal=_metrics(0.003),
        v01_summary=_summary(100),
        v02_summary=_summary(200),
        collection_summary=_collection(),
    )

    assert "v0_1" in text
    assert "v0_2" in text
    assert "Relative Improvement" in text
    assert "v0_2_goal_conditioned_improved = true" in text


def test_report_includes_phase45_caveat_and_recommendation():
    text = build_report(
        v01_obs=_metrics(0.006),
        v01_goal=_metrics(0.004),
        v02_obs=_metrics(0.005),
        v02_goal=_metrics(0.003),
        v01_summary=_summary(100),
        v02_summary=_summary(200, phase45=False),
        collection_summary=_collection(),
    )

    assert "v0_2_phase_4_or_5_present = false" in text
    assert "Recommendation" in text
    assert "phase labeling" in text


def test_report_cli_writes_markdown(tmp_path, monkeypatch):
    paths = {}
    for name, data in {
        "v01_obs": _metrics(0.006),
        "v01_goal": _metrics(0.004),
        "v02_obs": _metrics(0.005),
        "v02_goal": _metrics(0.003),
        "v01_summary": _summary(100),
        "v02_summary": _summary(200),
        "collection": _collection(),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[name] = path
    out = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--v01_obs_metrics", str(paths["v01_obs"]),
            "--v01_goal_metrics", str(paths["v01_goal"]),
            "--v02_obs_metrics", str(paths["v02_obs"]),
            "--v02_goal_metrics", str(paths["v02_goal"]),
            "--v01_summary", str(paths["v01_summary"]),
            "--v02_summary", str(paths["v02_summary"]),
            "--collection_summary", str(paths["collection"]),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Stage 20-G" in out.read_text(encoding="utf-8")
