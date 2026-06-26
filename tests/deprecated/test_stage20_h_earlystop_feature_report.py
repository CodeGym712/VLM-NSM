import json

from vlm_nsm.experiments.stage20_h_earlystop_feature_report import build_report, main


def _metrics(val, best=None):
    return {
        "train_loss": val * 0.9,
        "val_loss": val,
        "best_val_loss": best if best is not None else val * 0.8,
        "best_epoch": 2,
        "per_direction_val_loss": {"0": val, "1": val + 0.001},
        "per_phase_val_loss": {"0": val * 0.2, "1": val * 1.5, "3": val},
    }


def _summary(windows):
    return {
        "num_trajectories_loaded": 2,
        "num_windows": windows,
        "train_windows": int(windows * 0.8),
        "val_windows": windows - int(windows * 0.8),
        "teacher_profile_counts": {"cube2_robust": windows},
        "direction_counts": {"cube1_to_cube2": windows // 2, "cube2_to_cube1": windows // 2},
        "phase_counts": {"0": 10, "1": 20, "3": 30},
    }


def _norm():
    return {
        "obs_history": {"mean": [0.0, 1.0], "std": [1.0, 2.0]},
        "actions": {"mean": [0.0], "std": [1e-6]},
    }


def _payload():
    return {
        "metrics": {
            "v01_obs": _metrics(0.006),
            "v01_goal": _metrics(0.004),
            "v02_obs": _metrics(0.005, best=0.004),
            "v02_goal": _metrics(0.0051, best=0.003),
            "combined_obs": _metrics(0.004),
            "combined_goal": _metrics(0.0035),
        },
        "summaries": {
            "v01": _summary(100),
            "v02": _summary(200),
            "combined": _summary(300),
        },
        "normalization_stats": {
            "v01": _norm(),
            "v02": _norm(),
            "combined": _norm(),
        },
    }


def test_build_report_includes_early_stopping_and_feature_audit():
    text = build_report(**_payload())

    assert "Stage 20-H" in text
    assert "Early-Stopping Decision" in text
    assert "combined_best_goal_improves = true" in text
    assert "Feature Scaling Audit" in text
    assert "phase 4/5" in text or "Phase 4/5" in text


def test_cli_writes_report(tmp_path, monkeypatch):
    payload = _payload()
    paths = {}
    for group, entries in payload.items():
        for name, data in entries.items():
            path = tmp_path / f"{group}_{name}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            paths[f"{group}_{name}"] = path
    out = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--v01_obs_metrics", str(paths["metrics_v01_obs"]),
            "--v01_goal_metrics", str(paths["metrics_v01_goal"]),
            "--v02_obs_metrics", str(paths["metrics_v02_obs"]),
            "--v02_goal_metrics", str(paths["metrics_v02_goal"]),
            "--combined_obs_metrics", str(paths["metrics_combined_obs"]),
            "--combined_goal_metrics", str(paths["metrics_combined_goal"]),
            "--v01_summary", str(paths["summaries_v01"]),
            "--v02_summary", str(paths["summaries_v02"]),
            "--combined_summary", str(paths["summaries_combined"]),
            "--v01_norm", str(paths["normalization_stats_v01"]),
            "--v02_norm", str(paths["normalization_stats_v02"]),
            "--combined_norm", str(paths["normalization_stats_combined"]),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Recommendation" in out.read_text(encoding="utf-8")
