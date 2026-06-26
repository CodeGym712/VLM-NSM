import json

from vlm_nsm.experiments.stage20_p_gripper_magnitude_report import build_report, main


def _metrics(best=0.0018, phase1=0.008, direction1=0.003, action6=0.01, d1_action6=0.02):
    return {
        "train_loss": best * 1.1,
        "final_val_loss": best,
        "val_loss": best,
        "best_val_loss": best,
        "best_epoch": 3,
        "per_phase_val_loss": {"1": phase1, "3": 0.0001},
        "per_direction_val_loss": {"0": 0.0005, "1": direction1},
        "per_direction_phase_val_loss": {"1:1": phase1 * 1.5},
        "action_dim_mse": [0, 0, 0, 0, 0, 0, action6],
        "per_direction_action_dim_mse": {"0": [0, 0, 0, 0, 0, 0, action6 / 2], "1": [0, 0, 0, 0, 0, 0, d1_action6]},
        "gripper_magnitude_aux_weight": 0.5,
        "gripper_magnitude_aux_phase_ids": [1],
        "gripper_magnitude_mse": d1_action6,
        "per_phase_gripper_magnitude_mse": {"1": d1_action6},
        "per_direction_gripper_magnitude_mse": {"1": d1_action6},
        "per_direction_phase_gripper_magnitude_mse": {"1:1": d1_action6},
    }


def _summary():
    return {
        "num_windows": 10,
        "train_windows": 8,
        "val_windows": 2,
        "direction_counts": {"cube1_to_cube2": 5, "cube2_to_cube1": 5},
        "phase_counts": {"1": 4, "3": 6},
    }


def _audit():
    return {
        "diagnosis": {
            "likely_error_sources": ["magnitude_regression_or_timing"],
            "recommendation": "Prioritize gripper magnitude auxiliary head.",
        }
    }


def test_build_report_includes_magnitude_comparison_and_recommendation():
    text = build_report(
        baseline_metrics=_metrics(best=0.0017, phase1=0.009, action6=0.02, d1_action6=0.04),
        gripper_class_metrics=_metrics(best=0.0019, phase1=0.010, action6=0.03, d1_action6=0.05),
        gripper_magnitude_metrics=_metrics(best=0.0018, phase1=0.007, action6=0.015, d1_action6=0.03),
        audit_summary=_audit(),
        dataset_summary=_summary(),
    )

    assert "Stage 20-P" in text
    assert "gripper_magnitude_aux" in text
    assert "direction_1_action_dim_6_improved" in text
    assert "Recommendation" in text


def test_cli_writes_magnitude_report(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    magnitude = tmp_path / "magnitude.json"
    audit = tmp_path / "audit.json"
    summary = tmp_path / "summary.json"
    out = tmp_path / "report.md"
    baseline.write_text(json.dumps(_metrics()), encoding="utf-8")
    magnitude.write_text(json.dumps(_metrics()), encoding="utf-8")
    audit.write_text(json.dumps(_audit()), encoding="utf-8")
    summary.write_text(json.dumps(_summary()), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--baseline_metrics", str(baseline),
            "--gripper_magnitude_metrics", str(magnitude),
            "--audit_summary", str(audit),
            "--dataset_summary", str(summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Gripper Magnitude" in out.read_text(encoding="utf-8")
