import json

from vlm_nsm.experiments.stage20_n_gripper_aux_report import build_report, main


def _metrics(best, phase1=0.008, gap_hi=0.003, action6=0.012, gripper=True):
    data = {
        "train_loss": best * 0.9,
        "final_val_loss": best,
        "val_loss": best,
        "best_val_loss": best,
        "best_epoch": 5,
        "per_direction_val_loss": {"0": 0.0005, "1": gap_hi},
        "per_phase_val_loss": {"1": phase1, "3": 0.00005, "4": 0.00002, "5": 0.00001},
        "per_direction_phase_val_loss": {"1:1": phase1 + 0.001, "1:3": 0.0001},
        "per_direction_action_dim_mse": {"0": [0, 0, 0, 0, 0, 0, 0.005], "1": [0, 0, 0, 0, 0, 0, action6]},
        "action_dim_mse": [0, 0, 0, 0, 0, 0, action6],
    }
    if gripper:
        data.update({
            "gripper_accuracy": 0.95,
            "gripper_bce": 0.1,
            "per_phase_gripper_accuracy": {"1": 0.9, "4": 0.99},
            "per_direction_gripper_accuracy": {"0": 0.96, "1": 0.94},
        })
    return data


def _summary():
    return {
        "num_windows": 100,
        "train_windows": 80,
        "val_windows": 20,
        "direction_counts": {"cube1_to_cube2": 50, "cube2_to_cube1": 50},
        "phase_counts": {"1": 20, "4": 10, "5": 20},
    }


def test_build_report_includes_gripper_and_direction_analysis():
    text = build_report(
        baseline_metrics=_metrics(0.0017, phase1=0.0085, gap_hi=0.003, action6=0.012, gripper=False),
        gripper_aux_metrics=_metrics(0.0018, phase1=0.0075, gap_hi=0.0025, action6=0.010),
        dataset_summary=_summary(),
    )

    assert "Stage 20-N" in text
    assert "Gripper Classification Metrics" in text
    assert "Direction-1 Error Analysis" in text
    assert "phase_1_improved = true" in text
    assert "Recommendation" in text


def test_cli_writes_gripper_aux_report(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    aux = tmp_path / "aux.json"
    summary = tmp_path / "summary.json"
    out = tmp_path / "report.md"
    baseline.write_text(json.dumps(_metrics(0.0017, gripper=False)), encoding="utf-8")
    aux.write_text(json.dumps(_metrics(0.0018)), encoding="utf-8")
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--baseline_metrics", str(baseline),
            "--gripper_aux_metrics", str(aux),
            "--dataset_summary", str(summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "close/open" in out.read_text(encoding="utf-8")
