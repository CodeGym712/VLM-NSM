import json

from vlm_nsm.experiments.stage20_l_phase_labeled_lstm_report import build_report, main


def _metrics(best, phases):
    return {
        "train_loss": best * 0.9,
        "final_val_loss": best * 1.05,
        "val_loss": best * 1.05,
        "best_val_loss": best,
        "best_epoch": 3,
        "per_direction_val_loss": {"0": best, "1": best + 0.001},
        "per_phase_val_loss": phases,
        "action_dim_mse": [0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.02],
    }


def _summary(phase_counts):
    return {
        "num_windows": 100,
        "train_windows": 80,
        "val_windows": 20,
        "direction_counts": {"cube1_to_cube2": 50, "cube2_to_cube1": 50},
        "phase_counts": phase_counts,
    }


def test_build_report_includes_phase_4_5_and_recommendation():
    text = build_report(
        original_metrics=_metrics(0.0030, {"0": 0.1, "1": 0.009, "3": 0.003}),
        labeled_metrics=_metrics(0.0028, {"0": 0.1, "1": 0.008, "3": 0.002, "4": 0.001, "5": 0.0015}),
        combined_lstm_metrics=_metrics(0.0027, {"0": 0.1, "1": 0.007, "3": 0.002}),
        original_summary=_summary({"0": 10, "1": 20, "3": 70}),
        labeled_summary=_summary({"0": 10, "1": 20, "3": 40, "4": 10, "5": 20}),
    )

    assert "Stage 20-L" in text
    assert "phase_4_5_present = true" in text
    assert "release_retreat_losses_available = true" in text
    assert "Recommendation" in text


def test_cli_writes_report(tmp_path, monkeypatch):
    original_metrics = tmp_path / "original_metrics.json"
    labeled_metrics = tmp_path / "labeled_metrics.json"
    combined_metrics = tmp_path / "combined_metrics.json"
    original_summary = tmp_path / "original_summary.json"
    labeled_summary = tmp_path / "labeled_summary.json"
    out = tmp_path / "report.md"
    original_metrics.write_text(json.dumps(_metrics(0.0030, {"1": 0.009, "3": 0.003})), encoding="utf-8")
    labeled_metrics.write_text(json.dumps(_metrics(0.0028, {"1": 0.008, "3": 0.002, "4": 0.001, "5": 0.0015})), encoding="utf-8")
    combined_metrics.write_text(json.dumps(_metrics(0.0027, {"1": 0.007, "3": 0.002})), encoding="utf-8")
    original_summary.write_text(json.dumps(_summary({"0": 10, "1": 20, "3": 70})), encoding="utf-8")
    labeled_summary.write_text(json.dumps(_summary({"0": 10, "1": 20, "3": 40, "4": 10, "5": 20})), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--original_metrics", str(original_metrics),
            "--labeled_metrics", str(labeled_metrics),
            "--combined_lstm_metrics", str(combined_metrics),
            "--original_summary", str(original_summary),
            "--labeled_summary", str(labeled_summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "offline supervised validation" in out.read_text(encoding="utf-8")
