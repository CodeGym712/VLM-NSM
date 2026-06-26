import json

from vlm_nsm.experiments.stage20_j_lstm_report import build_report, main


def _metrics(model_type, best, phase1=0.007, phase3=0.002, action6=0.017, gap_hi=0.003):
    return {
        "model_type": model_type,
        "train_loss": best * 0.9,
        "final_val_loss": best * 1.05,
        "val_loss": best * 1.05,
        "best_val_loss": best,
        "best_epoch": 4,
        "per_direction_val_loss": {"0": best, "1": gap_hi},
        "per_phase_val_loss": {"0": best * 0.1, "1": phase1, "3": phase3},
        "action_dim_mse": [0.001, 0.001, 0.001, 0.0, 0.0, 0.0, action6],
    }


def _summary():
    return {
        "num_trajectories_loaded": 4,
        "num_windows": 40,
        "train_windows": 32,
        "val_windows": 8,
        "direction_counts": {"cube1_to_cube2": 20, "cube2_to_cube1": 20},
        "phase_counts": {"0": 5, "1": 10, "3": 25},
        "teacher_profile_counts": {"cube2_robust": 40},
    }


def test_build_report_includes_lstm_vs_gru_and_caveat():
    text = build_report(
        mlp_metrics=_metrics("mlp", 0.00286, phase1=0.0083, phase3=0.00237, action6=0.020),
        gru_metrics=_metrics("gru", 0.00248, phase1=0.0077, phase3=0.0017, action6=0.017, gap_hi=0.0031),
        lstm_metrics=_metrics("lstm", 0.00230, phase1=0.0070, phase3=0.0015, action6=0.016, gap_hi=0.0028),
        dataset_summary=_summary(),
    )

    assert "Stage 20-J" in text
    assert "LSTM" in text
    assert "lstm_beats_gru_best_val_loss = true" in text
    assert "Phase 4/5" in text
    assert "Recommendation" in text


def test_cli_writes_lstm_report(tmp_path, monkeypatch):
    mlp = tmp_path / "mlp.json"
    gru = tmp_path / "gru.json"
    lstm = tmp_path / "lstm.json"
    summary = tmp_path / "summary.json"
    out = tmp_path / "report.md"
    mlp.write_text(json.dumps(_metrics("mlp", 0.00286)), encoding="utf-8")
    gru.write_text(json.dumps(_metrics("gru", 0.00248)), encoding="utf-8")
    lstm.write_text(json.dumps(_metrics("lstm", 0.00230)), encoding="utf-8")
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--mlp_metrics", str(mlp),
            "--gru_metrics", str(gru),
            "--lstm_metrics", str(lstm),
            "--dataset_summary", str(summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "offline supervised validation" in out.read_text(encoding="utf-8")
