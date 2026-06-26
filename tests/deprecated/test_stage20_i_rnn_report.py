import json

from vlm_nsm.experiments.stage20_i_rnn_report import build_report, main


def _metrics(best, final=None, phase1=0.007, phase3=0.002, action6=0.02):
    return {
        "model_type": "gru",
        "train_loss": best * 0.9,
        "val_loss": final if final is not None else best * 1.1,
        "final_val_loss": final if final is not None else best * 1.1,
        "best_val_loss": best,
        "best_epoch": 3,
        "per_direction_val_loss": {"0": best, "1": best + 0.001},
        "per_phase_val_loss": {"0": best * 0.1, "1": phase1, "3": phase3},
        "action_dim_mse": [0.001, 0.001, 0.001, 0.0, 0.0, 0.0, action6],
    }


def _summary():
    return {
        "num_trajectories_loaded": 3,
        "num_windows": 30,
        "train_windows": 24,
        "val_windows": 6,
        "direction_counts": {"cube1_to_cube2": 15, "cube2_to_cube1": 15},
        "phase_counts": {"0": 5, "1": 10, "3": 15},
        "teacher_profile_counts": {"cube2_robust": 30},
    }


def test_build_report_includes_comparison_caveat_and_recommendation():
    text = build_report(
        mlp_metrics=_metrics(0.0028632263, phase1=0.0083287518, phase3=0.0023744537, action6=0.03),
        rnn_metrics=_metrics(0.0027),
        dataset_summary=_summary(),
    )

    assert "MLP vs RNN" in text
    assert "rnn_beats_mlp_best_val_loss = true" in text
    assert "Phase 4/5" in text
    assert "Recommendation" in text


def test_cli_writes_report(tmp_path, monkeypatch):
    mlp = tmp_path / "mlp.json"
    rnn = tmp_path / "rnn.json"
    summary = tmp_path / "summary.json"
    out = tmp_path / "report.md"
    mlp.write_text(json.dumps(_metrics(0.0028632263)), encoding="utf-8")
    rnn.write_text(json.dumps(_metrics(0.0027)), encoding="utf-8")
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--mlp_metrics", str(mlp),
            "--rnn_metrics", str(rnn),
            "--dataset_summary", str(summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "offline supervised validation" in out.read_text(encoding="utf-8")
