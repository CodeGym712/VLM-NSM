import json

from vlm_nsm.experiments.stage20_m_balanced_lstm_report import build_report, main


def _metrics(best, phase1=0.008, gap_hi=0.0025, phase4=0.00002, phase5=0.00001, action6=0.011):
    return {
        "train_loss": best * 0.9,
        "final_val_loss": best * 1.05,
        "val_loss": best * 1.05,
        "best_val_loss": best,
        "best_epoch": 4,
        "per_direction_val_loss": {"0": 0.0005, "1": gap_hi},
        "per_phase_val_loss": {"1": phase1, "3": 0.00005, "4": phase4, "5": phase5},
        "action_dim_mse": [0.001, 0.001, 0.001, 0.0, 0.0, 0.0, action6],
    }


def _summary():
    return {
        "num_windows": 100,
        "train_windows": 80,
        "val_windows": 20,
        "direction_counts": {"cube1_to_cube2": 50, "cube2_to_cube1": 50},
        "phase_counts": {"1": 20, "4": 10, "5": 20},
        "teacher_profile_counts": {"cube2_robust": 100},
    }


def test_build_report_includes_baseline_balanced_phase_and_direction():
    text = build_report(
        baseline_metrics=_metrics(0.00172, phase1=0.00857, gap_hi=0.00299, action6=0.011964),
        phase_balanced_metrics=_metrics(0.0018, phase1=0.0075, gap_hi=0.0028, action6=0.011),
        phase_direction_balanced_metrics=_metrics(0.00185, phase1=0.0077, gap_hi=0.0020, action6=0.010),
        weighted_loss_metrics=_metrics(0.0019, phase1=0.0078, gap_hi=0.0027, action6=0.0105),
        dataset_summary=_summary(),
    )

    assert "Stage 20-M" in text
    assert "phase_balanced" in text
    assert "phase_direction_balanced" in text
    assert "Phase 1" in text
    assert "Direction Gap" in text
    assert "phase4_5=true" in text
    assert "Recommendation" in text


def test_cli_writes_balanced_lstm_report(tmp_path, monkeypatch):
    paths = {}
    for name, payload in {
        "baseline": _metrics(0.00172),
        "phase_balanced": _metrics(0.0018),
        "phase_direction": _metrics(0.00185),
        "summary": _summary(),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    out = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--baseline_metrics", str(paths["baseline"]),
            "--phase_balanced_metrics", str(paths["phase_balanced"]),
            "--phase_direction_balanced_metrics", str(paths["phase_direction"]),
            "--dataset_summary", str(paths["summary"]),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Validation metrics are unweighted" in out.read_text(encoding="utf-8")
