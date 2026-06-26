import json

from vlm_nsm.experiments.stage20_e_report import build_report, main


def _metrics(val_loss):
    return {
        "train_loss": val_loss / 2,
        "val_loss": val_loss,
        "best_epoch": 1,
        "input_dim": 10,
        "per_direction_val_loss": {"0": val_loss, "1": val_loss + 0.1},
        "per_phase_val_loss": {"0": val_loss},
    }


def test_report_generator_produces_markdown_comparison():
    text = build_report(
        _metrics(0.2),
        _metrics(0.1),
        {"num_windows": 10, "train_windows": 8, "val_windows": 2},
    )

    assert "Stage 20-E Goal-Conditioned MLP Baseline Report" in text
    assert "obs_only" in text
    assert "goal_conditioned" in text
    assert "goal_conditioned_improves_val_loss = true" in text


def test_report_cli_writes_markdown(tmp_path, monkeypatch):
    obs = tmp_path / "obs.json"
    goal = tmp_path / "goal.json"
    summary = tmp_path / "summary.json"
    out = tmp_path / "report.md"
    obs.write_text(json.dumps(_metrics(0.2)))
    goal.write_text(json.dumps(_metrics(0.1)))
    summary.write_text(json.dumps({"num_windows": 10, "train_windows": 8, "val_windows": 2}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--obs_only_metrics", str(obs),
            "--goal_metrics", str(goal),
            "--dataset_summary", str(summary),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Comparison" in out.read_text()
