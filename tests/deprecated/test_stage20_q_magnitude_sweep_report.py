import json

from vlm_nsm.experiments.stage20_q_magnitude_sweep_report import build_report, main


def _metrics(weight, best, phase1=0.008, direction1=0.003, action6=0.012):
    return {
        "gripper_magnitude_aux_weight": weight,
        "train_loss": best * 1.1,
        "final_val_loss": best,
        "best_val_loss": best,
        "best_epoch": 2,
        "per_phase_val_loss": {"1": phase1},
        "per_direction_val_loss": {"0": 0.0005, "1": direction1},
        "action_dim_mse": [0, 0, 0, 0, 0, 0, action6],
        "per_direction_action_dim_mse": {"1": [0, 0, 0, 0, 0, 0, action6 * 2]},
    }


def _audit():
    return {
        "direct_label_available": False,
        "proxy_only": True,
        "feasibility": {
            "recommended_for_training_now": False,
            "holding_phase_separation": 0.8,
            "contact_phase_separation": 0.7,
            "phase_leak_risk": True,
        },
        "direction_phase_comparison": {
            "direction1_phase1": {
                "sample_mse": 0.01,
                "action_dim6_mse": 0.08,
                "sign_accuracy": 0.97,
                "target_close_ratio": 0.15,
                "pred_close_ratio": 0.16,
            }
        },
    }


def test_magnitude_sweep_report_includes_best_candidate_and_contact_caveat():
    text = build_report(
        baseline_metrics=_metrics("baseline", 0.002),
        weight025_metrics=_metrics(0.25, 0.0018),
        weight050_metrics=_metrics(0.5, 0.0016),
        weight075_metrics=_metrics(0.75, 0.0017),
        contact_audit=_audit(),
        baseline_audit=_audit(),
        weight025_audit=_audit(),
        weight050_audit=_audit(),
        weight075_audit=_audit(),
    )

    assert "Stage 20-Q" in text
    assert "best_magnitude_run = weight_0.50_stage20_p" in text
    assert "Best-Checkpoint Direction 1 / Phase 1 Audit" in text
    assert "contact_holding_training = not_run" in text
    assert "recommended_for_training_now = false" in text


def test_magnitude_sweep_cli_writes_report(tmp_path, monkeypatch):
    paths = {}
    for name, payload in {
        "baseline": _metrics("baseline", 0.002),
        "w025": _metrics(0.25, 0.0018),
        "w050": _metrics(0.5, 0.0016),
        "w075": _metrics(0.75, 0.0017),
        "audit": _audit(),
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
            "--weight025_metrics", str(paths["w025"]),
            "--weight050_metrics", str(paths["w050"]),
            "--weight075_metrics", str(paths["w075"]),
            "--contact_audit", str(paths["audit"]),
            "--baseline_audit", str(paths["audit"]),
            "--weight025_audit", str(paths["audit"]),
            "--weight050_audit", str(paths["audit"]),
            "--weight075_audit", str(paths["audit"]),
            "--output", str(out),
        ],
    )

    main()

    assert out.exists()
    assert "Magnitude Sweep" in out.read_text(encoding="utf-8")
