import argparse
import json

import torch

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset
from vlm_nsm.experiments.stage20_o_grasp_error_audit import build_markdown_report, main, run_audit
from vlm_nsm.training.train_stage20_goal_conditioned_rnn import GoalConditionedRNN


def _write_checkpoint(dataset_dir, checkpoint_path):
    ds = Stage20RoleConditionedDataset(dataset_dir, split="val", return_sequence=True)
    model = GoalConditionedRNN(
        input_dim=ds.input_dim,
        action_dim=ds.action_dim,
        model_type="lstm",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": ds.input_dim,
                "action_dim": ds.action_dim,
                "model_type": "lstm",
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
        },
        checkpoint_path,
    )


def test_grasp_error_audit_runs_on_fake_dataset(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(dataset, checkpoint)

    audit = run_audit(
        argparse.Namespace(
            dataset_dir=str(dataset),
            checkpoint=str(checkpoint),
            split="val",
            batch_size=4,
            device="cpu",
            top_k=3,
        )
    )

    assert audit["stage"] == "stage20_o"
    assert audit["num_samples"] == 4
    assert "1:1" in audit["groups"]["by_direction_phase"]
    assert "direction1_phase1" in audit["direction_phase_comparison"]
    assert audit["diagnosis"]["recommendation"]


def test_grasp_error_audit_markdown_includes_diagnosis(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)
    checkpoint = tmp_path / "best.pt"
    _write_checkpoint(dataset, checkpoint)
    audit = run_audit(
        argparse.Namespace(
            dataset_dir=str(dataset),
            checkpoint=str(checkpoint),
            split="val",
            batch_size=4,
            device="cpu",
            top_k=3,
        )
    )

    text = build_markdown_report(audit)

    assert "Stage 20-O" in text
    assert "Direction 0 vs Direction 1" in text
    assert "Action Dim 6 Distribution" in text
    assert "Trajectory Anomalies" in text
    assert "Recommendation" in text


def test_grasp_error_audit_cli_writes_json_and_markdown(tmp_path, monkeypatch):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=20)
    checkpoint = tmp_path / "best.pt"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    _write_checkpoint(dataset, checkpoint)

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--dataset_dir", str(dataset),
            "--checkpoint", str(checkpoint),
            "--split", "val",
            "--batch_size", "4",
            "--device", "cpu",
            "--output_json", str(output_json),
            "--output_md", str(output_md),
        ],
    )

    main()

    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["stage"] == "stage20_o"
    assert "likely_error_sources" in payload["diagnosis"]
