import argparse
import json

import pytest

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset
from vlm_nsm.training.train_stage20_goal_conditioned_rnn import train


def _args(dataset_dir, output_dir, *, overwrite=False):
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        model_type="gru",
        feature_mode="goal_conditioned",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        seed=0,
        device="cpu",
        sampler="none",
        loss_weighting="none",
        manual_phase_weights=None,
        weighted_metric_report=False,
        gripper_aux_weight=0.0,
        gripper_aux_phase_ids="1",
        overwrite=overwrite,
    )


def test_dataset_loader_return_sequence_shape(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    ds = Stage20RoleConditionedDataset(root, split="train", feature_mode="goal_conditioned", return_sequence=True)

    sample = ds[0]

    expected_dim = 11 + 20 + 16 + 16 + 20 + 20 + 1 + 6
    assert sample["features"].shape == (4, expected_dim)
    assert ds.input_dim == expected_dim


def test_gru_training_runs_one_epoch_on_fake_dataset_cpu(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"

    metrics = train(_args(dataset, output))

    assert (output / "best.pt").exists()
    assert (output / "last.pt").exists()
    assert (output / "config.json").exists()
    assert (output / "train_log.jsonl").exists()
    assert (output / "metrics.json").exists()
    assert (output / "loss_curve.json").exists()
    assert metrics["train_loss"] >= 0
    assert metrics["final_val_loss"] >= 0
    assert metrics["best_val_loss"] >= 0


def test_rnn_metrics_contain_required_fields(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    train(_args(dataset, output))

    metrics = json.loads((output / "metrics.json").read_text())
    assert "best_val_loss" in metrics
    assert "per_direction_val_loss" in metrics
    assert "per_phase_val_loss" in metrics
    assert "per_direction_phase_val_loss" in metrics
    assert "per_direction_action_dim_mse" in metrics
    assert "action_dim_mse" in metrics
    assert len(metrics["action_dim_mse"]) == 7


def test_rnn_output_dir_overwrite_protection(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    output.mkdir()

    with pytest.raises(FileExistsError):
        train(_args(dataset, output))

    train(_args(dataset, output, overwrite=True))
    assert (output / "metrics.json").exists()


def test_gripper_aux_head_runs_and_reports_metrics(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    args = _args(dataset, output)
    args.gripper_aux_weight = 0.2
    args.gripper_aux_phase_ids = "1"

    metrics = train(args)

    assert metrics["gripper_aux_weight"] == 0.2
    assert metrics["gripper_aux_phase_ids"] == [1]
    assert metrics["gripper_accuracy"] is not None
    assert "per_phase_gripper_accuracy" in metrics


def test_gripper_magnitude_aux_head_runs_and_reports_metrics(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    args = _args(dataset, output)
    args.gripper_magnitude_aux_weight = 0.5
    args.gripper_magnitude_aux_phase_ids = "1"

    metrics = train(args)

    assert metrics["gripper_magnitude_aux_weight"] == 0.5
    assert metrics["gripper_magnitude_aux_phase_ids"] == [1]
    assert metrics["gripper_magnitude_mse"] is not None
    assert "per_phase_gripper_magnitude_mse" in metrics
    assert "per_direction_phase_gripper_magnitude_mse" in metrics
