import argparse
import json

import pytest

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.training.train_stage20_goal_conditioned_mlp import train


def _args(dataset_dir, output_dir, *, overwrite=False):
    return argparse.Namespace(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        feature_mode="goal_conditioned",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        hidden_dim=16,
        num_layers=1,
        seed=0,
        device="cpu",
        overwrite=overwrite,
    )


def test_training_script_runs_one_epoch_on_fake_dataset_cpu(tmp_path):
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
    assert metrics["val_loss"] >= 0


def test_metrics_contain_required_fields(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    train(_args(dataset, output))

    metrics = json.loads((output / "metrics.json").read_text())
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "per_direction_val_loss" in metrics
    assert "per_phase_val_loss" in metrics
    assert "action_dim_mse" in metrics
    assert len(metrics["action_dim_mse"]) == 7


def test_output_dir_overwrite_protection(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    output.mkdir()

    with pytest.raises(FileExistsError):
        train(_args(dataset, output))

    train(_args(dataset, output, overwrite=True))
    assert (output / "metrics.json").exists()
