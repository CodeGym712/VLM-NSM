import argparse
import json

import numpy as np
import pytest

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset
from vlm_nsm.training.train_stage20_goal_conditioned_rnn import (
    build_sampler_weights,
    train,
)


def _args(dataset_dir, output_dir, **overrides):
    values = dict(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        model_type="lstm",
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
        overwrite=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_phase_balanced_sampler_inverse_frequency(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset", n=19)
    ds = Stage20RoleConditionedDataset(root, split="train", return_sequence=True)
    weights = build_sampler_weights(ds, "phase_balanced")

    assert weights is not None
    phase_to_weights = {}
    for record, weight in zip(ds.records, weights):
        phase_to_weights.setdefault(record["phase_id"], []).append(float(weight))
    counts = {phase: len(values) for phase, values in phase_to_weights.items()}
    for phase, values in phase_to_weights.items():
        assert np.allclose(values, values[0])
        for other, other_values in phase_to_weights.items():
            if counts[phase] > counts[other]:
                assert values[0] < other_values[0]


def test_phase_direction_balanced_sampler_handles_sparse_bins(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset", n=18)
    ds = Stage20RoleConditionedDataset(root, split="train", return_sequence=True)
    ds.records[0]["direction_id"] = 99

    weights = build_sampler_weights(ds, "phase_direction_balanced")

    assert weights is not None
    assert len(weights) == len(ds)
    assert np.isfinite(weights).all()
    assert float(weights.max()) > float(weights.min())


def test_weighted_loss_path_runs_one_epoch_cpu(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "weighted"

    metrics = train(
        _args(
            dataset,
            output,
            loss_weighting="manual",
            manual_phase_weights='{"0":1.0,"1":2.0,"2":1.0,"3":1.0}',
            weighted_metric_report=True,
        )
    )

    assert metrics["loss_weighting"] == "manual"
    assert metrics["manual_phase_weights"]["1"] == 2.0
    assert metrics["best_val_loss"] >= 0
    assert (output / "best.pt").exists()


def test_default_sampler_behavior_remains_unchanged(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "default"

    metrics = train(_args(dataset, output))
    config = json.loads((output / "config.json").read_text())

    assert metrics["sampler"] == "none"
    assert metrics["loss_weighting"] == "none"
    assert config["sampler"] == "none"
    assert config["loss_weighting"] == "none"


def test_balanced_rnn_overwrite_protection_still_works(tmp_path):
    dataset = write_fake_stage20_dataset(tmp_path / "dataset", n=16)
    output = tmp_path / "model"
    output.mkdir()

    with pytest.raises(FileExistsError):
        train(_args(dataset, output, sampler="phase_balanced"))

    train(_args(dataset, output, sampler="phase_balanced", overwrite=True))
    assert (output / "metrics.json").exists()
