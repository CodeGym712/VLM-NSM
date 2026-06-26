import torch

from stage20_fake_dataset_utils import write_fake_stage20_dataset
from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset


def test_dataset_loader_loads_fake_dataset(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    ds = Stage20RoleConditionedDataset(root, split="train", feature_mode="goal_conditioned")

    assert len(ds) == 8
    sample = ds[0]
    assert sample["action"].shape == (7,)
    assert sample["direction_label"] in {0, 1}


def test_obs_only_feature_mode_shape_is_correct(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    ds = Stage20RoleConditionedDataset(root, split="train", feature_mode="obs_only")

    assert ds[0]["features"].shape == (4 * 11,)


def test_goal_conditioned_feature_mode_shape_is_correct(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    ds = Stage20RoleConditionedDataset(root, split="train", feature_mode="goal_conditioned")

    expected = 4 * (11 + 20 + 16 + 16 + 20 + 20 + 1 + 6)
    assert ds[0]["features"].shape == (expected,)


def test_normalization_produces_finite_tensors(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    ds = Stage20RoleConditionedDataset(root, split="train", feature_mode="goal_conditioned", normalize=True)

    sample = ds[0]
    assert torch.isfinite(sample["features"]).all()
    assert torch.isfinite(sample["action"]).all()


def test_train_val_split_works(tmp_path):
    root = write_fake_stage20_dataset(tmp_path / "dataset")
    train = Stage20RoleConditionedDataset(root, split="train")
    val = Stage20RoleConditionedDataset(root, split="val")
    all_ds = Stage20RoleConditionedDataset(root, split="all")

    assert len(train) == 8
    assert len(val) == 4
    assert len(all_ds) == 12
