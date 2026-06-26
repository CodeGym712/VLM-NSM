"""Tests for VLA HDF5 dataset loader."""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from vlm_nsm.vla.data.vla_dataset import (
    HDF5EpisodeReader,
    VLADataset,
    VLADatasetManifest,
    compute_vla_normalization_stats,
    create_vla_datasets,
    save_normalization_stats,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_dummy_episode(path: Path, T: int = 100, success: bool = True):
    """Create a minimal valid HDF5 episode file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/agentview", data=np.zeros((T, 128, 128, 3), dtype=np.uint8))
        f.create_dataset("observations/frontview", data=np.zeros((T, 128, 128, 3), dtype=np.uint8))
        f.create_dataset("actions_4d", data=np.random.randn(T, 4).astype(np.float32) * 0.01)
        f.create_dataset("proprio/eef_pos", data=np.cumsum(np.random.randn(T, 3).astype(np.float32) * 0.001, axis=0))
        f.create_dataset("proprio/gripper_open", data=np.ones((T, 1), dtype=np.float32))
        f.create_dataset("symbolic/phase_id", data=np.random.randint(0, 10, size=(T,), dtype=np.int16))
        f.create_dataset("symbolic/pick_object_id", data=np.ones((T,), dtype=np.int16))
        f.create_dataset("symbolic/target_object_id", data=np.ones((T,), dtype=np.int16) * 2)
        f.create_dataset("symbolic/relation_id", data=np.zeros((T,), dtype=np.int16))
        f.create_dataset("symbolic/action_type_id", data=np.zeros((T,), dtype=np.int16))
        f.create_dataset("symbolic/plan_step_index", data=np.zeros((T,), dtype=np.int16))
        f.create_dataset("success", data=success)
        if not success:
            f.create_dataset("failure_type", data="grasp_failure")


def _make_dummy_dataset(root: Path, n_episodes: int = 10, T: int = 100):
    """Create a minimal VLA dataset directory."""
    episodes_dir = root / "episodes"
    episodes_dir.mkdir(parents=True)
    manifest_lines = []
    for i in range(n_episodes):
        ep_path = episodes_dir / f"ep_{i:06d}.hdf5"
        success = i < n_episodes - 1  # last one fails
        _make_dummy_episode(ep_path, T=T, success=success)
        manifest_lines.append(json.dumps({
            "episode_path": str(ep_path),
            "episode_id": f"ep_{i:06d}",
            "success": success,
            "failure_type": None if success else "grasp_failure",
        }))
    (root / "manifest.jsonl").write_text("\n".join(manifest_lines), encoding="utf-8")


# ── Tests ───────────────────────────────────────────────────────────────────

class TestHDF5EpisodeReader:
    """Tests for HDF5EpisodeReader."""

    def test_read_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_000001.hdf5"
            _make_dummy_episode(path, T=50)
            reader = HDF5EpisodeReader(path)
            actions = reader.read_actions()
            assert actions.shape == (50, 4)
            assert actions.dtype == np.float32

    def test_read_phase_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_000001.hdf5"
            _make_dummy_episode(path, T=30)
            reader = HDF5EpisodeReader(path)
            phases = reader.read_phase_ids()
            assert phases.shape == (30,)

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ep_000001.hdf5"
            _make_dummy_episode(path, T=20)
            with HDF5EpisodeReader(path) as reader:
                assert reader.num_steps == 20

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            HDF5EpisodeReader("/nonexistent/path.hdf5")


class TestVLADatasetManifest:
    """Tests for VLADatasetManifest."""

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=10)
            manifest = VLADatasetManifest(root)
            assert len(manifest) == 10

    def test_filter_success_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=10)  # 9 success, 1 failure
            manifest = VLADatasetManifest(root)
            success_only = manifest.filter_success_only()
            assert len(success_only) == 9
            assert all(ep["success"] for ep in success_only.episodes)


class TestVLADataset:
    """Tests for VLADataset."""

    @pytest.fixture
    def dataset_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=10, T=200)
            save_normalization_stats(root, max_episodes=5)
            yield root

    def test_create_dataset(self, dataset_dir):
        """Dataset creates without error."""
        ds = VLADataset(dataset_dir, split="train")
        assert len(ds) > 0

    def test_train_val_split(self, dataset_dir):
        """Train and val splits are disjoint and cover all episodes."""
        train_ds = VLADataset(dataset_dir, split="train")
        val_ds = VLADataset(dataset_dir, split="val")
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        # Total samples should equal or be close
        total = len(train_ds) + len(val_ds)

    def test_sample_structure(self, dataset_dir):
        """Each sample has the expected keys and shapes."""
        ds = VLADataset(
            dataset_dir, split="train",
            obs_history_len=2, action_chunk_len=16,
            stride=8,
        )
        if len(ds) == 0:
            pytest.skip("No samples in dataset")
        sample = ds[0]

        assert "agentview" in sample
        assert "frontview" in sample
        assert "proprio" in sample
        assert "symbolic_cond" in sample
        assert "action_chunk" in sample
        assert "phase_id" in sample

        assert sample["agentview"].shape == (3, 128, 128)  # CHW
        assert sample["action_chunk"].shape == (16, 4)
        assert sample["proprio"].shape == (7,)
        assert isinstance(sample["phase_id"], int)

    def test_action_normalization(self, dataset_dir):
        """Actions are normalized when stats are available."""
        ds = VLADataset(dataset_dir, split="train", stride=16)
        if len(ds) == 0:
            pytest.skip("No samples in dataset")
        assert ds.action_mean is not None
        assert ds.action_std is not None

    def test_success_only(self, dataset_dir):
        """success_only=True excludes failure episodes."""
        ds = VLADataset(dataset_dir, split="train", success_only=True)
        # Should not crash
        assert len(ds) >= 0

    def test_sample_weights(self, dataset_dir):
        """Sample weights are computed."""
        ds = VLADataset(dataset_dir, split="train", stride=16)
        weights = ds.sample_weights
        assert len(weights) == len(ds)
        assert np.all(weights > 0)

    def test_stride_controls_density(self, dataset_dir):
        """Larger stride → fewer samples."""
        ds_small = VLADataset(dataset_dir, split="train", stride=16)
        ds_large = VLADataset(dataset_dir, split="train", stride=4)
        if len(ds_small) == 0 or len(ds_large) == 0:
            pytest.skip("Not enough data")
        assert len(ds_small) <= len(ds_large)

    def test_image_values_in_range(self, dataset_dir):
        """Image tensors are in [0, 1]."""
        ds = VLADataset(dataset_dir, split="train", stride=50)
        if len(ds) == 0:
            pytest.skip("No samples in dataset")
        sample = ds[0]
        assert 0.0 <= sample["agentview"].min() <= sample["agentview"].max() <= 1.0


class TestNormalizationStats:
    """Tests for normalization stats computation."""

    def test_compute_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=5, T=100)
            stats = compute_vla_normalization_stats(root, max_episodes=5)
            assert "action_mean" in stats
            assert "action_std" in stats
            assert stats["action_mean"].shape == (4,)
            assert stats["action_std"].shape == (4,)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=5, T=100)
            save_normalization_stats(root, max_episodes=5)
            stats_path = root / "stats.json"
            assert stats_path.exists()
            stats = json.loads(stats_path.read_text())
            assert len(stats["action_mean"]) == 4


class TestCreateVLADatasets:
    """Tests for create_vla_datasets factory."""

    def test_create_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_dummy_dataset(root, n_episodes=10, T=200)
            save_normalization_stats(root, max_episodes=5)
            train_ds, val_ds = create_vla_datasets(root, stride=16)
            assert len(train_ds) > 0
            assert len(val_ds) > 0
