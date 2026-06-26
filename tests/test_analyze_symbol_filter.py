"""Tests for analyze_symbol_filter.py."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.analyze_symbol_filter import analyze_filter_stats


def _make_demo_npz(path: Path, T: int = 24, obs_dim: int = 16, action_dim: int = 4, include_full_obs: bool = False, full_obs_dim: int = 36):
    rng = np.random.RandomState(42)
    save_dict = dict(
        obs=rng.randn(T, obs_dim).astype(np.float32),
        action=rng.randn(T, action_dim).astype(np.float32),
        done=np.zeros(T, dtype=bool),
        episode_id=np.zeros(T, dtype=np.int32),
        skill_id=np.tile(np.arange(6), T // 6 + 1)[:T].astype(np.int32),
        step_id=np.tile(np.arange(6), T // 6 + 1)[:T].astype(np.int32),
        success=np.ones(T, dtype=bool),
        filtered_obs=rng.randn(T, obs_dim).astype(np.float32),
    )
    if include_full_obs:
        save_dict["full_obs"] = rng.randn(T, full_obs_dim).astype(np.float32)
    np.savez_compressed(path, **save_dict)


def test_analyze_filter_obs_mean_std():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "stats_test.npz"
        _make_demo_npz(demo_path)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert "obs_mean" in stats
        assert "obs_std" in stats
        assert isinstance(stats["obs_mean"], float)
        assert isinstance(stats["obs_std"], float)


def test_analyze_filter_num_transitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "transitions_test.npz"
        _make_demo_npz(demo_path, T=36)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert stats["num_transitions"] == 36


def test_analyze_filter_filtered_obs_dim():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "dim_test.npz"
        _make_demo_npz(demo_path, obs_dim=32)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert stats["filtered_obs_dim"] == 32


def test_analyze_filter_step_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "steps_test.npz"
        _make_demo_npz(demo_path, T=12)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert "step_distribution" in stats
        assert len(stats["step_distribution"]) > 0


def test_analyze_filter_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "meta_test.npz"
        meta_path = Path(tmpdir) / "meta_test.meta.json"
        _make_demo_npz(demo_path)
        meta = {
            "task": "stacking",
            "n_blocks": 3,
            "obs_dim": 16,
            "action_dim": 4,
            "num_episodes": 1,
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        stats = analyze_filter_stats(
            demo_path=str(demo_path),
            meta_path=str(meta_path),
        )
        assert stats["full_obs_dim"] == 16


def test_analyze_filter_full_obs_available():
    """When full_obs exists in npz, reduction_ratio should be computed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "full_obs_test.npz"
        _make_demo_npz(demo_path, obs_dim=16, include_full_obs=True, full_obs_dim=36)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert stats["full_obs_available"] is True
        assert stats["full_obs_dim"] == 36
        assert stats["filtered_obs_dim"] == 16
        expected_rr = (36 - 16) / 36
        assert stats["reduction_ratio"] == pytest.approx(expected_rr, abs=1e-5)


def test_analyze_filter_full_obs_not_available():
    """When full_obs is missing, should not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "no_full_obs.npz"
        _make_demo_npz(demo_path, include_full_obs=False)
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert stats["full_obs_available"] is False
        assert stats["reduction_ratio"] is None


def test_analyze_filter_full_obs_dim_mismatch():
    """When filtered_obs_dim differs from obs, use filtered_obs array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "dim_mismatch.npz"
        # Create npz where obs has dim 16 but filtered_obs has dim 8
        rng = np.random.RandomState(42)
        np.savez_compressed(
            demo_path,
            obs=rng.randn(24, 16).astype(np.float32),
            filtered_obs=rng.randn(24, 8).astype(np.float32),
            full_obs=rng.randn(24, 36).astype(np.float32),
            action=rng.randn(24, 4).astype(np.float32),
            done=np.zeros(24, dtype=bool),
            episode_id=np.zeros(24, dtype=np.int32),
            skill_id=np.tile(np.arange(6), 5)[:24].astype(np.int32),
            step_id=np.tile(np.arange(6), 5)[:24].astype(np.int32),
            success=np.ones(24, dtype=bool),
        )
        stats = analyze_filter_stats(demo_path=str(demo_path))
        assert stats["full_obs_available"] is True
        assert stats["full_obs_dim"] == 36
        assert stats["filtered_obs_dim"] == 8
        expected_rr = (36 - 8) / 36
        assert stats["reduction_ratio"] == pytest.approx(expected_rr, abs=1e-5)
