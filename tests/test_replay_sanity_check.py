"""Tests for run_replay_sanity_check.py."""

import json
import tempfile
from pathlib import Path

import numpy as np

from vlm_nsm.experiments.run_replay_sanity_check import run_sanity_check


def _make_demo_npz(path: Path, T: int = 24, obs_dim: int = 16, action_dim: int = 4):
    """Create a mock demonstration npz file."""
    rng = np.random.RandomState(42)
    np.savez_compressed(
        path,
        obs=rng.randn(T, obs_dim).astype(np.float32),
        action=rng.randn(T, action_dim).astype(np.float32),
        done=np.zeros(T, dtype=bool),
        episode_id=np.zeros(T, dtype=np.int32),
        skill_id=np.tile(np.arange(6), T // 6 + 1)[:T].astype(np.int32),
        step_id=np.tile(np.arange(6), T // 6 + 1)[:T].astype(np.int32),
        success=np.ones(T, dtype=bool),
    )


def test_replay_sanity_with_mock_demo():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "test_demo.npz"
        _make_demo_npz(demo_path)
        summary = run_sanity_check(
            demo_path=str(demo_path),
            n_blocks=3,
        )
        assert "total_steps_in_data" in summary
        assert "step_sequence_valid" in summary


def test_replay_sanity_step_sequence():
    """Step sequence should be contiguous."""
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / "seq_test.npz"
        _make_demo_npz(demo_path)
        summary = run_sanity_check(demo_path=str(demo_path), n_blocks=3)
        assert summary["step_sequence_valid"] is True


def test_replay_sanity_missing_file():
    """Should raise on missing file."""
    try:
        run_sanity_check(demo_path="/nonexistent/demo.npz", n_blocks=3)
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_replay_sanity_with_metadata():
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
            "controller_type": "scripted",
            "step_names": [],
            "created_at": "2026-05-11T00:00:00",
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        summary = run_sanity_check(
            demo_path=str(demo_path),
            meta_path=str(meta_path),
            n_blocks=3,
        )
        assert summary["obs_dim_match"] is True
        assert summary["action_dim_match"] is True
