"""Tests for collect_stacking_demos.py."""

import json
import tempfile
from pathlib import Path

import numpy as np

from vlm_nsm.experiments.collect_stacking_demos import collect_demos


def test_collect_demos_generates_npz():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "stacking_3_test.npz"
        summary = collect_demos(
            n_blocks=3,
            num_episodes=5,
            env_type="mock",
            controller_type="scripted",
            output_path=str(output),
        )

        assert output.exists()
        data = np.load(output)
        assert "obs" in data
        assert "action" in data
        assert "done" in data
        assert "episode_id" in data
        assert "skill_id" in data
        assert "step_id" in data
        assert "success" in data

        # Metadata
        meta_path = output.with_suffix(".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["task"] == "stacking"
        assert meta["n_blocks"] == 3
        assert meta["controller_type"] == "scripted"
        assert meta["num_episodes"] == 5
        assert meta["obs_dim"] == 16
        assert meta["action_dim"] == 4
        assert "step_names" in meta
        assert "created_at" in meta

        # Verify summary
        assert summary["num_episodes"] == 5
        assert summary["episode_success_rate"] == 1.0
        assert summary["num_success"] == 5


def test_collect_demos_npz_shapes():
    """Verify npz shapes are correct for BC training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "test_shapes.npz"
        collect_demos(
            n_blocks=3,
            num_episodes=3,
            controller_type="scripted",
            output_path=str(output),
        )

        data = np.load(output)
        T = data["obs"].shape[0]
        assert data["obs"].shape == (T, 16)
        assert data["action"].shape == (T, 4)
        assert data["done"].shape == (T,)
        assert data["episode_id"].shape == (T,)
        assert data["step_id"].shape == (T,)
        assert data["success"].shape == (T,)


def test_collect_demos_without_output():
    """Should work without saving npz."""
    summary = collect_demos(
        n_blocks=3,
        num_episodes=3,
        controller_type="scripted",
        output_path=None,
    )
    assert summary["num_episodes"] == 3
    assert summary["npz_path"] is None
