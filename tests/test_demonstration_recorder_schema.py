"""Tests for DemonstrationRecorder npz schema and metadata."""

import json
import tempfile
from pathlib import Path

import numpy as np

from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder


def test_recorder_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=4, action_dim=2)

        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        action = np.array([0.5, 0.6], dtype=np.float32)

        # Record 3 steps across 2 episodes
        recorder.record_step(obs, action, done=False, skill_id=0, step_id=0, success=True)
        recorder.record_step(obs, action, done=False, skill_id=1, step_id=1, success=True)
        recorder.record_step(obs, action, done=True, skill_id=2, step_id=2, success=True)
        recorder.end_episode()

        recorder.record_step(obs, action, done=False, skill_id=0, step_id=0, success=True)
        recorder.record_step(obs, action, done=True, skill_id=1, step_id=1, success=False)
        recorder.end_episode()

        npz_path, meta_path = recorder.save(
            "test_demo.npz",
            task="stacking",
            n_blocks=3,
            env_type="mock",
            controller_type="scripted",
            step_names=["reach_pick", "grasp", "lift"],
        )

        # Verify npz exists and has correct schema
        assert npz_path.exists()
        data = np.load(npz_path)
        assert "obs" in data
        assert "filtered_obs" in data
        assert "action" in data
        assert "done" in data
        assert "episode_id" in data
        assert "skill_id" in data
        assert "step_id" in data
        assert "success" in data

        assert data["obs"].shape == (5, 4)  # 5 steps, 4-dim obs
        assert data["filtered_obs"].shape == (5, 4)
        assert data["action"].shape == (5, 2)  # 5 steps, 2-dim action
        assert data["done"].shape == (5,)
        assert data["episode_id"].shape == (5,)
        assert data["skill_id"].shape == (5,)
        assert data["step_id"].shape == (5,)
        assert data["success"].shape == (5,)

        # filtered_obs should match obs when not separately provided
        assert np.allclose(data["obs"], data["filtered_obs"])

        # Verify metadata
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["task"] == "stacking"
        assert meta["n_blocks"] == 3
        assert meta["env_type"] == "mock"
        assert meta["controller_type"] == "scripted"
        assert meta["num_episodes"] == 2
        assert meta["obs_dim"] == 4
        assert meta["action_dim"] == 2
        assert meta["step_names"] == ["reach_pick", "grasp", "lift"]
        assert meta["filtered_obs_dim"] == 4
        assert "full_obs_dim" in meta
        assert "full_obs_available" in meta
        assert meta["full_obs_available"] is False
        assert "created_at" in meta


def test_recorder_with_full_obs():
    """When full_obs is provided, it should be saved and metadata updated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=4, action_dim=2, full_obs_dim=10)
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        full_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
        action = np.array([0.5, 0.6], dtype=np.float32)

        recorder.record_step(obs, action, done=True, skill_id=0, step_id=0, success=True, full_obs=full_obs)
        recorder.end_episode()

        npz_path, meta_path = recorder.save("full_obs_test.npz", task="stacking", n_blocks=3)

        data = np.load(npz_path)
        assert "full_obs" in data
        assert data["full_obs"].shape == (1, 10)
        assert "filtered_obs" in data
        assert data["filtered_obs"].shape == (1, 4)

        meta = json.loads(meta_path.read_text())
        assert meta["full_obs_available"] is True
        assert meta["full_obs_dim"] == 10
        assert meta["filtered_obs_dim"] == 4
        assert meta["reduction_ratio"] == 1.0 - (4 / 10)


def test_recorder_with_filtered_obs():
    """When filtered_obs is explicitly provided, it should be saved separately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=4, action_dim=2)
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        filtered = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        action = np.array([0.5, 0.6], dtype=np.float32)

        recorder.record_step(obs, action, done=True, skill_id=0, step_id=0, success=True, filtered_obs=filtered)
        recorder.end_episode()

        npz_path, _ = recorder.save("filtered_test.npz", task="stacking", n_blocks=3)

        data = np.load(npz_path)
        assert data["obs"].shape == (1, 4)
        assert data["filtered_obs"].shape == (1, 4)
        # filtered_obs should be the explicitly provided one
        assert np.allclose(data["filtered_obs"][0], [0.5, 0.6, 0.7, 0.8])


def test_recorder_empty_save():
    """Saving with no recorded steps should not crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=4, action_dim=2)
        npz_path, meta_path = recorder.save("empty.npz", task="hanoi", n_blocks=3)
        assert npz_path.exists()
        assert meta_path.exists()


def test_recorder_clear():
    """Clear should reset all buffers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=4, action_dim=2)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        recorder.record_step(obs, action, done=True, skill_id=0, step_id=0, success=True)
        recorder.end_episode()
        assert recorder.total_steps == 1
        assert recorder.total_episodes == 1
        recorder.clear()
        assert recorder.total_steps == 0
        assert recorder.total_episodes == 0


def test_recorder_compatible_with_bc():
    """Verify npz fields match expected BC input format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = DemonstrationRecorder(tmpdir, obs_dim=8, action_dim=3)
        for _ in range(3):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(3).astype(np.float32)
            recorder.record_step(obs, action, done=True, skill_id=0, step_id=0, success=True)
            recorder.end_episode()
        npz_path, _ = recorder.save("bc_compat.npz", task="stacking", n_blocks=3)
        data = np.load(npz_path)
        # BC expects (T, obs_dim) and (T, action_dim)
        assert data["obs"].ndim == 2
        assert data["action"].ndim == 2
        assert data["obs"].shape[1] == 8
        assert data["action"].shape[1] == 3
