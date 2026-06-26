import json

import numpy as np

from vlm_nsm.learning.skill_dataset import (
    SkillDataset,
    compute_normalization_stats,
    load_manifest,
)


def _write_fake_skill_dataset(root, *, success=True, n_steps=30, phase_pattern=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir()
    record = {
        "episode_id": "ep_000001",
        "skill_name": "pick_place",
        "object_id": "cube1",
        "target_object_id": "cube2",
        "relation": "on",
        "source": "table",
        "success": success,
        "num_steps": n_steps,
        "failure_type": None if success else "horizon",
        "trajectory_path": "trajectories/ep_000001.npz",
        "seed": 0,
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    phase_id = np.asarray(
        phase_pattern if phase_pattern is not None else [0] * n_steps,
        dtype=np.int32,
    )
    np.savez_compressed(
        root / "trajectories" / "ep_000001.npz",
        obs=np.arange(n_steps * 11, dtype=np.float32).reshape(n_steps, 11),
        action=np.arange(n_steps * 7, dtype=np.float32).reshape(n_steps, 7),
        phase_id=phase_id,
        skill_name=np.array(["pick_place"] * n_steps),
        skill_object_id=np.array(["cube1"] * n_steps),
        skill_target_object_id=np.array(["cube2"] * n_steps),
    )
    return record


def test_load_manifest(tmp_path):
    record = _write_fake_skill_dataset(tmp_path)

    records = load_manifest(tmp_path)

    assert records == [record]


def test_skill_dataset_shapes(tmp_path):
    _write_fake_skill_dataset(tmp_path, n_steps=30)

    dataset = SkillDataset(tmp_path, obs_history_len=4, action_chunk_len=6)
    item = dataset[0]

    assert len(dataset) == 21
    assert dataset.num_trajectories == 1
    assert item["obs_history"].shape == (4, 11)
    assert item["action_chunk"].shape == (6, 7)
    assert item["goal_condition"].tolist() == [0, 2, 3]
    assert item["object_id"] == "cube1"
    assert item["target_object_id"] == "cube2"


def test_skill_dataset_excludes_failures_by_default(tmp_path):
    _write_fake_skill_dataset(tmp_path, success=False, n_steps=30)

    dataset = SkillDataset(tmp_path, obs_history_len=4, action_chunk_len=6)
    dataset_with_failures = SkillDataset(
        tmp_path,
        obs_history_len=4,
        action_chunk_len=6,
        include_failures=True,
    )

    assert len(dataset) == 0
    assert len(dataset_with_failures) > 0


def test_skill_dataset_phase_filter(tmp_path):
    phase_pattern = [0] * 10 + [1] * 10 + [0] * 10
    _write_fake_skill_dataset(tmp_path, n_steps=30, phase_pattern=phase_pattern)

    dataset = SkillDataset(
        tmp_path,
        obs_history_len=4,
        action_chunk_len=3,
        phase_filter=[1],
    )

    assert len(dataset) > 0
    for i in range(len(dataset)):
        assert set(dataset[i]["phase_id"].tolist()) == {1}


def test_skill_dataset_skill_filter(tmp_path):
    _write_fake_skill_dataset(tmp_path, n_steps=30)

    keep = SkillDataset(tmp_path, obs_history_len=4, action_chunk_len=6, skill_filter=["pick_place"])
    drop = SkillDataset(tmp_path, obs_history_len=4, action_chunk_len=6, skill_filter=["place"])

    assert len(keep) > 0
    assert len(drop) == 0


def test_compute_normalization_stats(tmp_path):
    _write_fake_skill_dataset(tmp_path, n_steps=30)

    stats = compute_normalization_stats(tmp_path)

    assert len(stats["obs_mean"]) == 11
    assert len(stats["obs_std"]) == 11
    assert len(stats["action_mean"]) == 7
    assert len(stats["action_std"]) == 7
    assert min(stats["obs_std"]) > 0


def test_skill_dataset_normalize(tmp_path):
    _write_fake_skill_dataset(tmp_path, n_steps=30)
    stats = compute_normalization_stats(tmp_path)

    dataset = SkillDataset(
        tmp_path,
        obs_history_len=4,
        action_chunk_len=6,
        normalize=True,
        normalization_stats=stats,
    )

    item = dataset[0]
    assert item["obs_history"].dtype == np.float32
    assert item["action_chunk"].dtype == np.float32
