import json

import numpy as np

from vlm_nsm.experiments.stage20_k_relabel_phase_trajectories import (
    build_markdown,
    relabel_dataset,
)


def _write_fake_teacher_dataset(root):
    traj_dir = root / "trajectories"
    traj_dir.mkdir(parents=True)
    phase = np.array([0, 3, 3, 3, 3], dtype=np.int32)
    action = np.zeros((5, 7), dtype=np.float32)
    action[:, 6] = 1.0
    action[2:, 6] = -1.0
    action[4, 2] = 0.05
    np.savez_compressed(
        traj_dir / "ep0.npz",
        obs=np.zeros((5, 11), dtype=np.float32),
        action=action,
        phase_id=phase,
    )
    record = {
        "episode_id": "ep0",
        "trajectory_path": "trajectories/ep0.npz",
        "object_id": "cube1",
        "target_object_id": "cube2",
        "teacher_profile": "cube2_robust",
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"num_records": 1}), encoding="utf-8")


def test_relabel_dataset_copies_and_adds_phase_4_5(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_fake_teacher_dataset(src)

    summary = relabel_dataset(src, dst)

    assert summary["changed_trajectories"] == 1
    assert summary["release_steps_added"] == 2
    assert summary["retreat_steps_added"] == 1
    data = np.load(dst / "trajectories" / "ep0.npz", allow_pickle=True)
    assert data["phase_id"].tolist() == [0, 3, 4, 4, 5]
    assert data["phase_id_raw"].tolist() == [0, 3, 3, 3, 3]
    assert data["phase_label_changed"].tolist() == [False, False, True, True, True]
    assert data["phase_name"].tolist()[-2:] == ["release", "retreat"]
    records = (dst / "manifest.jsonl").read_text(encoding="utf-8")
    assert "phase_label_patch" in records


def test_build_markdown_mentions_recommendation():
    text = build_markdown(
        {
            "source_dataset": "a",
            "output_dataset": "b",
            "num_records": 1,
            "changed_trajectories": 1,
            "changed_steps": 3,
            "release_steps_added": 2,
            "retreat_steps_added": 1,
            "phase_counts_before": {"3": 4},
            "phase_counts_after": {"3": 1, "4": 2, "5": 1},
        }
    )

    assert "Stage 20-K" in text
    assert "release" in text
    assert "Recommendation" in text
