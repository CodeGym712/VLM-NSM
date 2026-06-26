import json

import numpy as np
import torch

from vlm_nsm.learning.train_act_skill_policy import train_act_skill_policy


def _write_fake_dataset(root, n_steps=40):
    root.mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir()
    records = []
    for ep in range(2):
        record = {
            "episode_id": f"ep_{ep+1:06d}",
            "skill_name": "pick_place",
            "object_id": "cube1",
            "target_object_id": "cube2",
            "relation": "on",
            "source": "table",
            "success": True,
            "num_steps": n_steps,
            "failure_type": None,
            "trajectory_path": f"trajectories/ep_{ep+1:06d}.npz",
            "seed": ep,
        }
        records.append(record)
        obs = np.random.RandomState(ep).randn(n_steps, 11).astype(np.float32)
        action = np.random.RandomState(ep + 10).randn(n_steps, 7).astype(np.float32)
        np.savez_compressed(
            root / record["trajectory_path"],
            obs=obs,
            action=action,
            phase_id=np.zeros((n_steps,), dtype=np.int32),
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def test_train_act_skill_policy_smoke(tmp_path):
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "model"
    _write_fake_dataset(dataset_dir)

    config = train_act_skill_policy(
        dataset=str(dataset_dir),
        output_dir=str(output_dir),
        obs_history=4,
        action_chunk=3,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        device="cpu",
        hidden_dims=[32],
        embedding_dim=4,
        seed=0,
    )

    assert (output_dir / "best.pt").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "train_log.json").exists()
    assert (output_dir / "normalization_stats.json").exists()
    assert config["train_samples"] > 0
    assert config["val_samples"] > 0

    ckpt = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["model_config"]["obs_history_len"] == 4
