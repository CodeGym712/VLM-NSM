import json
from pathlib import Path

import numpy as np


def write_fake_stage20_dataset(root: Path, n: int = 12, history: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, history, 11)).astype(np.float32)
    role = rng.normal(size=(n, history, 20)).astype(np.float32)
    goal = rng.normal(size=(n, history, 16)).astype(np.float32)
    goal_mask = np.ones((n, history, 16), dtype=np.float32)
    reached = rng.normal(size=(n, history, 20)).astype(np.float32)
    reached_mask = np.ones((n, history, 20), dtype=np.float32)
    progress = rng.uniform(0, 1, size=(n, history)).astype(np.float32)
    phase = np.tile(np.arange(history), (n, 1)).astype(np.int64) % 4
    actions = rng.normal(size=(n, 7)).astype(np.float32)
    np.savez_compressed(
        root / "windows.npz",
        obs_history=obs,
        role_state_history=role,
        goal_delta_history=goal,
        goal_delta_mask=goal_mask,
        reached_goal_delta_history=reached,
        reached_goal_delta_mask=reached_mask,
        progress_scalar_history=progress,
        phase_id_history=phase,
        actions=actions,
    )
    records = []
    for i in range(n):
        records.append({
            "window_id": f"w_{i:08d}",
            "direction_id": i % 2,
            "phase_id": i % 4,
            "direction": "cube1_to_cube2" if i % 2 == 0 else "cube2_to_cube1",
            "split": "train" if i < n - 4 else "val",
        })
    for name, subset in [
        ("manifest.jsonl", records),
        ("train_manifest.jsonl", records[: n - 4]),
        ("val_manifest.jsonl", records[n - 4 :]),
    ]:
        with (root / name).open("w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record) + "\n")

    summary = {
        "num_windows": n,
        "train_windows": n - 4,
        "val_windows": 4,
        "obs_dim": 11,
        "action_dim": 7,
        "role_state_dim": 20,
        "goal_delta_dim": 16,
        "reached_goal_delta_dim": 20,
        "history_len": history,
        "can_train_mlp_baseline": True,
        "recommendation": "ready_for_training",
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def stats(arr):
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
        return {"mean": flat.mean(axis=0).tolist(), "std": np.maximum(flat.std(axis=0), 1e-6).tolist()}

    norm = {
        "obs_history": stats(obs),
        "role_state_history": stats(role),
        "goal_delta_history": stats(goal),
        "reached_goal_delta_history": stats(reached),
        "actions": stats(actions),
    }
    (root / "normalization_stats.json").write_text(json.dumps(norm), encoding="utf-8")
    return root
