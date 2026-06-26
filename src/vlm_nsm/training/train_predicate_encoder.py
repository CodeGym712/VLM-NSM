"""Train Learned Predicate Encoder to replace hand-crafted 31-dim one-hot.

Extracts object relationships from HDF5 data and trains a small network
to output dense predicate embeddings (64-dim) from object poses + phase.

Replaces: symbolic_condition.py hand-crafted one-hot encoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PredicateEncoder(nn.Module):
    """Learned dense embedding for symbolic predicates.

    Input:  (pick_pose(3), target_pose(3), pick_size(1), target_size(1), phase_id(1))
    Output: (embed_dim,) + (n_relations,) logits for auxiliary relation classification
    """

    def __init__(self, embed_dim=64, n_relations=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_relations = n_relations

        self.net = nn.Sequential(
            nn.Linear(13, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, embed_dim + n_relations),
        )

    def forward(self, x):
        out = self.net(x)
        embed = out[:, :self.embed_dim]
        rel_logits = out[:, self.embed_dim:]
        return embed, rel_logits


def build_predicate_dataset(dataset_dir, max_episodes=100):
    """Extract predicate training samples with episode-level grouping.

    Fixes C1.4:
      - Episode-level train/val split (not sample-level)
      - Real object features (ID one-hot + sizes, not EEF proxy + hardcoded)
    """
    root = Path(dataset_dir)
    with open(root / "manifest.jsonl") as f:
        episodes = [json.loads(l) for l in f.readlines()]

    success_eps = [e for e in episodes if e.get("success")][:max_episodes]
    print(f"[PredData] {len(success_eps)} episodes")

    # Object size registry (known from 2-block task)
    OBJ_SIZES = {"table": 0.0, "cube1": 0.04, "cube2": 0.05, "cube3": 0.02}
    N_OBJ = 4  # table, cube1, cube2, cube3

    # Per-episode storage for episode-level split
    ep_data = {}  # ep_id → {"X": [], "Y_rel": [], "Y_phase": []}

    for ep in success_eps:
        ep_id = ep.get("episode_id", ep.get("episode_path", ""))
        X_ep, Yr_ep, Yp_ep = [], [], []

        with h5py.File(ep["episode_path"], "r") as h5:
            eef = h5["proprio/eef_pos"][:]
            pick_ids = h5["symbolic/pick_object_id"][:]
            target_ids = h5["symbolic/target_object_id"][:]
            rel_ids = h5["symbolic/relation_id"][:]
            phase_ids = h5["symbolic/phase_id"][:]
            n = eef.shape[0]

            for t in range(n):
                pid = int(pick_ids[t])
                tid = int(target_ids[t])
                # Object presence one-hot (table=0, cube1=1, cube2=2, cube3=3)
                pick_oh = np.zeros(N_OBJ, dtype=np.float32)
                target_oh = np.zeros(N_OBJ, dtype=np.float32)
                if 0 <= pid < N_OBJ:
                    pick_oh[pid] = 1.0
                if 0 <= tid < N_OBJ:
                    target_oh[tid] = 1.0

                # Sizes from registry
                pick_sz = OBJ_SIZES.get({0:"table",1:"cube1",2:"cube2",3:"cube3"}.get(pid,""), 0.04)
                target_sz = OBJ_SIZES.get({0:"table",1:"cube1",2:"cube2",3:"cube3"}.get(tid,""), 0.04)

                x = np.concatenate([
                    pick_oh,           # 4-dim: which object is being picked
                    target_oh,         # 4-dim: which object is the target
                    [pick_sz, target_sz],  # 2-dim: object sizes
                    eef[t],            # 3-dim: EEF position (context)
                ]).astype(np.float32)  # total: 13-dim

                X_ep.append(x)
                Yr_ep.append(int(rel_ids[t]))
                Yp_ep.append(int(phase_ids[t]))

        ep_data[ep_id] = {"X": np.stack(X_ep), "Y_rel": np.array(Yr_ep, dtype=np.int64),
                          "Y_phase": np.array(Yp_ep, dtype=np.int64), "n": len(X_ep)}

    # Episode-level split
    ep_ids = list(ep_data.keys())
    n_ep = len(ep_ids)
    n_train_ep = max(1, int(n_ep * 0.8))
    rng = np.random.RandomState(42)
    rng.shuffle(ep_ids)
    train_eps = set(ep_ids[:n_train_ep])
    val_eps = set(ep_ids[n_train_ep:])

    train_X, train_Yr, train_Yp = [], [], []
    val_X, val_Yr, val_Yp = [], [], []

    for ep_id, d in ep_data.items():
        if ep_id in train_eps:
            train_X.append(d["X"]); train_Yr.append(d["Y_rel"]); train_Yp.append(d["Y_phase"])
        else:
            val_X.append(d["X"]); val_Yr.append(d["Y_rel"]); val_Yp.append(d["Y_phase"])

    # Concatenate
    train_X = np.concatenate(train_X) if train_X else np.zeros((0, 13), dtype=np.float32)
    train_Yr = np.concatenate(train_Yr) if train_Yr else np.array([], dtype=np.int64)
    val_X = np.concatenate(val_X) if val_X else np.zeros((0, 13), dtype=np.float32)
    val_Yr = np.concatenate(val_Yr) if val_Yr else np.array([], dtype=np.int64)

    print(f"[PredData] train: {len(train_X)} samples ({len(train_eps)} eps), "
          f"val: {len(val_X)} samples ({len(val_eps)} eps)")

    # Normalize (fit on train only)
    X_mean = train_X.mean(axis=0)
    X_std = train_X.std(axis=0).clip(1e-6)
    train_X = (train_X - X_mean) / X_std
    val_X = (val_X - X_mean) / X_std

    return train_X, train_Yr, val_X, val_Yr, X_mean, X_std, 13


def train_predicate_encoder(dataset_dir, output_dir, epochs=50, batch_size=512, lr=1e-3, device="cuda:0"):
    """Build dataset (episode-level split) and train predicate encoder."""
    train_X, train_Yr, val_X, val_Yr, X_mean, X_std, input_dim = \
        build_predicate_dataset(dataset_dir)

    train_ds = TensorDataset(torch.from_numpy(train_X).float(),
                             torch.from_numpy(train_Yr))
    val_ds = TensorDataset(torch.from_numpy(val_X).float(),
                           torch.from_numpy(val_Yr))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = PredicateEncoder(embed_dim=64, n_relations=3).to(device)
    print(f"[PredTrain] Model: {sum(p.numel() for p in model.parameters()):,} params")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            _, rel_logits = model(xb)
            loss = ce_loss(rel_logits, yb)
            loss.backward(); opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                _, rel_logits = model(xb)
                val_loss += ce_loss(rel_logits, yb).item() * xb.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val:
            best_val = val_loss
        if epoch % 10 == 0 or epoch == 1:
            print(f"[PredTrain] epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": model.embed_dim, "n_relations": model.n_relations,
        "X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
        "best_val_loss": best_val,
    }, out / "best.pt")
    with open(out / "meta.json", "w") as f:
        json.dump({"best_val_loss": best_val, "input_dim": input_dim,
                   "n_train": len(train_X), "n_val": len(val_X)}, f, indent=2)
    print(f"[PredTrain] Saved to {out / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="generated/datasets/vla_demos/d1_small_200ep")
    parser.add_argument("--output_dir", default="generated/results/predicate/encoder_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train_predicate_encoder(args.dataset_dir, args.output_dir,
                            args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
