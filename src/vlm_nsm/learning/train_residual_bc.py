"""Train residual BC model: predict residual = oracle_action - bc_action.

Loads the aggregated reach dataset and trains an MLP to predict
the residual action (delta from BC to oracle). During deployment:
  final_action = bc_action + residual_prediction

The residual is NOT normalized (already near zero), preserving its
physical scale.

Usage:
    python3 -m vlm_nsm.learning.train_residual_bc \\
        --npz generated/demos/stage_7B-4/dagger_reach_aggregated/aggregated_reach_dataset.npz \\
        --target_key residual_action \\
        --epochs 100 --batch-size 64 --lr 1e-3 --hidden_dims 256 256 \\
        --seed 42 --device cuda \\
        --save generated/results/stage_7B-4/residual_bc/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.train_bc import MLPBCPolicy


class AggregatedReachDataset(Dataset):
    """Load aggregated reach dataset npz for residual BC training.

    Unlike RobosuiteDemoDataset, this does NOT normalize actions —
    residuals are used in raw physical space.
    """

    def __init__(
        self,
        npz_path: str | Path,
        target_key: str = "residual_action",
        condition_on_phase: bool = True,
        phase_key: str = "phase_id",
    ):
        data = np.load(npz_path)
        self.obs = data["filtered_obs"].astype(np.float32)  # (T, 11)
        self.target = data[target_key].astype(np.float32)  # (T, 7)
        self.phase_id = data.get(phase_key, np.zeros(len(self.obs), dtype=np.int64))
        self.episode_id = data.get("episode_id", np.arange(len(self.obs), dtype=np.int32))
        self.condition_on_phase = condition_on_phase
        self._n_phases = 6
        self.obs_dim = self.obs.shape[-1]
        self.action_dim = self.target.shape[-1]

        # Episode-level split
        self._episodes = self._compute_episode_boundaries(self.episode_id)
        self._flat_indices = list(range(len(self.obs)))

    @staticmethod
    def _compute_episode_boundaries(episode_id):
        episodes = []
        if len(episode_id) == 0:
            return episodes
        current = 0
        for i in range(1, len(episode_id)):
            if episode_id[i] != episode_id[i - 1]:
                episodes.append((current, i))
                current = i
        episodes.append((current, len(episode_id)))
        return episodes

    def __len__(self):
        return len(self._flat_indices)

    def __getitem__(self, idx):
        global_step = self._flat_indices[idx]
        obs = torch.from_numpy(self.obs[global_step]).float()
        target = torch.from_numpy(self.target[global_step]).float()

        if self.condition_on_phase:
            phase = int(self.phase_id[global_step])
            oh = torch.zeros(self._n_phases, dtype=torch.float32)
            if 0 <= phase < self._n_phases:
                oh[phase] = 1.0
            obs = torch.cat([obs, oh])

        return {"obs": obs, "action": target}

    @classmethod
    def train_val(cls, npz_path, val_ratio=0.2, seed=42,
                  target_key="residual_action", condition_on_phase=True):
        """Create train/val datasets with episode-level split."""
        # Full dataset
        data = np.load(npz_path)
        episode_id = data["episode_id"]
        episodes = cls._compute_episode_boundaries(cls, episode_id)

        rng = np.random.RandomState(seed)
        n_eps = len(episodes)
        n_val = max(1, int(n_eps * val_ratio))
        val_indices = set(rng.choice(n_eps, n_val, replace=False).tolist())

        return _build_split_datasets(
            npz_path, episodes, val_indices,
            target_key=target_key, condition_on_phase=condition_on_phase,
        )


def _build_split_datasets(npz_path, episodes, val_indices, target_key, condition_on_phase):
    """Build train/val datasets from precomputed episode split."""
    data = np.load(npz_path)
    obs_all = data["filtered_obs"].astype(np.float32)
    target_all = data[target_key].astype(np.float32) if target_key != "action" else data["action"].astype(np.float32)
    phase_all = data.get("phase_id", np.zeros(len(obs_all), dtype=np.int64))
    ep_id_all = data.get("episode_id", np.arange(len(obs_all), dtype=np.int32))

    train_indices = sorted(set(range(len(episodes))) - val_indices)

    def _make_ds(ep_indices_list):
        ds = cls.__new__(cls)  # noqa
        ds.obs_dim = obs_all.shape[-1]
        ds.action_dim = target_all.shape[-1]
        ds.condition_on_phase = condition_on_phase
        ds._n_phases = 6
        ds._flat_indices = []
        ds._episodes = []
        offset = 0
        obs_parts, target_parts, phase_parts = [], [], []
        for ei in ep_indices_list:
            s, e = episodes[ei]
            obs_parts.append(obs_all[s:e])
            target_parts.append(target_all[s:e])
            phase_parts.append(phase_all[s:e])
            ds._episodes.append((offset, offset + (e - s)))
            offset += (e - s)
        ds.obs = np.concatenate(obs_parts)
        ds.target = np.concatenate(target_parts)
        ds.phase_id = np.concatenate(phase_parts)
        ds._flat_indices = list(range(len(ds.obs)))
        ds._episode_indices = list(range(len(ds._episodes)))
        return ds

    class _TempDataset(cls):
        pass

    # Use _TempDataset to avoid issues with cls being a classmethod holder
    train_ds = _build_single_ds(obs_all, target_all, phase_all,
                                [episodes[i] for i in train_indices],
                                condition_on_phase)
    val_ds = _build_single_ds(obs_all, target_all, phase_all,
                              [episodes[i] for i in val_indices],
                              condition_on_phase)
    return train_ds, val_ds


def _build_single_ds(obs_all, target_all, phase_all, episode_slices, condition_on_phase):
    """Build a single dataset from episode slices."""
    class _SimpleDS(Dataset):
        def __init__(self, obs, target, phase_id, condition_on_phase):
            self.obs = obs
            self.target = target
            self.phase_id = phase_id
            self.condition_on_phase = condition_on_phase
            self._n_phases = 6
            self.obs_dim = obs.shape[-1]
            self.action_dim = target.shape[-1]
            self._flat_indices = list(range(len(obs)))

        def __len__(self):
            return len(self._flat_indices)

        def __getitem__(self, idx):
            gi = self._flat_indices[idx]
            o = torch.from_numpy(self.obs[gi]).float()
            t = torch.from_numpy(self.target[gi]).float()
            if self.condition_on_phase:
                phase = int(self.phase_id[gi])
                oh = torch.zeros(self._n_phases, dtype=torch.float32)
                if 0 <= phase < self._n_phases:
                    oh[phase] = 1.0
                o = torch.cat([o, oh])
            return {"obs": o, "action": t}

    obs_parts, target_parts, phase_parts = [], [], []
    for s, e in episode_slices:
        obs_parts.append(obs_all[s:e])
        target_parts.append(target_all[s:e])
        phase_parts.append(phase_all[s:e])
    obs = np.concatenate(obs_parts)
    target = np.concatenate(target_parts)
    phase_id = np.concatenate(phase_parts)
    return _SimpleDS(obs, target, phase_id, condition_on_phase)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        target = batch["action"].to(device)
        pred = model(obs)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        target = batch["action"].to(device)
        pred = model(obs)
        loss = loss_fn(pred, target)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train residual BC policy")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--target_key", type=str, default="residual_action")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device}")
    print(f"Target key: {args.target_key}")

    # Load dataset
    data = np.load(args.npz)
    obs_all = data["filtered_obs"].astype(np.float32)
    target_all = data[args.target_key].astype(np.float32)
    episode_id = data["episode_id"]
    n_eps = len(np.unique(episode_id))

    # Episode-level split
    episode_boundaries = []
    cur = 0
    for i in range(1, len(episode_id)):
        if episode_id[i] != episode_id[i - 1]:
            episode_boundaries.append((cur, i))
            cur = i
    episode_boundaries.append((cur, len(episode_id)))

    rng = np.random.RandomState(args.seed)
    n_val = max(1, int(n_eps * args.val_ratio))
    val_ep_indices = set(rng.choice(n_eps, n_val, replace=False).tolist())

    train_ep_indices = sorted(set(range(n_eps)) - val_ep_indices)
    train_ds = _build_single_ds(
        obs_all, target_all, data.get("phase_id", np.zeros(len(obs_all), dtype=np.int64)),
        [episode_boundaries[i] for i in train_ep_indices],
        condition_on_phase=True,
    )
    val_ds = _build_single_ds(
        obs_all, target_all, data.get("phase_id", np.zeros(len(obs_all), dtype=np.int64)),
        [episode_boundaries[i] for i in val_ep_indices],
        condition_on_phase=True,
    )

    print(f"Train: {len(train_ds)} steps, Val: {len(val_ds)} steps")

    # Model
    obs_dim = train_ds.obs_dim + (6 if train_ds.condition_on_phase else 0)
    model = MLPBCPolicy(obs_dim=obs_dim, action_dim=train_ds.action_dim, hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    hd_str = "->".join(str(h) for h in args.hidden_dims)
    print(f"Model: MLP({obs_dim}->{hd_str}->{train_ds.action_dim}) [{n_params:,} params]")

    # Residual target statistics
    target_vals = target_all
    print(f"\nTarget ({args.target_key}) statistics:")
    print(f"  Mean: {np.mean(target_vals, axis=0)}")
    print(f"  Std:  {np.std(target_vals, axis=0)}")
    print(f"  Abs max: {np.max(np.abs(target_vals), axis=0)}")

    # Train
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nTraining for {args.epochs} epochs...")
    t_start = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        v_loss = validate(model, val_loader, loss_fn, device)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}: train_loss={t_loss:.8f}  val_loss={v_loss:.8f}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.8f}")

    # Save
    if args.save:
        save_path = ensure_dir(Path(args.save).parent) / Path(args.save).name
        torch.save({
            "model_state_dict": best_state,
            "train_loss": t_loss,
            "val_loss": best_val_loss,
            "obs_dim": obs_dim,
            "action_dim": train_ds.action_dim,
            "hidden_dims": args.hidden_dims,
            "target_key": args.target_key,
            "condition_on_phase": True,
            "obs_dim_raw": train_ds.obs_dim,
        }, save_path)
        print(f"Model saved to {save_path}")

    # Save summary
    summary = {
        "target_key": args.target_key,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dims": args.hidden_dims,
        "seed": args.seed,
        "n_params": n_params,
        "train_steps": len(train_ds),
        "val_steps": len(val_ds),
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
    }
    if args.save:
        summary_path = Path(args.save).parent / "train_residual_bc_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
