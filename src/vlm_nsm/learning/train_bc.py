"""Train MLP Behavior Cloning policy on robosuite demonstration trajectories.

Usage (single npz):
    python3 -m vlm_nsm.learning.train_bc \\
        --npz generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz \\
        --epochs 200 --batch-size 64 --lr 1e-3 \\
        --save generated/models/stage_7B-1/bc_mlp.pt

Usage (directory of per-episode npz files):
    python3 -m vlm_nsm.learning.train_bc \\
        --demo_dir generated/demos/stage_7A-12/all_semantic_success \\
        --obs_key filtered_obs --hidden_dims 256 256 --episode_split \\
        --epochs 200 --batch-size 64 --lr 1e-3 --device cuda \\
        --save generated/models/stage_7B-1/bc_mlp.pt

Runs on CPU by default; add ``--device cuda`` for GPU training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.dataset import RobosuiteDemoDataset


class MLPBCPolicy(nn.Module):
    """Simple MLP policy for behavior cloning:  obs -> action.

    Supports variable hidden layer configuration via ``hidden_dims`` list.
    """

    def __init__(self, obs_dim: int = 11, action_dim: int = 7, hidden_dims: list[int] | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        pred = model(obs)
        loss = loss_fn(pred, action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    """Validate, return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        pred = model(obs)
        loss = loss_fn(pred, action)
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def _compute_dataset_stats(train_ds, val_ds) -> dict:
    """Compute dataset statistics summary."""
    return {
        "train_episodes": train_ds.num_episodes,
        "train_steps": len(train_ds),
        "val_episodes": val_ds.num_episodes,
        "val_steps": len(val_ds),
        "obs_dim": train_ds.obs_dim,
        "action_dim": train_ds.action_dim,
        "action_mean": train_ds.action_mean.tolist() if train_ds.action_mean is not None else None,
        "action_std": train_ds.action_std.tolist() if train_ds.action_std is not None else None,
    }


def _create_masked_mse_loss(mask_dims: list[int], action_dim: int) -> nn.Module:
    """Create MSE loss that only applies to specified action dims."""
    mask = torch.ones(action_dim, dtype=torch.bool)
    for d in mask_dims:
        if 0 <= d < action_dim:
            mask[d] = False

    class MaskedMSELoss(nn.Module):
        def __init__(self, mask: torch.Tensor):
            super().__init__()
            self.register_buffer("mask", mask)

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            diff = pred - target
            masked_diff = diff[:, self.mask]
            return torch.mean(masked_diff ** 2)

    return MaskedMSELoss(mask)


def main():
    parser = argparse.ArgumentParser(description="Train MLP BC policy")
    # Data source (mutually exclusive)
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to demonstration npz file (single file with episode_id)")
    parser.add_argument("--demo_dir", type=str, default=None,
                        help="Directory of per-episode npz files")
    parser.add_argument("--obs_key", type=str, default="obs",
                        help="Key to use for observations in npz files (default: 'obs')")
    parser.add_argument("--episode_split", action="store_true", default=True,
                        help="Split episodes by episode_id in single npz (default: True)")
    parser.add_argument("--no-episode-split", action="store_false", dest="episode_split",
                        help="Treat all steps independently (no episode boundary detection)")
    # Stage 7B-2: phase conditioning
    parser.add_argument("--phase_key", type=str, default="phase_id",
                        help="Key for phase_id in npz (default: 'phase_id')")
    parser.add_argument("--condition_on_phase", action="store_true", default=False,
                        help="Append phase one-hot encoding to observation")
    # Stage 7B-3: phase filter (train on specific phase only)
    parser.add_argument("--phase-filter", type=str, default=None,
                        help="Phase name to filter training data (e.g. 'reach_pick', 'reach_place')")
    # Stage 7B-2: action normalization
    parser.add_argument("--min_action_std", type=float, default=1e-3,
                        help="Floor for action std to avoid numerical instability (default: 1e-3)")
    parser.add_argument("--fixed_zero_action_dims", type=int, nargs="*", default=None,
                        help="Action dims known to be always zero (e.g. 3 4 5)")
    parser.add_argument("--mask_near_zero_action_dims", action="store_true", default=False,
                        help="Exclude near-zero action dims from MSE loss")
    # Training params
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="MLP hidden layer widths (e.g. 256 256 for two hidden layers)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save model checkpoint .pt")
    parser.add_argument("--save-state-dict-only", action="store_true", default=True,
                        help="Save only state_dict (default)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    # Validate data source
    if args.npz is None and args.demo_dir is None:
        parser.error("Either --npz or --demo_dir must be provided")

    # Resolve fixed_zero_action_dims
    fixed_zero_dims = args.fixed_zero_action_dims
    if fixed_zero_dims is not None:
        print(f"Fixed zero action dims: {fixed_zero_dims}")

    # Load data (Stage 7B-2: pass condition_on_phase, phase_key, min_action_std)
    if args.demo_dir is not None:
        print(f"Loading demonstrations from directory {args.demo_dir}")
        train_ds, val_ds = RobosuiteDemoDataset.from_directory(
            args.demo_dir, val_ratio=args.val_ratio, seed=args.seed, obs_key=args.obs_key,
            condition_on_phase=args.condition_on_phase, phase_key=args.phase_key,
            min_action_std=args.min_action_std,
        )
    else:
        print(f"Loading demonstrations from {args.npz} (obs_key={args.obs_key}, episode_split={args.episode_split})")
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            args.npz, val_ratio=args.val_ratio, seed=args.seed,
            obs_key=args.obs_key, episode_split=args.episode_split,
            condition_on_phase=args.condition_on_phase, phase_key=args.phase_key,
            min_action_std=args.min_action_std,
        )

    print(f"Train episodes: {train_ds.num_episodes}, steps: {len(train_ds)}")
    print(f"Val episodes:   {val_ds.num_episodes}, steps: {len(val_ds)}")

    # Stage 7B-3: optional phase filter
    PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
    phase_filter_id: int | None = None
    if args.phase_filter is not None:
        if args.phase_filter not in PHASE_NAMES:
            parser.error(f"--phase-filter must be one of {PHASE_NAMES}, got '{args.phase_filter}'")
        phase_filter_id = PHASE_NAMES.index(args.phase_filter)
        # Filter flat indices by phase_id
        if train_ds.phase_id is not None:
            train_mask = train_ds.phase_id[train_ds._flat_indices] == phase_filter_id
            train_ds._flat_indices = [idx for idx, m in zip(train_ds._flat_indices, train_mask) if m]
            print(f"Phase filter '{args.phase_filter}' (id={phase_filter_id}): "
                  f"train steps {len(train_ds)}")
        if val_ds.phase_id is not None:
            val_mask = val_ds.phase_id[val_ds._flat_indices] == phase_filter_id
            val_ds._flat_indices = [idx for idx, m in zip(val_ds._flat_indices, val_mask) if m]
            print(f"Phase filter '{args.phase_filter}' (id={phase_filter_id}): "
                  f"val steps {len(val_ds)}")

    # Save dataset stats
    dataset_stats = _compute_dataset_stats(train_ds, val_ds)
    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results" / "stage_7B-1")
    stats_path = results_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(dataset_stats, indent=2, default=str), encoding="utf-8")
    print(f"Dataset stats saved to {stats_path}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model (Stage 7B-2: use obs_dim_with_phase if conditioning)
    obs_dim = train_ds.obs_dim_with_phase if args.condition_on_phase else train_ds.obs_dim
    model = MLPBCPolicy(obs_dim=obs_dim, action_dim=train_ds.action_dim,
                        hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Stage 7B-2: optional masked MSE loss for near-zero action dims
    if args.mask_near_zero_action_dims and train_ds.near_zero_std_dims:
        mask_dims = train_ds.near_zero_std_dims
        if fixed_zero_dims is not None:
            mask_dims = sorted(set(mask_dims) | set(fixed_zero_dims))
        print(f"Masking MSE loss for dims: {mask_dims}")
        loss_fn = _create_masked_mse_loss(mask_dims, train_ds.action_dim)
    else:
        loss_fn = nn.MSELoss()

    # Print model and normalization stats
    n_params = sum(p.numel() for p in model.parameters())
    hd_str = "->".join(str(h) for h in args.hidden_dims)
    print(f"Model: MLP({obs_dim}->{hd_str}->{train_ds.action_dim})"
          f"{' [phase-conditioned]' if args.condition_on_phase else ''}")
    print(f"Parameters: {n_params:,}")
    if train_ds.action_mean is not None:
        print(f"Action mean: {train_ds.action_mean}")
        print(f"Action std (used):  {train_ds.action_std}")
        if train_ds.action_std_raw is not None:
            print(f"Action std (raw):  {train_ds.action_std_raw}")
        if train_ds.near_zero_std_dims:
            print(f"Near-zero std dims (raw std < {args.min_action_std}): {train_ds.near_zero_std_dims}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Save (Stage 7B-2: include metadata for eval loading)
    if args.save:
        save_path = ensure_dir(Path(args.save).parent) / Path(args.save).name
        if args.save_state_dict_only:
            torch.save(best_state, save_path)
        else:
            torch.save({
                "model_state_dict": best_state,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": obs_dim,
                "action_dim": train_ds.action_dim,
                "hidden_dims": args.hidden_dims,
                "condition_on_phase": args.condition_on_phase,
                "phase_dim": train_ds._n_phases if args.condition_on_phase else 0,
                "obs_dim_raw": train_ds.obs_dim,
            }, save_path)
        print(f"Model saved to {save_path}")

    # Summary (Stage 7B-2: added phase conditioning and action norm fields)
    summary = {
        "data_source": args.demo_dir if args.demo_dir else args.npz,
        "obs_key": args.obs_key,
        "episode_split": args.episode_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dims": args.hidden_dims,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "device": str(device),
        "n_params": n_params,
        "train_episodes": train_ds.num_episodes,
        "train_steps": len(train_ds),
        "val_episodes": val_ds.num_episodes,
        "val_steps": len(val_ds),
        "train_loss": train_loss,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        # Stage 7B-2
        "condition_on_phase": args.condition_on_phase,
        "phase_key": args.phase_key if args.condition_on_phase else None,
        "obs_dim_raw": train_ds.obs_dim,
        "obs_dim_with_phase": train_ds.obs_dim_with_phase if args.condition_on_phase else None,
        "min_action_std": args.min_action_std,
        "near_zero_std_dims": train_ds.near_zero_std_dims,
        "fixed_zero_action_dims": fixed_zero_dims,
        "mask_near_zero_action_dims": args.mask_near_zero_action_dims,
        # Stage 7B-3
        "phase_filter": args.phase_filter,
        "phase_filter_id": phase_filter_id,
    }

    summary_path = results_dir / "train_bc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Training summary saved to {summary_path}")

    # Save action normalization stats separately for eval_bc (Stage 7B-2: extended)
    if train_ds.action_mean is not None and train_ds.action_std is not None:
        action_norm: dict = {
            "action_mean": train_ds.action_mean.tolist(),
            "action_std": train_ds.action_std.tolist(),
            "obs_dim": train_ds.obs_dim,
            "action_dim": train_ds.action_dim,
            "min_action_std": args.min_action_std,
            "near_zero_std_dims": train_ds.near_zero_std_dims,
            "fixed_zero_action_dims": fixed_zero_dims,
        }
        if train_ds.action_std_raw is not None:
            action_norm["action_std_raw"] = train_ds.action_std_raw.tolist()
        norm_path = results_dir / "action_norm_stats.json"
        norm_path.write_text(json.dumps(action_norm, indent=2), encoding="utf-8")
        print(f"Action norm stats saved to {norm_path}")


if __name__ == "__main__":
    main()
