"""Train Learned Inverse Dynamics model from HDF5 data.

Usage:
    # Build dataset and train
    python -m vlm_nsm.training.train_inverse_dynamics \
        --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
        --output_dir generated/results/id/id_model_v1 \
        --epochs 50 --batch_size 256 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vlm_nsm.data.id_training_dataset import build_id_dataset


def train_id_model(
    dataset_dir: str,
    output_dir: str,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda:0",
    max_episodes: int | None = None,
):
    """Build dataset and train LearnedIDModel."""
    # Build dataset
    print(f"[ID train] Building dataset from {dataset_dir}...")
    data = build_id_dataset(dataset_dir, max_episodes=max_episodes)

    train_X = torch.from_numpy(data["train_X"])
    train_Y = torch.from_numpy(data["train_Y"])
    val_X = torch.from_numpy(data["val_X"])
    val_Y = torch.from_numpy(data["val_Y"])

    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"[ID train] {len(train_ds)} train, {len(val_ds)} val samples")

    # Create model
    from vlm_nsm.embodied.controllers.learned_id_model import LearnedIDModel

    model = LearnedIDModel(input_dim=data["input_dim"], output_dim=data["output_dim"])
    model.to(device)
    print(f"[ID train] Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6)})

        if epoch % 5 == 0 or epoch == 1 or is_best:
            marker = " *" if is_best else ""
            print(f"[ID train] epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f}{marker}")

    # Save model
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": data["input_dim"],
        "output_dim": data["output_dim"],
        "action_mean": data["stats"]["action_mean"],
        "action_std": data["stats"]["action_std"],
        "best_val_loss": best_val_loss,
        "history": history,
    }
    torch.save(ckpt, out / "best.pt")

    # Save metadata
    meta = {
        "input_dim": data["input_dim"],
        "output_dim": data["output_dim"],
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "best_val_loss": best_val_loss,
        "action_mean": data["stats"]["action_mean"],
        "action_std": data["stats"]["action_std"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save training history
    with open(out / "history.jsonl", "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")

    print(f"[ID train] Best val_loss={best_val_loss:.6f}, saved to {out}")
    return model, data


def main():
    parser = argparse.ArgumentParser(description="Train Learned Inverse Dynamics model")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to VLA HDF5 dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes (for quick testing)")
    args = parser.parse_args()

    train_id_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
