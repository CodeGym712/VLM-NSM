"""Train M0: ResNet18 + MLP BC baseline for VLA pipeline validation.

This is the simplest VLA training script — purely for verifying:
  1. HDF5 data loading works
  2. RGB → action prediction learns
  3. Action scale is correct
  4. Overfitting on small data is possible

Not intended for production VLA training (use M1 Diffusion for that).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from vlm_nsm.vla.models.vision_encoder import VLABCModel
from vlm_nsm.vla.data.vla_dataset import VLADataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Datasets
    train_ds = VLADataset(
        args.dataset_dir, split="train",
        obs_history_len=1,  # M0 uses single-frame
        action_chunk_len=1,  # M0 predicts single action
        stride=args.stride,
        image_size=args.image_size,
        success_only=True,
        cache_in_memory=True,
    )
    val_ds = VLADataset(
        args.dataset_dir, split="val",
        obs_history_len=1, action_chunk_len=1,
        stride=args.stride,
        image_size=args.image_size,
        success_only=True,
        cache_in_memory=True,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = VLABCModel(
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_bn=not args.train_bn,
        vision_output_dim=args.vision_dim,
        symbolic_dim=train_ds.symbolic_dim,
        hidden_dims=tuple(args.hidden_dims),
        action_dim=4,
        dropout=args.dropout,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Config
    config = {
        "model": "VLABCModel",
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "vision_dim": args.vision_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "dataset_dir": str(args.dataset_dir),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "image_size": args.image_size,
        "symbolic_dim": train_ds.symbolic_dim,
        "action_dim": 4,
    }
    _write_json(output / "config.json", config)

    best_val = float("inf")
    best_epoch = -1
    loss_curve: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            av = batch["agentview"].to(device)
            fv = batch["frontview"].to(device)
            prop = batch["proprio"].to(device)
            sym = batch["symbolic_cond"].to(device)
            target = batch["action_chunk"].to(device)
            if target.dim() == 3:
                target = target[:, 0, :]  # (B, chunk, 4) → (B, 4)

            pred = model(av, fv, prop, sym)
            loss = criterion(pred, target)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        # Validation
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                av = batch["agentview"].to(device)
                fv = batch["frontview"].to(device)
                prop = batch["proprio"].to(device)
                sym = batch["symbolic_cond"].to(device)
                target = batch["action_chunk"].to(device)
                if target.dim() == 3:
                    target = target[:, 0, :]

                pred = model(av, fv, prop, sym)
                loss = criterion(pred, target)
                val_losses.append(float(loss.detach().cpu().item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        loss_curve.append(row)

        # Save checkpoint
        state = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(state, output / "last.pt", _use_new_zipfile_serialization=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(state, output / "best.pt")

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # Final metrics
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "loss_curve": loss_curve,
    }
    _write_json(output / "metrics.json", metrics)
    (output / "loss_curve.jsonl").write_text(
        "\n".join(json.dumps(r) for r in loss_curve), encoding="utf-8")

    print(f"\nBest: epoch={best_epoch}, val_loss={best_val:.6f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train M0 VLA BC baseline")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--train_bn", action="store_true")
    parser.add_argument("--vision_dim", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train(args)
    print(json.dumps({
        "best_val_loss": metrics["best_val_loss"],
        "best_epoch": metrics["best_epoch"],
    }, indent=2))


if __name__ == "__main__":
    main()
