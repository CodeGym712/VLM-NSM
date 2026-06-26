"""Train M1: Diffusion Policy for VLA action generation.

M1 uses Diffusion Policy (ResNet18 encoder + Conditional 1D UNet) to
generate action chunks via denoising diffusion.

Training: predict noise added to action chunk → MSE loss
Inference: DDIM sampling from random noise → action chunk
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

from vlm_nsm.vla.models.diffusion_policy import DiffusionPolicy
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

    # Datasets — M1 uses action chunks
    train_ds = VLADataset(
        args.dataset_dir, split="train",
        obs_history_len=2,           # 2 frames for velocity info
        action_chunk_len=args.horizon,
        action_horizon=args.horizon,
        stride=args.stride,
        image_size=args.image_size,
        success_only=True,
        cache_in_memory=True,
        operator_filter=args.operator_filter,
        use_real_symbolic=not args.zero_symbolic,
        symbolic_shuffle_seed=args.symbolic_shuffle_seed,
        zero_phase=args.zero_phase,
    )
    val_ds = VLADataset(
        args.dataset_dir, split="val",
        obs_history_len=2,
        action_chunk_len=args.horizon,
        action_horizon=args.horizon,
        stride=args.stride * 2,      # sparser val sampling
        image_size=args.image_size,
        success_only=True,
        cache_in_memory=True,
        operator_filter=args.operator_filter,
        use_real_symbolic=not args.zero_symbolic,
        symbolic_shuffle_seed=args.symbolic_shuffle_seed,
        zero_phase=args.zero_phase,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Model
    model = DiffusionPolicy(
        backbone=args.backbone,
        pretrained=args.pretrained,
        freeze_bn=not args.train_bn,
        vision_output_dim=args.vision_dim,
        symbolic_dim=train_ds.symbolic_dim,
        action_dim=4,
        horizon=args.horizon,
        unet_dims=tuple(args.unet_dims),
        num_diffusion_steps=args.diffusion_steps,
        num_inference_steps=args.inference_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, cond_dim={model.cond_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # EMA for inference stability
    ema_decay = args.ema_decay
    ema_model = None
    if ema_decay > 0:
        ema_model = DiffusionPolicy(
            backbone=args.backbone, pretrained=False, freeze_bn=True,
            vision_output_dim=args.vision_dim, symbolic_dim=train_ds.symbolic_dim,
            action_dim=4, horizon=args.horizon, unet_dims=tuple(args.unet_dims),
            num_diffusion_steps=args.diffusion_steps,
            num_inference_steps=args.inference_steps,
        ).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False

    config = {
        "model": "DiffusionPolicy",
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "vision_dim": args.vision_dim,
        "horizon": args.horizon,
        "unet_dims": list(args.unet_dims),
        "diffusion_steps": args.diffusion_steps,
        "inference_steps": args.inference_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "dataset_dir": str(args.dataset_dir),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "image_size": args.image_size,
        "action_dim": 4,
        "symbolic_dim": train_ds.symbolic_dim,
        "cond_dim": model.cond_dim,
        "ema_decay": ema_decay,
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
            action = batch["action_chunk"].to(device)

            # Diffusion: predict noise
            noise_pred, noise = model(av, fv, prop, sym, action)
            loss = criterion(noise_pred, noise)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA
            if ema_model is not None:
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            train_losses.append(float(loss.detach().cpu().item()))

        scheduler.step()
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")

        # Validation (predict noise, not sample)
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                av = batch["agentview"].to(device)
                fv = batch["frontview"].to(device)
                prop = batch["proprio"].to(device)
                sym = batch["symbolic_cond"].to(device)
                action = batch["action_chunk"].to(device)

                noise_pred, noise = model(av, fv, prop, sym, action)
                loss = criterion(noise_pred, noise)
                val_losses.append(float(loss.detach().cpu().item()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        loss_curve.append(row)

        # Save
        state = {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model else None,
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(state, output / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(state, output / "best.pt")

        if epoch % args.log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    metrics = {
        "model": "DiffusionPolicy",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "params": n_params,
    }
    _write_json(output / "metrics.json", metrics)

    print(f"\nBest: epoch={best_epoch}, val_loss={best_val:.6f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train M1 Diffusion Policy")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--train_bn", action="store_true")
    parser.add_argument("--vision_dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--unet_dims", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--inference_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--operator_filter", default=None, choices=["Pick", "PlaceOn"],
                        help="Per-operator data split (Phase 3.0a)")
    parser.add_argument("--zero_symbolic", action="store_true",
                        help="Train with all-zero symbolic condition (B4 baseline)")
    parser.add_argument("--symbolic_shuffle_seed", type=int, default=0,
                        help="If >0, apply fixed permutation to real symbolic dims (B4.6 ablation)")
    parser.add_argument("--zero_phase", action="store_true",
                        help="Zero out phase_onehot portion of symbolic vector (ablation)")
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
