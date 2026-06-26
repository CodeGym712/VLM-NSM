"""Train Stage 20 goal-conditioned MLP action predictor."""

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

from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset


class GoalConditionedMLP(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def _mean_dict(values: dict[str, list[float]]) -> dict[str, float]:
    return {str(k): float(np.mean(v)) for k, v in sorted(values.items()) if v}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses: list[float] = []
    action_errors: list[np.ndarray] = []
    direction_losses: dict[str, list[float]] = {}
    phase_losses: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            y = batch["action"].to(device)
            pred = model(x)
            per_dim = criterion(pred, y)
            per_sample = per_dim.mean(dim=1)
            losses.extend(per_sample.detach().cpu().numpy().astype(float).tolist())
            action_errors.append(per_dim.detach().cpu().numpy())
            for label, loss in zip(batch["direction_label"].numpy().tolist(), per_sample.detach().cpu().numpy().tolist()):
                direction_losses.setdefault(str(int(label)), []).append(float(loss))
            for label, loss in zip(batch["phase_label"].numpy().tolist(), per_sample.detach().cpu().numpy().tolist()):
                phase_losses.setdefault(str(int(label)), []).append(float(loss))
    if not losses:
        return {
            "loss": float("inf"),
            "per_direction_loss": {},
            "per_phase_loss": {},
            "action_dim_mse": [],
        }
    action_arr = np.concatenate(action_errors, axis=0)
    return {
        "loss": float(np.mean(losses)),
        "per_direction_loss": _mean_dict(direction_losses),
        "per_phase_loss": _mean_dict(phase_losses),
        "action_dim_mse": action_arr.mean(axis=0).astype(float).tolist(),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_ds = Stage20RoleConditionedDataset(args.dataset_dir, split="train", feature_mode=args.feature_mode)
    val_ds = Stage20RoleConditionedDataset(args.dataset_dir, split="val", feature_mode=args.feature_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GoalConditionedMLP(
        input_dim=train_ds.input_dim,
        action_dim=train_ds.action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    config = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output),
        "feature_mode": args.feature_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seed": args.seed,
        "requested_device": args.device,
        "device": str(device),
        "input_dim": train_ds.input_dim,
        "action_dim": train_ds.action_dim,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    _write_json(output / "config.json", config)

    best_val = float("inf")
    best_epoch = -1
    loss_curve: list[dict[str, float]] = []
    log_path = output / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                x = batch["features"].to(device)
                y = batch["action"].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
            val_metrics = evaluate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
            }
            loss_curve.append(row)
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()

            state = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
            }
            torch.save(state, output / "last.pt")
            if val_metrics["loss"] < best_val:
                best_val = float(val_metrics["loss"])
                best_epoch = epoch
                torch.save(state, output / "best.pt")

    final_val = evaluate(model, val_loader, device)
    train_eval = evaluate(model, train_loader, device)
    metrics = {
        "feature_mode": args.feature_mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_loss": train_eval["loss"],
        "val_loss": final_val["loss"],
        "per_direction_val_loss": final_val["per_direction_loss"],
        "per_phase_val_loss": final_val["per_phase_loss"],
        "action_dim_mse": final_val["action_dim_mse"],
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "input_dim": train_ds.input_dim,
        "action_dim": train_ds.action_dim,
        "checkpoint_best": str(output / "best.pt"),
        "checkpoint_last": str(output / "last.pt"),
    }
    _write_json(output / "metrics.json", metrics)
    (output / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 20 goal-conditioned MLP")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_mode", choices=["obs_only", "goal_conditioned"], default="goal_conditioned")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
