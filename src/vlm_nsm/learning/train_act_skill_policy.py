"""Train ACT-lite skill policy on skill-labeled demonstrations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from vlm_nsm.learning.act_skill_policy import ACTSkillPolicy
from vlm_nsm.learning.skill_dataset import SkillDataset, compute_normalization_stats


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_history = batch["obs_history"].float().to(device)
    goal_condition = batch["goal_condition"].long().to(device)
    action_chunk = batch["action_chunk"].float().to(device)
    return obs_history, goal_condition, action_chunk


def train_epoch(model, loader, optimizer, loss_fn, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        obs_history, goal_condition, action_chunk = _batch_to_device(batch, device)
        pred = model(obs_history, goal_condition)
        loss = loss_fn(pred, action_chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        obs_history, goal_condition, action_chunk = _batch_to_device(batch, device)
        pred = model(obs_history, goal_condition)
        loss = loss_fn(pred, action_chunk)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_act_skill_policy(
    *,
    dataset: str,
    output_dir: str,
    obs_history: int = 10,
    action_chunk: int = 16,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    val_split: float = 0.2,
    seed: int = 42,
    hidden_dims: list[int] | None = None,
    embedding_dim: int = 16,
    phase_filter: list[int] | None = None,
    include_failures: bool = False,
    stride: int = 1,
) -> dict[str, Any]:
    """Train ACT-lite and save checkpoint/config/logs."""
    torch.manual_seed(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _device(device)

    stats = compute_normalization_stats(dataset, only_success=not include_failures)
    full_dataset = SkillDataset(
        dataset,
        obs_history_len=obs_history,
        action_chunk_len=action_chunk,
        normalize=True,
        normalization_stats=stats,
        phase_filter=phase_filter,
        include_failures=include_failures,
        stride=stride,
    )
    if len(full_dataset) == 0:
        raise ValueError("SkillDataset produced zero training samples")

    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = max(1, len(full_dataset) - n_val)
    if n_train + n_val > len(full_dataset):
        n_val = len(full_dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sample = full_dataset[0]
    obs_dim = int(sample["obs_history"].shape[-1])
    action_dim = int(sample["action_chunk"].shape[-1])
    model = ACTSkillPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_history_len=obs_history,
        action_chunk_len=action_chunk,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims or [256, 256],
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    log: list[dict[str, float | int]] = []
    best_val = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, dev)
        val_loss = validate(model, val_loader, loss_fn, dev) if n_val > 0 else train_loss
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(row)
        print(f"epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model.config(),
                    "normalization_stats": stats,
                    "dataset": dataset,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                out_dir / "best.pt",
            )

    config = {
        **model.config(),
        "dataset": dataset,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(dev),
        "val_split": val_split,
        "seed": seed,
        "phase_filter": phase_filter,
        "include_failures": include_failures,
        "stride": stride,
        "train_samples": n_train,
        "val_samples": n_val,
        "num_trajectories": full_dataset.num_trajectories,
        "best_val_loss": best_val,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (out_dir / "normalization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train ACT-lite skill policy")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--obs_history", type=int, default=10)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--phase_filter", type=int, nargs="*", default=None)
    parser.add_argument("--include_failures", action="store_true", default=False)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    train_act_skill_policy(
        dataset=args.dataset,
        output_dir=args.output_dir,
        obs_history=args.obs_history,
        action_chunk=args.action_chunk,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_split=args.val_split,
        seed=args.seed,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        phase_filter=args.phase_filter,
        include_failures=args.include_failures,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
