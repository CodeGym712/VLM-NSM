"""Train Stage 20 goal-conditioned recurrent BC action predictor.

DEPRECATED (2026-06-10): Stage 20 state-based BC路线已于6月初放弃。
闭环验证(Stage 20-Z DAgger Iter2) confirmed 0% stacking success。
当前路线: Phase 0-2.0 vision-based VLA + Diffusion Policy, 见 note/controller.md。
本文件保留仅作为 archive, 不再用于新实验。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset


ModelType = Literal["gru", "lstm"]
SamplerMode = Literal["none", "phase_balanced", "phase_direction_balanced"]
LossWeightingMode = Literal["none", "inverse_phase_freq", "manual"]


class GoalConditionedRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        model_type: ModelType = "gru",
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_gripper_head: bool = False,
        use_gripper_magnitude_head: bool = False,
        use_holding_head: bool = False,
    ):
        super().__init__()
        if model_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported model_type: {model_type}")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        ) if use_gripper_head else None
        self.gripper_magnitude_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        ) if use_gripper_magnitude_head else None
        # Stage 20-S: holding auxiliary head (binary classification)
        self.holding_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        ) if use_holding_head else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def predict_gripper_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.gripper_head is None:
            raise RuntimeError("gripper_head is disabled")
        return self.gripper_head(self.encode(x)).squeeze(-1)

    def predict_gripper_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        if self.gripper_magnitude_head is None:
            raise RuntimeError("gripper_magnitude_head is disabled")
        return self.gripper_magnitude_head(self.encode(x)).squeeze(-1)

    # Stage 20-S: holding prediction head
    def predict_holding_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.holding_head is None:
            raise RuntimeError("holding_head is disabled")
        return self.holding_head(self.encode(x)).squeeze(-1)


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
    direction_phase_losses: dict[str, list[float]] = {}
    direction_action_errors: dict[str, list[np.ndarray]] = {}
    gripper_correct: list[float] = []
    gripper_bce: list[float] = []
    gripper_phase_acc: dict[str, list[float]] = {}
    gripper_direction_acc: dict[str, list[float]] = {}
    has_gripper_head = bool(getattr(model, "gripper_head", None) is not None)
    gripper_magnitude_errors: list[float] = []
    gripper_magnitude_phase_errors: dict[str, list[float]] = {}
    gripper_magnitude_direction_errors: dict[str, list[float]] = {}
    gripper_magnitude_direction_phase_errors: dict[str, list[float]] = {}
    has_gripper_magnitude_head = bool(getattr(model, "gripper_magnitude_head", None) is not None)
    # Stage 20-S: holding head metrics
    has_holding_head = bool(getattr(model, "holding_head", None) is not None)
    holding_correct: list[float] = []
    holding_bce: list[float] = []
    holding_phase_acc: dict[str, list[float]] = {}
    holding_direction_acc: dict[str, list[float]] = {}
    holding_phase_bce: dict[str, list[float]] = {}
    holding_direction_bce: dict[str, list[float]] = {}
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
            per_dim_np = per_dim.detach().cpu().numpy()
            dir_labels = batch["direction_label"].numpy().tolist()
            phase_labels_np = batch["phase_label"].numpy().tolist()
            sample_losses_np = per_sample.detach().cpu().numpy().tolist()
            for row_idx, (direction_label, phase_label, loss) in enumerate(zip(dir_labels, phase_labels_np, sample_losses_np)):
                direction_key = str(int(direction_label))
                direction_phase_losses.setdefault(f"{direction_key}:{int(phase_label)}", []).append(float(loss))
                direction_action_errors.setdefault(direction_key, []).append(per_dim_np[row_idx])
            if has_gripper_head:
                logits = model.predict_gripper_logits(x)
                target = (y[:, 6] > 0.0).float()
                bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
                pred_label = (torch.sigmoid(logits) >= 0.5).float()
                correct = (pred_label == target).float()
                gripper_correct.extend(correct.detach().cpu().numpy().astype(float).tolist())
                gripper_bce.extend(bce.detach().cpu().numpy().astype(float).tolist())
                for label, ok in zip(batch["phase_label"].numpy().tolist(), correct.detach().cpu().numpy().tolist()):
                    gripper_phase_acc.setdefault(str(int(label)), []).append(float(ok))
                for label, ok in zip(batch["direction_label"].numpy().tolist(), correct.detach().cpu().numpy().tolist()):
                    gripper_direction_acc.setdefault(str(int(label)), []).append(float(ok))
            if has_gripper_magnitude_head:
                mag_pred = model.predict_gripper_magnitude(x)
                mag_target = y[:, 6]
                mag_error = (mag_pred - mag_target) ** 2
                mag_error_np = mag_error.detach().cpu().numpy().astype(float)
                gripper_magnitude_errors.extend(mag_error_np.tolist())
                for label, err in zip(batch["phase_label"].numpy().tolist(), mag_error_np.tolist()):
                    gripper_magnitude_phase_errors.setdefault(str(int(label)), []).append(float(err))
                for label, err in zip(batch["direction_label"].numpy().tolist(), mag_error_np.tolist()):
                    gripper_magnitude_direction_errors.setdefault(str(int(label)), []).append(float(err))
                for direction_label, phase_label, err in zip(dir_labels, phase_labels_np, mag_error_np.tolist()):
                    gripper_magnitude_direction_phase_errors.setdefault(f"{int(direction_label)}:{int(phase_label)}", []).append(float(err))
            # Stage 20-S: holding aux evaluation
            if has_holding_head:
                phase_labels_dev = batch["phase_label"].to(device)
                bce, correct, _target = _holding_aux_loss(model, x, batch, phase_labels_dev, set())
                if bce is not None and correct is not None:
                    holding_correct.extend(correct.detach().cpu().numpy().astype(float).tolist())
                    holding_bce.append(float(bce.detach().cpu().item()))
                    for label, ok in zip(batch["phase_label"].numpy().tolist(), correct.detach().cpu().numpy().tolist()):
                        holding_phase_acc.setdefault(str(int(label)), []).append(float(ok))
                    for label, ok in zip(batch["direction_label"].numpy().tolist(), correct.detach().cpu().numpy().tolist()):
                        holding_direction_acc.setdefault(str(int(label)), []).append(float(ok))
    if not losses:
        return {
            "loss": float("inf"),
            "per_direction_loss": {},
            "per_phase_loss": {},
            "action_dim_mse": [],
            "per_direction_phase_loss": {},
            "per_direction_action_dim_mse": {},
            "gripper_accuracy": None,
            "gripper_bce": None,
            "per_phase_gripper_accuracy": {},
            "per_direction_gripper_accuracy": {},
            "gripper_magnitude_mse": None,
            "per_phase_gripper_magnitude_mse": {},
            "per_direction_gripper_magnitude_mse": {},
            "per_direction_phase_gripper_magnitude_mse": {},
            # Stage 20-S
            "holding_accuracy": None,
            "holding_bce": None,
            "per_phase_holding_accuracy": {},
            "per_direction_holding_accuracy": {},
        }
    action_arr = np.concatenate(action_errors, axis=0)
    direction_action_dim_mse = {
        key: np.stack(values, axis=0).mean(axis=0).astype(float).tolist()
        for key, values in sorted(direction_action_errors.items())
        if values
    }
    return {
        "loss": float(np.mean(losses)),
        "per_direction_loss": _mean_dict(direction_losses),
        "per_phase_loss": _mean_dict(phase_losses),
        "per_direction_phase_loss": _mean_dict(direction_phase_losses),
        "per_direction_action_dim_mse": direction_action_dim_mse,
        "action_dim_mse": action_arr.mean(axis=0).astype(float).tolist(),
        "gripper_accuracy": float(np.mean(gripper_correct)) if gripper_correct else None,
        "gripper_bce": float(np.mean(gripper_bce)) if gripper_bce else None,
        "per_phase_gripper_accuracy": _mean_dict(gripper_phase_acc),
        "per_direction_gripper_accuracy": _mean_dict(gripper_direction_acc),
        "gripper_magnitude_mse": float(np.mean(gripper_magnitude_errors)) if gripper_magnitude_errors else None,
        "per_phase_gripper_magnitude_mse": _mean_dict(gripper_magnitude_phase_errors),
        "per_direction_gripper_magnitude_mse": _mean_dict(gripper_magnitude_direction_errors),
        "per_direction_phase_gripper_magnitude_mse": _mean_dict(gripper_magnitude_direction_phase_errors),
        # Stage 20-S: holding aux metrics
        "holding_accuracy": float(np.mean(holding_correct)) if holding_correct else None,
        "holding_bce": float(np.mean(holding_bce)) if holding_bce else None,
        "per_phase_holding_accuracy": _mean_dict(holding_phase_acc),
        "per_direction_holding_accuracy": _mean_dict(holding_direction_acc),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _record_phase(record: dict[str, Any]) -> int:
    return int(record.get("phase_id", -1))


def _record_direction(record: dict[str, Any]) -> int:
    return int(record.get("direction_id", -1))


def build_sampler_weights(dataset: Stage20RoleConditionedDataset, mode: str) -> np.ndarray | None:
    """Build per-sample weights for phase/direction balanced sampling."""
    if mode == "none":
        return None
    if mode not in {"phase_balanced", "phase_direction_balanced"}:
        raise ValueError(f"Unsupported sampler: {mode}")

    keys: list[Any] = []
    for record in dataset.records:
        phase = _record_phase(record)
        if mode == "phase_balanced":
            keys.append(phase)
        else:
            keys.append((phase, _record_direction(record)))
    counts: dict[Any, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray([1.0 / max(counts[key], 1) for key in keys], dtype=np.float64)
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    return weights


def _load_manual_phase_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(raw)
    return {str(k): float(v) for k, v in payload.items()}


def build_loss_phase_weights(
    dataset: Stage20RoleConditionedDataset,
    mode: str,
    manual_phase_weights: str | None = None,
) -> dict[str, float]:
    """Build phase weights for per-sample weighted MSE."""
    if mode == "none":
        return {}
    if mode == "manual":
        weights = _load_manual_phase_weights(manual_phase_weights)
        if not weights:
            raise ValueError("manual loss weighting requires --manual_phase_weights")
        return weights
    if mode != "inverse_phase_freq":
        raise ValueError(f"Unsupported loss_weighting: {mode}")

    counts: dict[str, int] = {}
    for record in dataset.records:
        key = str(_record_phase(record))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {}
    total = float(sum(counts.values()))
    num_classes = float(len(counts))
    return {phase: total / (num_classes * count) for phase, count in sorted(counts.items())}


def _weighted_action_mse(pred: torch.Tensor, target: torch.Tensor, phase_labels: torch.Tensor, phase_weights: dict[str, float]) -> torch.Tensor:
    per_sample = ((pred - target) ** 2).mean(dim=1)
    if not phase_weights:
        return per_sample.mean()
    weights = torch.ones_like(per_sample)
    for phase, weight in phase_weights.items():
        weights = torch.where(phase_labels == int(phase), torch.full_like(weights, float(weight)), weights)
    return (per_sample * weights).sum() / torch.clamp(weights.sum(), min=1e-8)


def _parse_phase_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(x.strip()) for x in str(raw).split(",") if x.strip()}


def _gripper_aux_loss(model: GoalConditionedRNN, x: torch.Tensor, y: torch.Tensor, phase_labels: torch.Tensor, phase_ids: set[int]) -> torch.Tensor | None:
    if model.gripper_head is None:
        return None
    logits = model.predict_gripper_logits(x)
    target = (y[:, 6] > 0.0).float()
    if phase_ids:
        mask = torch.zeros_like(target, dtype=torch.bool)
        for phase_id in phase_ids:
            mask = mask | (phase_labels == phase_id)
        if not bool(mask.any()):
            return None
        logits = logits[mask]
        target = target[mask]
    return nn.functional.binary_cross_entropy_with_logits(logits, target)


def _gripper_magnitude_aux_loss(
    model: GoalConditionedRNN,
    x: torch.Tensor,
    y: torch.Tensor,
    phase_labels: torch.Tensor,
    phase_ids: set[int],
) -> torch.Tensor | None:
    if model.gripper_magnitude_head is None:
        return None
    pred = model.predict_gripper_magnitude(x)
    target = y[:, 6]
    if phase_ids:
        mask = torch.zeros_like(target, dtype=torch.bool)
        for phase_id in phase_ids:
            mask = mask | (phase_labels == phase_id)
        if not bool(mask.any()):
            return None
        pred = pred[mask]
        target = target[mask]
    return nn.functional.mse_loss(pred, target)


# Stage 20-S: holding auxiliary loss
def _holding_aux_loss(
    model: GoalConditionedRNN,
    x: torch.Tensor,
    batch: dict[str, torch.Tensor],
    phase_labels: torch.Tensor,
    phase_ids: set[int],
    phase_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compute holding BCE loss on samples with holding_label_mask=1.

    Returns (loss, accuracy_tensor, target_tensor) or (None, None, None) if no valid samples.
    """
    if model.holding_head is None:
        return None, None, None
    holding_label = batch.get("holding_label")
    holding_mask = batch.get("holding_label_mask")
    if holding_label is None or holding_mask is None:
        return None, None, None
    # Move to same device as phase_labels
    holding_label = holding_label.to(phase_labels.device).reshape(-1)
    holding_mask = holding_mask.to(phase_labels.device).reshape(-1)
    logits = model.predict_holding_logits(x)
    target = (holding_label > 0.5).float()
    mask = (holding_mask > 0.5).bool()
    if phase_ids:
        phase_mask = torch.zeros_like(mask, dtype=torch.bool)
        for pid in phase_ids:
            phase_mask = phase_mask | (phase_labels == pid)
        mask = mask & phase_mask
    if not bool(mask.any()):
        return None, None, None
    logits_masked = logits[mask]
    target_masked = target[mask]
    bce = nn.functional.binary_cross_entropy_with_logits(logits_masked, target_masked, reduction="mean")
    pred_label = (torch.sigmoid(logits_masked) >= 0.5).float()
    correct = (pred_label == target_masked).float()
    return bce, correct, target_masked


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_ds = Stage20RoleConditionedDataset(
        args.dataset_dir,
        split="train",
        feature_mode=args.feature_mode,
        return_sequence=True,
    )
    val_ds = Stage20RoleConditionedDataset(
        args.dataset_dir,
        split="val",
        feature_mode=args.feature_mode,
        return_sequence=True,
    )
    sampler_mode = getattr(args, "sampler", "none")
    loss_weighting = getattr(args, "loss_weighting", "none")
    manual_phase_weights_raw = getattr(args, "manual_phase_weights", None)
    weighted_metric_report = bool(getattr(args, "weighted_metric_report", False))
    gripper_aux_weight = float(getattr(args, "gripper_aux_weight", 0.0) or 0.0)
    gripper_aux_phase_ids = _parse_phase_ids(getattr(args, "gripper_aux_phase_ids", "1"))
    gripper_magnitude_aux_weight = float(getattr(args, "gripper_magnitude_aux_weight", 0.0) or 0.0)
    gripper_magnitude_aux_phase_ids = _parse_phase_ids(getattr(args, "gripper_magnitude_aux_phase_ids", "1"))
    # Stage 20-S: holding aux
    holding_aux_weight = float(getattr(args, "holding_aux_weight", 0.0) or 0.0)
    holding_aux_phase_ids = _parse_phase_ids(getattr(args, "holding_aux_phase_ids", "2,3"))
    holding_aux_phase_weights_raw = getattr(args, "holding_aux_phase_weights", None)
    holding_aux_phase_weights = _load_manual_phase_weights(holding_aux_phase_weights_raw) if holding_aux_phase_weights_raw else {}
    sampler_weights = build_sampler_weights(train_ds, sampler_mode)
    sampler = None
    if sampler_weights is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=len(sampler_weights),
            replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    phase_loss_weights = build_loss_phase_weights(train_ds, loss_weighting, manual_phase_weights_raw)

    model = GoalConditionedRNN(
        input_dim=train_ds.input_dim,
        action_dim=train_ds.action_dim,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gripper_head=gripper_aux_weight > 0.0,
        use_gripper_magnitude_head=gripper_magnitude_aux_weight > 0.0,
        use_holding_head=holding_aux_weight > 0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output),
        "model_type": args.model_type,
        "feature_mode": args.feature_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seed": args.seed,
        "requested_device": args.device,
        "device": str(device),
        "input_dim": train_ds.input_dim,
        "action_dim": train_ds.action_dim,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "return_sequence": True,
        "sampler": sampler_mode,
        "loss_weighting": loss_weighting,
        "manual_phase_weights": phase_loss_weights if loss_weighting == "manual" else {},
        "phase_loss_weights": phase_loss_weights,
        "weighted_metric_report": weighted_metric_report,
        "gripper_aux_weight": gripper_aux_weight,
        "gripper_aux_phase_ids": sorted(gripper_aux_phase_ids),
        "gripper_magnitude_aux_weight": gripper_magnitude_aux_weight,
        "gripper_magnitude_aux_phase_ids": sorted(gripper_magnitude_aux_phase_ids),
        # Stage 20-S: holding aux config
        "holding_aux_weight": holding_aux_weight,
        "holding_aux_phase_ids": sorted(holding_aux_phase_ids),
        "holding_aux_phase_weights": holding_aux_phase_weights,
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
                phase_labels = batch["phase_label"].to(device)
                pred = model(x)
                loss = _weighted_action_mse(pred, y, phase_labels, phase_loss_weights)
                aux_loss = _gripper_aux_loss(model, x, y, phase_labels, gripper_aux_phase_ids)
                if aux_loss is not None:
                    loss = loss + gripper_aux_weight * aux_loss
                mag_aux_loss = _gripper_magnitude_aux_loss(model, x, y, phase_labels, gripper_magnitude_aux_phase_ids)
                if mag_aux_loss is not None:
                    loss = loss + gripper_magnitude_aux_weight * mag_aux_loss
                # Stage 20-S: holding aux loss
                if holding_aux_weight > 0.0:
                    holding_loss, _h_correct, _h_target = _holding_aux_loss(
                        model, x, batch, phase_labels, holding_aux_phase_ids, holding_aux_phase_weights)
                    if holding_loss is not None:
                        loss = loss + holding_aux_weight * holding_loss
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
            if val_metrics.get("gripper_accuracy") is not None:
                row["val_gripper_accuracy"] = val_metrics["gripper_accuracy"]
            if val_metrics.get("gripper_magnitude_mse") is not None:
                row["val_gripper_magnitude_mse"] = val_metrics["gripper_magnitude_mse"]
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
        "model_type": args.model_type,
        "feature_mode": args.feature_mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_loss": train_eval["loss"],
        "final_val_loss": final_val["loss"],
        "val_loss": final_val["loss"],
        "per_direction_val_loss": final_val["per_direction_loss"],
        "per_phase_val_loss": final_val["per_phase_loss"],
        "per_direction_phase_val_loss": final_val.get("per_direction_phase_loss", {}),
        "per_direction_action_dim_mse": final_val.get("per_direction_action_dim_mse", {}),
        "action_dim_mse": final_val["action_dim_mse"],
        "gripper_accuracy": final_val.get("gripper_accuracy"),
        "gripper_bce": final_val.get("gripper_bce"),
        "per_phase_gripper_accuracy": final_val.get("per_phase_gripper_accuracy", {}),
        "per_direction_gripper_accuracy": final_val.get("per_direction_gripper_accuracy", {}),
        "gripper_magnitude_mse": final_val.get("gripper_magnitude_mse"),
        "per_phase_gripper_magnitude_mse": final_val.get("per_phase_gripper_magnitude_mse", {}),
        "per_direction_gripper_magnitude_mse": final_val.get("per_direction_gripper_magnitude_mse", {}),
        "per_direction_phase_gripper_magnitude_mse": final_val.get("per_direction_phase_gripper_magnitude_mse", {}),
        # Stage 20-S: holding metrics
        "holding_accuracy": final_val.get("holding_accuracy"),
        "holding_bce": final_val.get("holding_bce"),
        "per_phase_holding_accuracy": final_val.get("per_phase_holding_accuracy", {}),
        "per_direction_holding_accuracy": final_val.get("per_direction_holding_accuracy", {}),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "input_dim": train_ds.input_dim,
        "action_dim": train_ds.action_dim,
        "checkpoint_best": str(output / "best.pt"),
        "checkpoint_last": str(output / "last.pt"),
        "sampler": sampler_mode,
        "loss_weighting": loss_weighting,
        "manual_phase_weights": phase_loss_weights if loss_weighting == "manual" else {},
        "phase_loss_weights": phase_loss_weights,
        "gripper_aux_weight": gripper_aux_weight,
        "gripper_aux_phase_ids": sorted(gripper_aux_phase_ids),
        "gripper_magnitude_aux_weight": gripper_magnitude_aux_weight,
        "gripper_magnitude_aux_phase_ids": sorted(gripper_magnitude_aux_phase_ids),
        # Stage 20-S
        "holding_aux_weight": holding_aux_weight,
        "holding_aux_phase_ids": sorted(holding_aux_phase_ids),
        "holding_aux_phase_weights": holding_aux_phase_weights,
    }
    if weighted_metric_report and phase_loss_weights:
        metrics["weighted_train_loss"] = train_loss
    _write_json(output / "metrics.json", metrics)
    (output / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage 20 goal-conditioned RNN")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--feature_mode", choices=["obs_only", "goal_conditioned"], default="goal_conditioned")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler", choices=["none", "phase_balanced", "phase_direction_balanced"], default="none")
    parser.add_argument("--loss_weighting", choices=["none", "inverse_phase_freq", "manual"], default="none")
    parser.add_argument("--manual_phase_weights", default=None)
    parser.add_argument("--weighted_metric_report", nargs="?", const=True, default=False, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"})
    parser.add_argument("--gripper_aux_weight", type=float, default=0.0)
    parser.add_argument("--gripper_aux_phase_ids", default="1")
    parser.add_argument("--gripper_magnitude_aux_weight", type=float, default=0.0)
    parser.add_argument("--gripper_magnitude_aux_phase_ids", default="1")
    # Stage 20-S: holding aux args
    parser.add_argument("--holding_aux_weight", type=float, default=0.0)
    parser.add_argument("--holding_aux_phase_ids", default="2,3")
    parser.add_argument("--holding_aux_phase_weights", default=None)
    parser.add_argument("--holding_label_key", default="holding_label")
    parser.add_argument("--holding_label_mask_key", default="holding_label_mask")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
