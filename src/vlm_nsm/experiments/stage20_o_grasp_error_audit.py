"""Offline Stage 20-O audit for direction-1 / phase-1 grasp errors."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset
from vlm_nsm.training.train_stage20_goal_conditioned_rnn import GoalConditionedRNN, resolve_device


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _window_index(record: dict[str, Any]) -> int:
    match = re.search(r"(\d+)$", str(record.get("window_id", "")))
    if not match:
        raise ValueError(f"Cannot parse window_id: {record.get('window_id')}")
    return int(match.group(1))


def _to_float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _safe_percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return _to_float(np.percentile(values, q))


def _series_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": _to_float(values.mean()),
        "std": _to_float(values.std()),
        "min": _to_float(values.min()),
        "p05": _safe_percentile(values, 5),
        "p25": _safe_percentile(values, 25),
        "median": _safe_percentile(values, 50),
        "p75": _safe_percentile(values, 75),
        "p95": _safe_percentile(values, 95),
        "max": _to_float(values.max()),
    }


def _mse(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return _to_float(np.mean(values**2))


def _mean_vector(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    return np.asarray(values, dtype=np.float64).mean(axis=0).astype(float).tolist()


def _summarize_predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if target.size == 0:
        return {"count": 0}
    error = pred - target
    dim6_target = target[:, 6]
    dim6_pred = pred[:, 6]
    dim6_error = dim6_pred - dim6_target
    target_close = dim6_target > 0.0
    pred_close = dim6_pred > 0.0
    close_magnitude = np.maximum(dim6_target, 0.0)
    open_magnitude = np.maximum(-dim6_target, 0.0)
    return {
        "count": int(target.shape[0]),
        "sample_mse": _mse(error),
        "action_dim_mse": _mean_vector(error**2),
        "action_dim6_target": _series_stats(dim6_target),
        "action_dim6_prediction": _series_stats(dim6_pred),
        "action_dim6_error": _series_stats(dim6_error),
        "action_dim6_abs_error": _series_stats(np.abs(dim6_error)),
        "action_dim6_mse": _mse(dim6_error),
        "target_close_ratio": _to_float(np.mean(target_close)),
        "pred_close_ratio": _to_float(np.mean(pred_close)),
        "sign_accuracy": _to_float(np.mean(target_close == pred_close)),
        "target_close_magnitude": _series_stats(close_magnitude[target_close]),
        "target_open_magnitude": _series_stats(open_magnitude[~target_close]),
    }


def _summarize_feature_distribution(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    if indices.size == 0:
        return {"count": 0}
    role = arrays["role_state_history"][indices, -1, :]
    goal = arrays["goal_delta_history"][indices, -1, :]
    progress = arrays["progress_scalar_history"][indices, -1]
    rel_gripper_to_top = np.linalg.norm(role[:, 10:13], axis=1)
    rel_top_to_bottom = np.linalg.norm(role[:, 13:16], axis=1)
    return {
        "count": int(indices.size),
        "gripper_open": _series_stats(role[:, 3]),
        "relative_gripper_to_top_norm": _series_stats(rel_gripper_to_top),
        "relative_top_to_bottom_norm": _series_stats(rel_top_to_bottom),
        "goal_gripper_open_delta": _series_stats(goal[:, 6]),
        "progress_scalar": _series_stats(progress),
    }


def _load_model(
    checkpoint_path: str | Path,
    dataset: Stage20RoleConditionedDataset,
    device: torch.device,
) -> GoalConditionedRNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    use_gripper_head = any(str(k).startswith("gripper_head.") for k in state_dict)
    use_gripper_magnitude_head = any(str(k).startswith("gripper_magnitude_head.") for k in state_dict)
    use_holding_head = any(str(k).startswith("holding_head.") for k in state_dict)
    model = GoalConditionedRNN(
        input_dim=int(config.get("input_dim", dataset.input_dim)),
        action_dim=int(config.get("action_dim", dataset.action_dim)),
        model_type=config.get("model_type", "lstm"),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        use_gripper_head=use_gripper_head,
        use_gripper_magnitude_head=use_gripper_magnitude_head,
        use_holding_head=use_holding_head,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _collect_predictions(
    *,
    dataset_dir: str | Path,
    checkpoint: str | Path,
    split: str,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    dataset = Stage20RoleConditionedDataset(
        dataset_dir,
        split=split,
        feature_mode="goal_conditioned",
        return_sequence=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _load_model(checkpoint, dataset, device)
    index_to_record = {_window_index(record): record for record in dataset.records}

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    directions: list[int] = []
    phases: list[int] = []
    indices: list[int] = []
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            pred = model(x).detach().cpu().numpy()
            target = batch["action"].detach().cpu().numpy()
            batch_indices = [int(x) for x in batch["index"].numpy().tolist()]
            preds.append(pred)
            targets.append(target)
            directions.extend(int(x) for x in batch["direction_label"].numpy().tolist())
            phases.extend(int(x) for x in batch["phase_label"].numpy().tolist())
            indices.extend(batch_indices)
            records.extend(index_to_record[i] for i in batch_indices)

    return {
        "dataset": dataset,
        "prediction": np.concatenate(preds, axis=0) if preds else np.zeros((0, 7), dtype=np.float32),
        "target": np.concatenate(targets, axis=0) if targets else np.zeros((0, 7), dtype=np.float32),
        "direction": np.asarray(directions, dtype=np.int64),
        "phase": np.asarray(phases, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "records": records,
        "device": str(device),
    }


def _group_summary(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    phase: np.ndarray,
) -> dict[str, Any]:
    by_direction: dict[str, Any] = {}
    for d in sorted(set(direction.astype(int).tolist())):
        mask = direction == d
        by_direction[str(d)] = _summarize_predictions(pred[mask], target[mask])

    by_phase: dict[str, Any] = {}
    for p in sorted(set(phase.astype(int).tolist())):
        mask = phase == p
        by_phase[str(p)] = _summarize_predictions(pred[mask], target[mask])

    by_direction_phase: dict[str, Any] = {}
    for d in sorted(set(direction.astype(int).tolist())):
        for p in sorted(set(phase.astype(int).tolist())):
            mask = (direction == d) & (phase == p)
            if mask.any():
                by_direction_phase[f"{d}:{p}"] = _summarize_predictions(pred[mask], target[mask])

    return {
        "by_direction": by_direction,
        "by_phase": by_phase,
        "by_direction_phase": by_direction_phase,
    }


def _trajectory_anomalies(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    records: list[dict[str, Any]],
    direction: np.ndarray,
    phase: np.ndarray,
    target_direction: int,
    target_phase: int,
    top_k: int,
) -> list[dict[str, Any]]:
    mask = (direction == target_direction) & (phase == target_phase)
    selected = np.where(mask)[0]
    grouped: dict[str, list[int]] = {}
    for row in selected:
        record = records[int(row)]
        episode_id = str(record.get("episode_id", "unknown"))
        grouped.setdefault(episode_id, []).append(int(row))

    rows: list[dict[str, Any]] = []
    for episode_id, sample_rows in grouped.items():
        sample_idx = np.asarray(sample_rows, dtype=np.int64)
        error = pred[sample_idx] - target[sample_idx]
        dim6_error = error[:, 6]
        example = records[int(sample_idx[0])]
        rows.append(
            {
                "episode_id": episode_id,
                "count": int(sample_idx.size),
                "placement_bin": example.get("placement_bin", "unknown"),
                "success": example.get("success"),
                "action_dim6_mse": _mse(dim6_error),
                "sample_mse": _mse(error),
                "target_close_ratio": _to_float(np.mean(target[sample_idx, 6] > 0.0)),
                "target_action_dim6_mean": _to_float(np.mean(target[sample_idx, 6])),
                "pred_action_dim6_mean": _to_float(np.mean(pred[sample_idx, 6])),
                "max_abs_action_dim6_error": _to_float(np.max(np.abs(dim6_error))),
            }
        )
    rows.sort(key=lambda item: float(item["action_dim6_mse"] or 0.0), reverse=True)
    return rows[:top_k]


def _placement_summary(
    *,
    pred: np.ndarray,
    target: np.ndarray,
    records: list[dict[str, Any]],
    direction: np.ndarray,
    phase: np.ndarray,
    target_direction: int,
    target_phase: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for row in np.where((direction == target_direction) & (phase == target_phase))[0]:
        placement = str(records[int(row)].get("placement_bin", "unknown"))
        groups.setdefault(placement, []).append(int(row))
    out: dict[str, Any] = {}
    for placement, rows in sorted(groups.items()):
        idx = np.asarray(rows, dtype=np.int64)
        out[placement] = _summarize_predictions(pred[idx], target[idx])
    return out


def _compare_direction_phase(
    summaries: dict[str, Any],
    d0_phase_key: str = "0:1",
    d1_phase_key: str = "1:1",
) -> dict[str, Any]:
    dp = summaries["by_direction_phase"]
    left = dp.get(d0_phase_key, {"count": 0})
    right = dp.get(d1_phase_key, {"count": 0})
    return {
        "direction0_phase1": left,
        "direction1_phase1": right,
        "action_dim6_mse_gap": (
            None
            if left.get("action_dim6_mse") is None or right.get("action_dim6_mse") is None
            else _to_float(float(right["action_dim6_mse"]) - float(left["action_dim6_mse"]))
        ),
        "sample_mse_gap": (
            None
            if left.get("sample_mse") is None or right.get("sample_mse") is None
            else _to_float(float(right["sample_mse"]) - float(left["sample_mse"]))
        ),
        "target_close_ratio_gap": (
            None
            if left.get("count", 0) == 0 or right.get("count", 0) == 0
            else _to_float(float(right["target_close_ratio"]) - float(left["target_close_ratio"]))
        ),
        "sign_accuracy_gap": (
            None
            if left.get("count", 0) == 0 or right.get("count", 0) == 0
            else _to_float(float(right["sign_accuracy"]) - float(left["sign_accuracy"]))
        ),
    }


def _diagnose(comparison: dict[str, Any], target_summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> dict[str, Any]:
    d1 = comparison.get("direction1_phase1", {})
    d0 = comparison.get("direction0_phase1", {})
    action6_gap = comparison.get("action_dim6_mse_gap")
    sign_acc = d1.get("sign_accuracy")
    close_gap = comparison.get("target_close_ratio_gap")
    episode_concentration = 0.0
    if anomalies and d1.get("count"):
        episode_concentration = sum(int(row["count"]) for row in anomalies[:5]) / max(int(d1["count"]), 1)
    likely: list[str] = []
    if sign_acc is not None and sign_acc >= 0.95 and action6_gap is not None and action6_gap > 0:
        likely.append("magnitude_regression_or_timing")
    if close_gap is not None and abs(close_gap) > 0.15:
        likely.append("direction_specific_target_distribution")
    if episode_concentration > 0.30:
        likely.append("trajectory_level_outliers")
    if d0.get("count", 0) and d1.get("count", 0):
        d0_rel = ((target_summary.get("direction0_phase1") or {}).get("relative_gripper_to_top_norm") or {}).get("mean")
        d1_rel = ((target_summary.get("direction1_phase1") or {}).get("relative_gripper_to_top_norm") or {}).get("mean")
        if d0_rel is not None and d1_rel is not None and abs(float(d1_rel) - float(d0_rel)) > 0.01:
            likely.append("direction_specific_state_distribution")
    if not likely:
        likely.append("needs_more_fine_grained_audit")

    if "magnitude_regression_or_timing" in likely:
        recommendation = (
            "Prioritize a gripper magnitude auxiliary head or contact/holding auxiliary target. "
            "The binary close/open decision is not the main failure signal."
        )
    elif "direction_specific_target_distribution" in likely or "trajectory_level_outliers" in likely:
        recommendation = (
            "Prioritize direction-1 targeted data inspection/collection and keep the auxiliary target simple until "
            "the target distribution is cleaned."
        )
    else:
        recommendation = (
            "Run a deeper per-trajectory audit before adding another training objective; do not start rollout smoke yet."
        )
    return {
        "likely_error_sources": likely,
        "top5_episode_concentration": _to_float(episode_concentration),
        "recommendation": recommendation,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    collected = _collect_predictions(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    dataset: Stage20RoleConditionedDataset = collected["dataset"]
    pred = collected["prediction"]
    target = collected["target"]
    direction = collected["direction"]
    phase = collected["phase"]
    records = collected["records"]
    indices = collected["indices"]

    summaries = _group_summary(pred=pred, target=target, direction=direction, phase=phase)
    comparison = _compare_direction_phase(summaries)
    target_mask_0 = (direction == 0) & (phase == 1)
    target_mask_1 = (direction == 1) & (phase == 1)
    feature_comparison = {
        "direction0_phase1": _summarize_feature_distribution(dataset.arrays, indices[target_mask_0]),
        "direction1_phase1": _summarize_feature_distribution(dataset.arrays, indices[target_mask_1]),
    }
    anomalies = _trajectory_anomalies(
        pred=pred,
        target=target,
        records=records,
        direction=direction,
        phase=phase,
        target_direction=1,
        target_phase=1,
        top_k=args.top_k,
    )
    placement = _placement_summary(
        pred=pred,
        target=target,
        records=records,
        direction=direction,
        phase=phase,
        target_direction=1,
        target_phase=1,
    )
    diagnosis = _diagnose(comparison, feature_comparison, anomalies)

    return {
        "stage": "stage20_o",
        "split": args.split,
        "device": collected["device"],
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "num_samples": int(target.shape[0]),
        "dataset_summary": _load_json(Path(args.dataset_dir) / "summary.json"),
        "overall": _summarize_predictions(pred, target),
        "groups": summaries,
        "direction_phase_comparison": comparison,
        "feature_distribution_comparison": feature_comparison,
        "direction1_phase1_placement_summary": placement,
        "direction1_phase1_trajectory_anomalies": anomalies,
        "diagnosis": diagnosis,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _compact_stats(stats: dict[str, Any], key: str) -> str:
    section = stats.get(key) or {}
    return (
        f"mean={_fmt(section.get('mean'))}, std={_fmt(section.get('std'))}, "
        f"p05={_fmt(section.get('p05'))}, p50={_fmt(section.get('median'))}, p95={_fmt(section.get('p95'))}"
    )


def build_markdown_report(audit: dict[str, Any]) -> str:
    comparison = audit["direction_phase_comparison"]
    d0 = comparison["direction0_phase1"]
    d1 = comparison["direction1_phase1"]
    diagnosis = audit["diagnosis"]
    feature = audit["feature_distribution_comparison"]
    anomaly_rows = "\n".join(
        "| {episode_id} | {count} | {placement_bin} | {action_dim6_mse} | {sample_mse} | {max_abs_action_dim6_error} |".format(
            episode_id=row.get("episode_id"),
            count=row.get("count"),
            placement_bin=row.get("placement_bin"),
            action_dim6_mse=_fmt(row.get("action_dim6_mse")),
            sample_mse=_fmt(row.get("sample_mse")),
            max_abs_action_dim6_error=_fmt(row.get("max_abs_action_dim6_error")),
        )
        for row in audit["direction1_phase1_trajectory_anomalies"][:10]
    )
    if not anomaly_rows:
        anomaly_rows = "| N/A | 0 | N/A | N/A | N/A | N/A |"

    return f"""# Stage 20-O Direction 1 / Phase 1 Grasp Error Audit

## Status

```text
Stage 20-O = PASS
audit_type = offline inference only
rollout_smoke = not_run
training = not_run
```

## Inputs

```text
dataset = {audit['dataset_dir']}
checkpoint = {audit['checkpoint']}
split = {audit['split']}
device = {audit['device']}
num_samples = {audit['num_samples']}
```

## Direction 0 vs Direction 1 / Phase 1

| Slice | Count | Sample MSE | Action Dim 6 MSE | Sign Accuracy | Target Close Ratio | Pred Close Ratio |
|-------|------:|-----------:|-----------------:|--------------:|-------------------:|-----------------:|
| direction 0 / phase 1 | {d0.get('count', 0)} | {_fmt(d0.get('sample_mse'))} | {_fmt(d0.get('action_dim6_mse'))} | {_fmt(d0.get('sign_accuracy'))} | {_fmt(d0.get('target_close_ratio'))} | {_fmt(d0.get('pred_close_ratio'))} |
| direction 1 / phase 1 | {d1.get('count', 0)} | {_fmt(d1.get('sample_mse'))} | {_fmt(d1.get('action_dim6_mse'))} | {_fmt(d1.get('sign_accuracy'))} | {_fmt(d1.get('target_close_ratio'))} | {_fmt(d1.get('pred_close_ratio'))} |

```text
sample_mse_gap = {_fmt(comparison.get('sample_mse_gap'))}
action_dim6_mse_gap = {_fmt(comparison.get('action_dim6_mse_gap'))}
target_close_ratio_gap = {_fmt(comparison.get('target_close_ratio_gap'))}
sign_accuracy_gap = {_fmt(comparison.get('sign_accuracy_gap'))}
```

## Action Dim 6 Distribution

```text
direction0_phase1_target = {_compact_stats(d0, 'action_dim6_target')}
direction0_phase1_prediction = {_compact_stats(d0, 'action_dim6_prediction')}
direction0_phase1_abs_error = {_compact_stats(d0, 'action_dim6_abs_error')}

direction1_phase1_target = {_compact_stats(d1, 'action_dim6_target')}
direction1_phase1_prediction = {_compact_stats(d1, 'action_dim6_prediction')}
direction1_phase1_abs_error = {_compact_stats(d1, 'action_dim6_abs_error')}
```

## State Distribution Check

```text
direction0_phase1_relative_gripper_to_top = {_compact_stats(feature['direction0_phase1'], 'relative_gripper_to_top_norm')}
direction1_phase1_relative_gripper_to_top = {_compact_stats(feature['direction1_phase1'], 'relative_gripper_to_top_norm')}
direction0_phase1_gripper_open = {_compact_stats(feature['direction0_phase1'], 'gripper_open')}
direction1_phase1_gripper_open = {_compact_stats(feature['direction1_phase1'], 'gripper_open')}
direction0_phase1_goal_gripper_open_delta = {_compact_stats(feature['direction0_phase1'], 'goal_gripper_open_delta')}
direction1_phase1_goal_gripper_open_delta = {_compact_stats(feature['direction1_phase1'], 'goal_gripper_open_delta')}
```

## Direction 1 / Phase 1 Trajectory Anomalies

| Episode | Count | Placement | Action Dim 6 MSE | Sample MSE | Max Abs Action Dim 6 Error |
|---------|------:|-----------|-----------------:|-----------:|---------------------------:|
{anomaly_rows}

## Diagnosis

```text
likely_error_sources = {diagnosis['likely_error_sources']}
top5_episode_concentration = {_fmt(diagnosis.get('top5_episode_concentration'))}
```

## Recommendation

{diagnosis['recommendation']}
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 20-O offline grasp error audit")
    parser.add_argument("--dataset_dir", default="generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled")
    parser.add_argument("--checkpoint", default="generated/results/stage_20/goal_conditioned_lstm_v0_2_phase_labeled_stage20_l/best.pt")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", default="generated/results/stage_20/stage20_o_grasp_error_audit.json")
    parser.add_argument("--output_md", default="generated/results/stage_20/stage20_o_grasp_error_audit.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run_audit(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    _write_json(output_json, audit)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown_report(audit), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
