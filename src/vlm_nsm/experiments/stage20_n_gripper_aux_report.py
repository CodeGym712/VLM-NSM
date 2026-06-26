"""Generate Stage 20-N gripper auxiliary-head report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _final(metrics: dict[str, Any]) -> Any:
    return metrics.get("final_val_loss", metrics.get("val_loss"))


def _phase_loss(metrics: dict[str, Any], phase: str) -> float | None:
    losses = metrics.get("per_phase_val_loss") or {}
    return float(losses[phase]) if phase in losses else None


def _direction_loss(metrics: dict[str, Any], direction: str) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    return float(losses[direction]) if direction in losses else None


def _direction_gap(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    if len(losses) < 2:
        return None
    vals = [float(v) for v in losses.values()]
    return max(vals) - min(vals)


def _action6(metrics: dict[str, Any]) -> float | None:
    values = metrics.get("action_dim_mse") or []
    return float(values[6]) if len(values) > 6 else None


def _direction_action6(metrics: dict[str, Any], direction: str) -> float | None:
    values = (metrics.get("per_direction_action_dim_mse") or {}).get(direction)
    return float(values[6]) if values and len(values) > 6 else None


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _top_direction_phase_losses(metrics: dict[str, Any], direction: str = "1", limit: int = 6) -> str:
    losses = metrics.get("per_direction_phase_val_loss") or {}
    rows: list[tuple[str, float]] = []
    for key, value in losses.items():
        if str(key).startswith(f"{direction}:"):
            rows.append((str(key), float(value)))
    if not rows:
        return "N/A"
    rows.sort(key=lambda item: item[1], reverse=True)
    return "\n".join(f"{key} = {_fmt(value)}" for key, value in rows[:limit])


def build_report(
    *,
    baseline_metrics: dict[str, Any],
    gripper_aux_metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> str:
    phase1_improved = (_phase_loss(gripper_aux_metrics, "1") or float("inf")) < (_phase_loss(baseline_metrics, "1") or float("inf"))
    direction_gap_improved = (_direction_gap(gripper_aux_metrics) or float("inf")) < (_direction_gap(baseline_metrics) or float("inf"))
    action6_improved = (_action6(gripper_aux_metrics) or float("inf")) < (_action6(baseline_metrics) or float("inf"))
    best_loss_ok = float(gripper_aux_metrics.get("best_val_loss", float("inf"))) <= float(baseline_metrics.get("best_val_loss", float("inf"))) * 1.10
    direction1_action6_improved = (
        _direction_action6(gripper_aux_metrics, "1") is not None
        and _direction_action6(baseline_metrics, "1") is not None
        and _direction_action6(gripper_aux_metrics, "1") < _direction_action6(baseline_metrics, "1")
    )

    strong_count = sum([phase1_improved, direction_gap_improved, action6_improved, direction1_action6_improved, best_loss_ok])
    if strong_count >= 3 and (phase1_improved or action6_improved):
        status = "STRONG_PASS"
        recommendation = (
            "Use the gripper-aux checkpoint as the next offline candidate, then prepare a small rollout smoke plan "
            "with scripted fallback and explicit unsafe/runtime reporting."
        )
    elif best_loss_ok:
        status = "PASS"
        recommendation = (
            "The auxiliary head trains stably but does not solve all bottlenecks. Keep comparing against the Stage 20-L "
            "baseline, inspect direction-1 phase losses, and consider a stronger gripper auxiliary target before rollout."
        )
    else:
        status = "PARTIAL_PASS"
        recommendation = (
            "Do not proceed to rollout. Reduce auxiliary weight or separate gripper classification from regression loss."
        )

    return f"""# Stage 20-N Gripper Auxiliary Head Report

## Status

```text
Stage 20-N = {status}
```

## Dataset Summary

```text
num_windows = {dataset_summary.get('num_windows')}
train_windows = {dataset_summary.get('train_windows')}
val_windows = {dataset_summary.get('val_windows')}
direction_counts = {_counts(dataset_summary.get('direction_counts'))}
phase_counts = {_counts(dataset_summary.get('phase_counts'))}
```

## Baseline vs Gripper-Aux LSTM

| Run | Train Loss | Final Val Loss | Best Val Loss | Best Epoch | Phase 1 | Direction 0 | Direction 1 | Direction Gap | Action Dim 6 |
|-----|-----------:|---------------:|--------------:|-----------:|--------:|------------:|------------:|--------------:|-------------:|
| Stage 20-L baseline | {_fmt(baseline_metrics.get('train_loss'))} | {_fmt(_final(baseline_metrics))} | {_fmt(baseline_metrics.get('best_val_loss'))} | {baseline_metrics.get('best_epoch')} | {_fmt(_phase_loss(baseline_metrics, '1'))} | {_fmt(_direction_loss(baseline_metrics, '0'))} | {_fmt(_direction_loss(baseline_metrics, '1'))} | {_fmt(_direction_gap(baseline_metrics))} | {_fmt(_action6(baseline_metrics))} |
| gripper_aux | {_fmt(gripper_aux_metrics.get('train_loss'))} | {_fmt(_final(gripper_aux_metrics))} | {_fmt(gripper_aux_metrics.get('best_val_loss'))} | {gripper_aux_metrics.get('best_epoch')} | {_fmt(_phase_loss(gripper_aux_metrics, '1'))} | {_fmt(_direction_loss(gripper_aux_metrics, '0'))} | {_fmt(_direction_loss(gripper_aux_metrics, '1'))} | {_fmt(_direction_gap(gripper_aux_metrics))} | {_fmt(_action6(gripper_aux_metrics))} |

## Gripper Classification Metrics

```text
gripper_accuracy = {_fmt(gripper_aux_metrics.get('gripper_accuracy'))}
gripper_bce = {_fmt(gripper_aux_metrics.get('gripper_bce'))}
per_phase_gripper_accuracy = {_losses(gripper_aux_metrics.get('per_phase_gripper_accuracy'))}
per_direction_gripper_accuracy = {_losses(gripper_aux_metrics.get('per_direction_gripper_accuracy'))}
```

## Direction-1 Error Analysis

```text
baseline_direction_1_action_dim_6_mse = {_fmt(_direction_action6(baseline_metrics, '1'))}
gripper_aux_direction_1_action_dim_6_mse = {_fmt(_direction_action6(gripper_aux_metrics, '1'))}
direction_1_action_dim_6_improved = {str(direction1_action6_improved).lower()}

Top direction-1 phase losses:
{_top_direction_phase_losses(gripper_aux_metrics, '1')}
```

## Assessment

```text
phase_1_improved = {str(phase1_improved).lower()}
direction_gap_improved = {str(direction_gap_improved).lower()}
action_dim_6_improved = {str(action6_improved).lower()}
best_loss_within_10pct = {str(best_loss_ok).lower()}
```

## Interpretation

- The auxiliary head predicts close/open as a binary target from action dim 6 while action regression remains the main objective.
- Validation losses remain unweighted and best.pt is selected by unweighted best_val_loss.
- Direction 1 remains a separate risk if its loss or action dim 6 MSE stays high.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-N gripper auxiliary report")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--gripper_aux_metrics", required=True)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        baseline_metrics=_load(args.baseline_metrics),
        gripper_aux_metrics=_load(args.gripper_aux_metrics),
        dataset_summary=_load(args.dataset_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
