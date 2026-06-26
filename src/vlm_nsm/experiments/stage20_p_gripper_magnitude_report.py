"""Generate Stage 20-P gripper magnitude auxiliary-head report."""

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


def _direction_phase_loss(metrics: dict[str, Any], direction: str, phase: str) -> float | None:
    losses = metrics.get("per_direction_phase_val_loss") or {}
    key = f"{direction}:{phase}"
    return float(losses[key]) if key in losses else None


def _direction_phase_mag(metrics: dict[str, Any], direction: str, phase: str) -> float | None:
    losses = metrics.get("per_direction_phase_gripper_magnitude_mse") or {}
    key = f"{direction}:{phase}"
    return float(losses[key]) if key in losses else None


def _audit_direction_phase(audit: dict[str, Any], direction: str, phase: str) -> dict[str, Any]:
    comparison = audit.get("direction_phase_comparison") or {}
    if direction == "0" and phase == "1":
        return comparison.get("direction0_phase1") or {}
    if direction == "1" and phase == "1":
        return comparison.get("direction1_phase1") or {}
    return ((audit.get("groups") or {}).get("by_direction_phase") or {}).get(f"{direction}:{phase}") or {}


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def build_report(
    *,
    baseline_metrics: dict[str, Any],
    gripper_class_metrics: dict[str, Any] | None,
    gripper_magnitude_metrics: dict[str, Any],
    audit_summary: dict[str, Any],
    dataset_summary: dict[str, Any],
    magnitude_audit_summary: dict[str, Any] | None = None,
) -> str:
    phase1_improved = (_phase_loss(gripper_magnitude_metrics, "1") or float("inf")) < (_phase_loss(baseline_metrics, "1") or float("inf"))
    direction_gap_improved = (_direction_gap(gripper_magnitude_metrics) or float("inf")) < (_direction_gap(baseline_metrics) or float("inf"))
    action6_improved = (_action6(gripper_magnitude_metrics) or float("inf")) < (_action6(baseline_metrics) or float("inf"))
    metric_direction1_action6_improved = (
        _direction_action6(gripper_magnitude_metrics, "1") is not None
        and _direction_action6(baseline_metrics, "1") is not None
        and _direction_action6(gripper_magnitude_metrics, "1") < _direction_action6(baseline_metrics, "1")
    )
    baseline_audit_d1p1 = _audit_direction_phase(audit_summary, "1", "1")
    magnitude_audit_d1p1 = _audit_direction_phase(magnitude_audit_summary or {}, "1", "1")
    audit_direction1_action6_improved = (
        baseline_audit_d1p1.get("action_dim6_mse") is not None
        and magnitude_audit_d1p1.get("action_dim6_mse") is not None
        and float(magnitude_audit_d1p1["action_dim6_mse"]) < float(baseline_audit_d1p1["action_dim6_mse"])
    )
    direction1_action6_improved = metric_direction1_action6_improved or audit_direction1_action6_improved
    best_loss_ok = float(gripper_magnitude_metrics.get("best_val_loss", float("inf"))) <= float(baseline_metrics.get("best_val_loss", float("inf"))) * 1.10
    strong_count = sum([phase1_improved, direction_gap_improved, action6_improved, direction1_action6_improved, best_loss_ok])
    status = "STRONG_PASS" if strong_count >= 3 and (phase1_improved or action6_improved) else "PASS" if best_loss_ok else "PARTIAL_PASS"

    class_row = ""
    if gripper_class_metrics:
        class_row = (
            f"| gripper_class_aux | {_fmt(gripper_class_metrics.get('train_loss'))} | {_fmt(_final(gripper_class_metrics))} | "
            f"{_fmt(gripper_class_metrics.get('best_val_loss'))} | {gripper_class_metrics.get('best_epoch')} | "
            f"{_fmt(_phase_loss(gripper_class_metrics, '1'))} | {_fmt(_direction_loss(gripper_class_metrics, '0'))} | "
            f"{_fmt(_direction_loss(gripper_class_metrics, '1'))} | {_fmt(_direction_gap(gripper_class_metrics))} | "
            f"{_fmt(_action6(gripper_class_metrics))} | {_fmt(_direction_action6(gripper_class_metrics, '1'))} |\n"
        )

    if action6_improved or direction1_action6_improved:
        if magnitude_audit_summary:
            recommendation = (
                "Keep the gripper magnitude auxiliary variant as the next offline candidate. The fresh offline audit is "
                "already complete; next compare a contact/holding auxiliary target or a lower magnitude weight before any rollout smoke."
            )
        else:
            recommendation = (
                "Keep the gripper magnitude auxiliary variant as the next offline candidate and run a fresh Stage 20-O audit "
                "against this checkpoint before any rollout smoke."
            )
    else:
        recommendation = (
            "Do not move to rollout smoke. The next candidate should be contact/holding auxiliary target or direction-1 "
            "phase-1 targeted data inspection, because pure magnitude supervision did not reduce the main bottleneck."
        )

    audit_diag = audit_summary.get("diagnosis", {})
    return f"""# Stage 20-P Gripper Magnitude Auxiliary Head Report

## Status

```text
Stage 20-P = {status}
rollout_smoke = not_run
contact_holding_target = deferred
```

## Dataset Summary

```text
num_windows = {dataset_summary.get('num_windows')}
train_windows = {dataset_summary.get('train_windows')}
val_windows = {dataset_summary.get('val_windows')}
direction_counts = {_counts(dataset_summary.get('direction_counts'))}
phase_counts = {_counts(dataset_summary.get('phase_counts'))}
```

## Stage 20-O Audit Context

```text
likely_error_sources = {audit_diag.get('likely_error_sources')}
recommendation = {audit_diag.get('recommendation')}
```

## Baseline vs Auxiliary Variants

| Run | Train Loss | Final Val Loss | Best Val Loss | Best Epoch | Phase 1 | Direction 0 | Direction 1 | Direction Gap | Action Dim 6 | Direction 1 Action Dim 6 |
|-----|-----------:|---------------:|--------------:|-----------:|--------:|------------:|------------:|--------------:|-------------:|-------------------------:|
| Stage 20-L baseline | {_fmt(baseline_metrics.get('train_loss'))} | {_fmt(_final(baseline_metrics))} | {_fmt(baseline_metrics.get('best_val_loss'))} | {baseline_metrics.get('best_epoch')} | {_fmt(_phase_loss(baseline_metrics, '1'))} | {_fmt(_direction_loss(baseline_metrics, '0'))} | {_fmt(_direction_loss(baseline_metrics, '1'))} | {_fmt(_direction_gap(baseline_metrics))} | {_fmt(_action6(baseline_metrics))} | {_fmt(_direction_action6(baseline_metrics, '1'))} |
{class_row}| gripper_magnitude_aux | {_fmt(gripper_magnitude_metrics.get('train_loss'))} | {_fmt(_final(gripper_magnitude_metrics))} | {_fmt(gripper_magnitude_metrics.get('best_val_loss'))} | {gripper_magnitude_metrics.get('best_epoch')} | {_fmt(_phase_loss(gripper_magnitude_metrics, '1'))} | {_fmt(_direction_loss(gripper_magnitude_metrics, '0'))} | {_fmt(_direction_loss(gripper_magnitude_metrics, '1'))} | {_fmt(_direction_gap(gripper_magnitude_metrics))} | {_fmt(_action6(gripper_magnitude_metrics))} | {_fmt(_direction_action6(gripper_magnitude_metrics, '1'))} |

## Gripper Magnitude Head Metrics

```text
gripper_magnitude_aux_weight = {gripper_magnitude_metrics.get('gripper_magnitude_aux_weight')}
gripper_magnitude_aux_phase_ids = {gripper_magnitude_metrics.get('gripper_magnitude_aux_phase_ids')}
gripper_magnitude_mse = {_fmt(gripper_magnitude_metrics.get('gripper_magnitude_mse'))}
per_phase_gripper_magnitude_mse = {_losses(gripper_magnitude_metrics.get('per_phase_gripper_magnitude_mse'))}
per_direction_gripper_magnitude_mse = {_losses(gripper_magnitude_metrics.get('per_direction_gripper_magnitude_mse'))}
direction1_phase1_gripper_magnitude_mse = {_fmt(_direction_phase_mag(gripper_magnitude_metrics, '1', '1'))}
```

## Direction 1 / Phase 1 Check

```text
baseline_direction1_phase1_loss = {_fmt(_direction_phase_loss(baseline_metrics, '1', '1'))}
magnitude_direction1_phase1_loss = {_fmt(_direction_phase_loss(gripper_magnitude_metrics, '1', '1'))}
baseline_direction1_action_dim6_mse = {_fmt(_direction_action6(baseline_metrics, '1'))}
magnitude_direction1_action_dim6_mse = {_fmt(_direction_action6(gripper_magnitude_metrics, '1'))}
baseline_audit_direction1_phase1_sample_mse = {_fmt(baseline_audit_d1p1.get('sample_mse'))}
magnitude_audit_direction1_phase1_sample_mse = {_fmt(magnitude_audit_d1p1.get('sample_mse'))}
baseline_audit_direction1_phase1_action_dim6_mse = {_fmt(baseline_audit_d1p1.get('action_dim6_mse'))}
magnitude_audit_direction1_phase1_action_dim6_mse = {_fmt(magnitude_audit_d1p1.get('action_dim6_mse'))}
```

## Assessment

```text
phase_1_improved = {str(phase1_improved).lower()}
direction_gap_improved = {str(direction_gap_improved).lower()}
action_dim_6_improved = {str(action6_improved).lower()}
direction_1_action_dim_6_improved = {str(direction1_action6_improved).lower()}
metric_direction_1_action_dim_6_improved = {str(metric_direction1_action6_improved).lower()}
audit_direction_1_phase_1_action_dim_6_improved = {str(audit_direction1_action6_improved).lower()}
best_loss_within_10pct = {str(best_loss_ok).lower()}
```

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-P gripper magnitude report")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--gripper_class_metrics", default=None)
    parser.add_argument("--gripper_magnitude_metrics", required=True)
    parser.add_argument("--audit_summary", required=True)
    parser.add_argument("--magnitude_audit_summary", default=None)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        baseline_metrics=_load(args.baseline_metrics),
        gripper_class_metrics=_load(args.gripper_class_metrics) if args.gripper_class_metrics else None,
        gripper_magnitude_metrics=_load(args.gripper_magnitude_metrics),
        audit_summary=_load(args.audit_summary),
        magnitude_audit_summary=_load(args.magnitude_audit_summary) if args.magnitude_audit_summary else None,
        dataset_summary=_load(args.dataset_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
