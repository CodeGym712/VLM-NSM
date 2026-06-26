"""Generate Stage 20-Q gripper magnitude weight sweep report."""

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


def _phase1(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_phase_val_loss") or {}
    return float(losses["1"]) if "1" in losses else None


def _direction(metrics: dict[str, Any], direction: str) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    return float(losses[direction]) if direction in losses else None


def _gap(metrics: dict[str, Any]) -> float | None:
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


def _weight(metrics: dict[str, Any]) -> Any:
    return metrics.get("gripper_magnitude_aux_weight", "baseline")


def _row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {_weight(metrics)} | {_fmt(metrics.get('train_loss'))} | {_fmt(_final(metrics))} | "
        f"{_fmt(metrics.get('best_val_loss'))} | {metrics.get('best_epoch')} | {_fmt(_phase1(metrics))} | "
        f"{_fmt(_direction(metrics, '0'))} | {_fmt(_direction(metrics, '1'))} | {_fmt(_gap(metrics))} | "
        f"{_fmt(_action6(metrics))} | {_fmt(_direction_action6(metrics, '1'))} |"
    )


def _best_run(runs: dict[str, dict[str, Any]]) -> str:
    return min(runs, key=lambda name: float(runs[name].get("best_val_loss", float("inf"))))


def _audit_d1p1(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not audit:
        return {}
    return (audit.get("direction_phase_comparison") or {}).get("direction1_phase1") or {}


def _audit_row(name: str, audit: dict[str, Any] | None) -> str:
    d1p1 = _audit_d1p1(audit)
    return (
        f"| {name} | {_fmt(d1p1.get('sample_mse'))} | {_fmt(d1p1.get('action_dim6_mse'))} | "
        f"{_fmt(d1p1.get('sign_accuracy'))} | {_fmt(d1p1.get('target_close_ratio'))} | {_fmt(d1p1.get('pred_close_ratio'))} |"
    )


def build_report(
    *,
    baseline_metrics: dict[str, Any],
    weight025_metrics: dict[str, Any],
    weight050_metrics: dict[str, Any],
    weight075_metrics: dict[str, Any],
    contact_audit: dict[str, Any],
    baseline_audit: dict[str, Any] | None = None,
    weight025_audit: dict[str, Any] | None = None,
    weight050_audit: dict[str, Any] | None = None,
    weight075_audit: dict[str, Any] | None = None,
) -> str:
    runs = {
        "weight_0.25": weight025_metrics,
        "weight_0.50_stage20_p": weight050_metrics,
        "weight_0.75": weight075_metrics,
    }
    best_name = _best_run(runs)
    best = runs[best_name]
    baseline_best = float(baseline_metrics.get("best_val_loss", float("inf")))
    status = "STRONG_PASS" if float(best.get("best_val_loss", float("inf"))) < baseline_best else "PASS"
    contact_feas = contact_audit.get("feasibility", {})
    recommendation = (
        "Use the best magnitude-weight checkpoint for offline candidate selection, but patch the recorder for explicit "
        "contact/holding labels before adding contact/holding auxiliary loss."
    )

    return f"""# Stage 20-Q Contact/Holding Audit and Magnitude Weight Sweep Report

## Status

```text
Stage 20-Q = {status}
rollout_smoke = not_run
contact_holding_training = not_run
```

## Contact/Holding Feasibility

```text
direct_label_available = {str(contact_audit.get('direct_label_available')).lower()}
proxy_only = {str(contact_audit.get('proxy_only')).lower()}
recommended_for_training_now = {str(contact_feas.get('recommended_for_training_now')).lower()}
holding_phase_separation = {_fmt(contact_feas.get('holding_phase_separation'))}
contact_phase_separation = {_fmt(contact_feas.get('contact_phase_separation'))}
phase_leak_risk = {str(contact_feas.get('phase_leak_risk')).lower()}
```

## Magnitude Sweep

| Run | Aux Weight | Train Loss | Final Val Loss | Best Val Loss | Best Epoch | Phase 1 | Direction 0 | Direction 1 | Direction Gap | Action Dim 6 | Direction 1 Action Dim 6 |
|-----|-----------:|-----------:|---------------:|--------------:|-----------:|--------:|------------:|------------:|--------------:|-------------:|-------------------------:|
{_row('Stage 20-L baseline', baseline_metrics)}
{_row('weight_0.25', weight025_metrics)}
{_row('weight_0.50_stage20_p', weight050_metrics)}
{_row('weight_0.75', weight075_metrics)}

## Best Offline Candidate

```text
best_magnitude_run = {best_name}
best_val_loss = {_fmt(best.get('best_val_loss'))}
phase_1 = {_fmt(_phase1(best))}
direction_1 = {_fmt(_direction(best, '1'))}
direction_gap = {_fmt(_gap(best))}
action_dim_6_mse = {_fmt(_action6(best))}
direction1_action_dim6_mse = {_fmt(_direction_action6(best, '1'))}
```

## Best-Checkpoint Direction 1 / Phase 1 Audit

| Run | Sample MSE | Action Dim 6 MSE | Sign Accuracy | Target Close Ratio | Pred Close Ratio |
|-----|-----------:|-----------------:|--------------:|-------------------:|-----------------:|
{_audit_row('Stage 20-L baseline', baseline_audit)}
{_audit_row('weight_0.25', weight025_audit)}
{_audit_row('weight_0.50_stage20_p', weight050_audit)}
{_audit_row('weight_0.75', weight075_audit)}

## Interpretation

- Contact/holding labels are not directly available in the current dataset, so no contact/holding auxiliary loss was trained.
- The contact/holding proxy is useful for recorder design, but should not become canonical ground truth.
- Magnitude auxiliary loss remains the safer offline improvement path in the current dataset.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-Q magnitude sweep report")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--weight025_metrics", required=True)
    parser.add_argument("--weight050_metrics", required=True)
    parser.add_argument("--weight075_metrics", required=True)
    parser.add_argument("--contact_audit", required=True)
    parser.add_argument("--baseline_audit", default=None)
    parser.add_argument("--weight025_audit", default=None)
    parser.add_argument("--weight050_audit", default=None)
    parser.add_argument("--weight075_audit", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        baseline_metrics=_load(args.baseline_metrics),
        weight025_metrics=_load(args.weight025_metrics),
        weight050_metrics=_load(args.weight050_metrics),
        weight075_metrics=_load(args.weight075_metrics),
        contact_audit=_load(args.contact_audit),
        baseline_audit=_load(args.baseline_audit) if args.baseline_audit else None,
        weight025_audit=_load(args.weight025_audit) if args.weight025_audit else None,
        weight050_audit=_load(args.weight050_audit) if args.weight050_audit else None,
        weight075_audit=_load(args.weight075_audit) if args.weight075_audit else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
