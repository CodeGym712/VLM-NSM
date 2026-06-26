"""Generate Stage 20-M report for balanced LSTM variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_PHASE1 = 0.0085728995
BASELINE_DIRECTION_GAP = 0.0024741058
BASELINE_ACTION_DIM6 = 0.011964
BEST_TOLERANCE = 0.0019
LOW_PHASE45_THRESHOLD = 1e-3


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
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


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {_fmt(metrics.get('train_loss'))} | {_fmt(_final(metrics))} | "
        f"{_fmt(metrics.get('best_val_loss'))} | {metrics.get('best_epoch')} | "
        f"{_fmt(_phase_loss(metrics, '1'))} | {_fmt(_phase_loss(metrics, '3'))} | "
        f"{_fmt(_phase_loss(metrics, '4'))} | {_fmt(_phase_loss(metrics, '5'))} | "
        f"{_fmt(_direction_loss(metrics, '0'))} | {_fmt(_direction_loss(metrics, '1'))} | "
        f"{_fmt(_direction_gap(metrics))} | {_fmt(_action6(metrics))} |"
    )


def _assessment(label: str, metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    phase4 = _phase_loss(metrics, "4")
    phase5 = _phase_loss(metrics, "5")
    phase45_stable = (
        phase4 is not None
        and phase5 is not None
        and phase4 < LOW_PHASE45_THRESHOLD
        and phase5 < LOW_PHASE45_THRESHOLD
    )
    return {
        "improves_phase_1": (_phase_loss(metrics, "1") or float("inf")) < (_phase_loss(baseline, "1") or BASELINE_PHASE1),
        "improves_direction_gap": (_direction_gap(metrics) or float("inf")) < (_direction_gap(baseline) or BASELINE_DIRECTION_GAP),
        "preserves_phase_4_5": phase45_stable,
        "keeps_best_val_loss_within_tolerance": float(metrics.get("best_val_loss", float("inf"))) <= BEST_TOLERANCE,
        "improves_action_dim_6": (_action6(metrics) or float("inf")) < (_action6(baseline) or BASELINE_ACTION_DIM6),
    }


def build_report(
    *,
    baseline_metrics: dict[str, Any],
    phase_balanced_metrics: dict[str, Any],
    phase_direction_balanced_metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
    weighted_loss_metrics: dict[str, Any] | None = None,
) -> str:
    runs: list[tuple[str, dict[str, Any]]] = [
        ("baseline", baseline_metrics),
        ("phase_balanced", phase_balanced_metrics),
        ("phase_direction_balanced", phase_direction_balanced_metrics),
    ]
    if weighted_loss_metrics is not None:
        runs.append(("weighted_loss", weighted_loss_metrics))
    assessments = {label: _assessment(label, metrics, baseline_metrics) for label, metrics in runs[1:]}
    strong_counts = {
        label: sum(
            [
                values["improves_phase_1"],
                values["improves_direction_gap"],
                values["preserves_phase_4_5"],
                values["keeps_best_val_loss_within_tolerance"],
                values["improves_action_dim_6"],
            ]
        )
        for label, values in assessments.items()
    }
    best_label = min(runs, key=lambda item: float(item[1].get("best_val_loss", float("inf"))))[0]
    any_strong = any(count >= 2 for count in strong_counts.values())
    targeted_improved = any(
        values["improves_phase_1"] or values["improves_direction_gap"] or values["improves_action_dim_6"]
        for values in assessments.values()
    )
    if any_strong and targeted_improved:
        recommendation = (
            f"Use `{best_label}` as the next offline checkpoint candidate, then run a small rollout smoke only with "
            "scripted fallback and explicit safety reporting. Continue sampler work if phase 1 or direction gap remains high."
        )
    elif any_strong:
        recommendation = (
            "Balanced runs preserve phase 4/5 and keep best_val_loss within tolerance, but they do not improve the main "
            "bottlenecks. Keep the Stage 20-L baseline checkpoint for now; next try a gripper classification head, "
            "direction-balanced oversampling with softer phase weights, or reduced manual weights before rollout smoke."
        )
    else:
        recommendation = (
            "Balanced training did not satisfy enough criteria. Try reduced weights, direction oversampling, or a gripper "
            "classification head before rollout smoke. ACT-lite remains later."
        )

    assessment_text = "\n".join(
        [
            f"{label}: "
            f"phase1={str(values['improves_phase_1']).lower()}, "
            f"direction_gap={str(values['improves_direction_gap']).lower()}, "
            f"phase4_5={str(values['preserves_phase_4_5']).lower()}, "
            f"best_loss_tol={str(values['keeps_best_val_loss_within_tolerance']).lower()}, "
            f"action6={str(values['improves_action_dim_6']).lower()}"
            for label, values in assessments.items()
        ]
    )

    return f"""# Stage 20-M Balanced LSTM Report

## Dataset Summary

```text
num_windows = {dataset_summary.get('num_windows')}
train_windows = {dataset_summary.get('train_windows')}
val_windows = {dataset_summary.get('val_windows')}
direction_counts = {_counts(dataset_summary.get('direction_counts'))}
phase_counts = {_counts(dataset_summary.get('phase_counts'))}
teacher_profile_counts = {_counts(dataset_summary.get('teacher_profile_counts'))}
```

## Metrics

| Run | Train Loss | Final Val Loss | Best Val Loss | Best Epoch | Phase 1 | Phase 3 | Phase 4 | Phase 5 | Direction 0 | Direction 1 | Direction Gap | Action Dim 6 |
|-----|-----------:|---------------:|--------------:|-----------:|--------:|--------:|--------:|--------:|------------:|------------:|--------------:|-------------:|
{chr(10).join(_row(label, metrics) for label, metrics in runs)}

## Assessment

```text
baseline_phase_1 = {_fmt(_phase_loss(baseline_metrics, '1'))}
baseline_direction_gap = {_fmt(_direction_gap(baseline_metrics))}
baseline_action_dim_6 = {_fmt(_action6(baseline_metrics))}
best_loss_tolerance = {_fmt(BEST_TOLERANCE)}

{assessment_text}
best_by_best_val_loss = {best_label}
```

## Interpretation

- Validation metrics are unweighted for every run.
- `best.pt` selection still uses unweighted best_val_loss.
- Phase 4/5 preservation means both release and retreat losses remain finite and below {LOW_PHASE45_THRESHOLD}.
- Direction gap should be monitored separately from global best_val_loss.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-M balanced LSTM report")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--phase_balanced_metrics", required=True)
    parser.add_argument("--phase_direction_balanced_metrics", required=True)
    parser.add_argument("--weighted_loss_metrics", default=None)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        baseline_metrics=_load(args.baseline_metrics) or {},
        phase_balanced_metrics=_load(args.phase_balanced_metrics) or {},
        phase_direction_balanced_metrics=_load(args.phase_direction_balanced_metrics) or {},
        weighted_loss_metrics=_load(args.weighted_loss_metrics),
        dataset_summary=_load(args.dataset_summary) or {},
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
