"""Generate Stage 20-L report for phase-labeled LSTM retraining."""

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


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _phase(summary: dict[str, Any], phase: str) -> int:
    return int((summary.get("phase_counts") or {}).get(phase, 0))


def _phase_loss(metrics: dict[str, Any], phase: str) -> float | None:
    losses = metrics.get("per_phase_val_loss") or {}
    return float(losses[phase]) if phase in losses else None


def _gap(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    if len(losses) < 2:
        return None
    values = [float(v) for v in losses.values()]
    return max(values) - min(values)


def _action(values: list[Any] | None) -> str:
    if not values:
        return "N/A"
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def build_report(
    *,
    original_metrics: dict[str, Any],
    labeled_metrics: dict[str, Any],
    combined_lstm_metrics: dict[str, Any],
    original_summary: dict[str, Any],
    labeled_summary: dict[str, Any],
) -> str:
    original_best = float(original_metrics.get("best_val_loss", float("inf")))
    labeled_best = float(labeled_metrics.get("best_val_loss", float("inf")))
    combined_best = float(combined_lstm_metrics.get("best_val_loss", float("inf")))
    labeled_beats_original = labeled_best < original_best
    labeled_beats_combined = labeled_best < combined_best
    phase45_present = _phase(labeled_summary, "4") > 0 and _phase(labeled_summary, "5") > 0
    release_loss = _phase_loss(labeled_metrics, "4")
    retreat_loss = _phase_loss(labeled_metrics, "5")
    release_retreat_finite = release_loss is not None and retreat_loss is not None
    phase1_improves = (
        _phase_loss(labeled_metrics, "1") is not None
        and _phase_loss(original_metrics, "1") is not None
        and _phase_loss(labeled_metrics, "1") < _phase_loss(original_metrics, "1")
    )
    phase3_improves = (
        _phase_loss(labeled_metrics, "3") is not None
        and _phase_loss(original_metrics, "3") is not None
        and _phase_loss(labeled_metrics, "3") < _phase_loss(original_metrics, "3")
    )
    gap_improves = _gap(labeled_metrics) is not None and _gap(original_metrics) is not None and _gap(labeled_metrics) < _gap(original_metrics)

    strong_count = sum([labeled_beats_original, release_retreat_finite, phase1_improves, phase3_improves, gap_improves])
    if labeled_beats_original and release_retreat_finite and strong_count >= 3:
        status = "STRONG_PASS"
        recommendation = (
            "Use the phase-labeled dataset for the next recurrent baseline, then implement a phase-balanced sampler "
            "focused on phases 1, 4, and 5 before any rollout smoke."
        )
    elif release_retreat_finite:
        status = "PASS"
        recommendation = (
            "Keep the phase-labeled dataset because release/retreat are now represented. Next try a phase-balanced "
            "sampler; use best_val_loss and per-phase losses to decide between original and labeled checkpoints."
        )
    else:
        status = "PARTIAL_PASS"
        recommendation = (
            "The relabeled dataset did not produce usable phase 4/5 losses. Inspect relabeling and dataset splits "
            "before sampler work."
        )

    return f"""# Stage 20-L Phase-Labeled LSTM Report

## Status

```text
Stage 20-L = {status}
```

## Dataset Comparison

| Dataset | Windows | Train | Val | Phase 4 | Phase 5 |
|---------|--------:|------:|----:|--------:|--------:|
| v0_2 original | {original_summary.get('num_windows')} | {original_summary.get('train_windows')} | {original_summary.get('val_windows')} | {_phase(original_summary, '4')} | {_phase(original_summary, '5')} |
| v0_2 phase_labeled | {labeled_summary.get('num_windows')} | {labeled_summary.get('train_windows')} | {labeled_summary.get('val_windows')} | {_phase(labeled_summary, '4')} | {_phase(labeled_summary, '5')} |

Direction counts:

```text
original: {_counts(original_summary.get('direction_counts'))}
labeled: {_counts(labeled_summary.get('direction_counts'))}
```

Phase counts:

```text
original: {_counts(original_summary.get('phase_counts'))}
labeled: {_counts(labeled_summary.get('phase_counts'))}
phase_4_5_present = {str(phase45_present).lower()}
```

## LSTM Metrics

| Model | Train Loss | Final Val Loss | Best Val Loss | Best Epoch |
|-------|-----------:|---------------:|--------------:|-----------:|
| v0_2 original LSTM | {_fmt(original_metrics.get('train_loss'))} | {_fmt(_final(original_metrics))} | {_fmt(original_metrics.get('best_val_loss'))} | {original_metrics.get('best_epoch')} |
| v0_2 phase_labeled LSTM | {_fmt(labeled_metrics.get('train_loss'))} | {_fmt(_final(labeled_metrics))} | {_fmt(labeled_metrics.get('best_val_loss'))} | {labeled_metrics.get('best_epoch')} |
| combined LSTM context | {_fmt(combined_lstm_metrics.get('train_loss'))} | {_fmt(_final(combined_lstm_metrics))} | {_fmt(combined_lstm_metrics.get('best_val_loss'))} | {combined_lstm_metrics.get('best_epoch')} |

```text
labeled_beats_original_best_val_loss = {str(labeled_beats_original).lower()}
labeled_beats_combined_best_val_loss = {str(labeled_beats_combined).lower()}
original_best_val_loss = {_fmt(original_best)}
labeled_best_val_loss = {_fmt(labeled_best)}
combined_best_val_loss = {_fmt(combined_best)}
```

## Per-Direction Validation Loss

```text
original: {_losses(original_metrics.get('per_direction_val_loss'))}
labeled: {_losses(labeled_metrics.get('per_direction_val_loss'))}
combined_context: {_losses(combined_lstm_metrics.get('per_direction_val_loss'))}
original_direction_gap = {_fmt(_gap(original_metrics))}
labeled_direction_gap = {_fmt(_gap(labeled_metrics))}
labeled_direction_gap_improves_original = {str(gap_improves).lower()}
```

## Per-Phase Validation Loss

```text
original: {_losses(original_metrics.get('per_phase_val_loss'))}
labeled: {_losses(labeled_metrics.get('per_phase_val_loss'))}
combined_context: {_losses(combined_lstm_metrics.get('per_phase_val_loss'))}
release_phase_4_loss = {_fmt(release_loss)}
retreat_phase_5_loss = {_fmt(retreat_loss)}
release_retreat_losses_available = {str(release_retreat_finite).lower()}
phase_1_improves_original = {str(phase1_improves).lower()}
phase_3_improves_original = {str(phase3_improves).lower()}
```

## Action-Dim MSE

```text
original: {_action(original_metrics.get('action_dim_mse'))}
labeled: {_action(labeled_metrics.get('action_dim_mse'))}
combined_context: {_action(combined_lstm_metrics.get('action_dim_mse'))}
```

## Interpretation

- The original v0_2 dataset cannot evaluate release/retreat separately because phase 4/5 are absent.
- The phase-labeled dataset exposes phase 4/5 losses, so release/retreat representation can now be audited.
- This is still offline supervised validation; no closed-loop control claim is made.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-L phase-labeled LSTM report")
    parser.add_argument("--original_metrics", required=True)
    parser.add_argument("--labeled_metrics", required=True)
    parser.add_argument("--combined_lstm_metrics", required=True)
    parser.add_argument("--original_summary", required=True)
    parser.add_argument("--labeled_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        original_metrics=_load(args.original_metrics),
        labeled_metrics=_load(args.labeled_metrics),
        combined_lstm_metrics=_load(args.combined_lstm_metrics),
        original_summary=_load(args.original_summary),
        labeled_summary=_load(args.labeled_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
