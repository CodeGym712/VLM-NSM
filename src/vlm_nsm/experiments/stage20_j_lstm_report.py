"""Generate Stage 20-J LSTM-vs-GRU report."""

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


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _action(values: list[Any] | None) -> str:
    if not values:
        return "N/A"
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def _gap(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    if len(losses) < 2:
        return None
    vals = [float(v) for v in losses.values()]
    return max(vals) - min(vals)


def _phase(metrics: dict[str, Any], phase: str) -> float | None:
    losses = metrics.get("per_phase_val_loss") or {}
    return float(losses[phase]) if phase in losses else None


def build_report(
    *,
    mlp_metrics: dict[str, Any],
    gru_metrics: dict[str, Any],
    lstm_metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> str:
    mlp_best = float(mlp_metrics.get("best_val_loss", float("inf")))
    gru_best = float(gru_metrics.get("best_val_loss", float("inf")))
    lstm_best = float(lstm_metrics.get("best_val_loss", float("inf")))
    lstm_beats_mlp = lstm_best < mlp_best
    lstm_beats_gru = lstm_best < gru_best

    gru_phase1 = _phase(gru_metrics, "1")
    gru_phase3 = _phase(gru_metrics, "3")
    lstm_phase1 = _phase(lstm_metrics, "1")
    lstm_phase3 = _phase(lstm_metrics, "3")
    phase1_improves_gru = lstm_phase1 is not None and gru_phase1 is not None and lstm_phase1 < gru_phase1
    phase3_improves_gru = lstm_phase3 is not None and gru_phase3 is not None and lstm_phase3 < gru_phase3

    gru_action = gru_metrics.get("action_dim_mse") or []
    lstm_action = lstm_metrics.get("action_dim_mse") or []
    action6_improves_gru = len(gru_action) > 6 and len(lstm_action) > 6 and float(lstm_action[6]) < float(gru_action[6])
    gap_improves_gru = _gap(lstm_metrics) is not None and _gap(gru_metrics) is not None and _gap(lstm_metrics) < _gap(gru_metrics)

    strong_count = sum([lstm_beats_gru, phase1_improves_gru, phase3_improves_gru, action6_improves_gru, gap_improves_gru])
    if lstm_beats_gru and strong_count >= 2:
        verdict = "STRONG_PASS"
        recommendation = (
            "Use the better recurrent checkpoint as the offline sequence baseline, then run a small rollout smoke "
            "only after keeping phase 4/5 caveat explicit. Also patch phase labeling before ACT-lite."
        )
    elif lstm_beats_mlp:
        verdict = "PASS"
        recommendation = (
            "LSTM improves over MLP but does not clearly beat GRU. Keep GRU as the default recurrent baseline, "
            "then try phase-balanced sampling and phase-label patching."
        )
    else:
        verdict = "PARTIAL_PASS"
        recommendation = (
            "Keep GRU as the recurrent baseline. Inspect LSTM optimization, phase balance, and sequence features "
            "before further recurrent variants."
        )

    return f"""# Stage 20-J LSTM Variant Report

## Status

```text
Stage 20-J = {verdict}
```

## Dataset Summary

```text
num_trajectories_loaded = {dataset_summary.get('num_trajectories_loaded')}
num_windows = {dataset_summary.get('num_windows')}
train_windows = {dataset_summary.get('train_windows')}
val_windows = {dataset_summary.get('val_windows')}
direction_counts = {_counts(dataset_summary.get('direction_counts'))}
phase_counts = {_counts(dataset_summary.get('phase_counts'))}
teacher_profile_counts = {_counts(dataset_summary.get('teacher_profile_counts'))}
```

## Caveats

- Phase 4/5 are absent; release/retreat skill learning is still unproven.
- Phase 3 remains composite placement.
- This is offline supervised validation only, not closed-loop success.

## MLP vs GRU vs LSTM

| Model | Train Loss | Final Val Loss | Best Val Loss | Best Epoch |
|-------|-----------:|---------------:|--------------:|-----------:|
| goal-conditioned MLP | {_fmt(mlp_metrics.get('train_loss'))} | {_fmt(_final(mlp_metrics))} | {_fmt(mlp_metrics.get('best_val_loss'))} | {mlp_metrics.get('best_epoch')} |
| goal-conditioned GRU | {_fmt(gru_metrics.get('train_loss'))} | {_fmt(_final(gru_metrics))} | {_fmt(gru_metrics.get('best_val_loss'))} | {gru_metrics.get('best_epoch')} |
| goal-conditioned LSTM | {_fmt(lstm_metrics.get('train_loss'))} | {_fmt(_final(lstm_metrics))} | {_fmt(lstm_metrics.get('best_val_loss'))} | {lstm_metrics.get('best_epoch')} |

```text
lstm_beats_mlp_best_val_loss = {str(lstm_beats_mlp).lower()}
lstm_beats_gru_best_val_loss = {str(lstm_beats_gru).lower()}
mlp_best_val_loss = {_fmt(mlp_best)}
gru_best_val_loss = {_fmt(gru_best)}
lstm_best_val_loss = {_fmt(lstm_best)}
```

## Per-Direction Validation Loss

```text
MLP: {_losses(mlp_metrics.get('per_direction_val_loss'))}
GRU: {_losses(gru_metrics.get('per_direction_val_loss'))}
LSTM: {_losses(lstm_metrics.get('per_direction_val_loss'))}
gru_direction_gap = {_fmt(_gap(gru_metrics))}
lstm_direction_gap = {_fmt(_gap(lstm_metrics))}
lstm_direction_gap_improves_gru = {str(gap_improves_gru).lower()}
```

## Per-Phase Validation Loss

```text
MLP: {_losses(mlp_metrics.get('per_phase_val_loss'))}
GRU: {_losses(gru_metrics.get('per_phase_val_loss'))}
LSTM: {_losses(lstm_metrics.get('per_phase_val_loss'))}
lstm_phase_1_improves_gru = {str(phase1_improves_gru).lower()}
lstm_phase_3_improves_gru = {str(phase3_improves_gru).lower()}
```

## Action-Dim MSE

```text
MLP: {_action(mlp_metrics.get('action_dim_mse'))}
GRU: {_action(gru_metrics.get('action_dim_mse'))}
LSTM: {_action(lstm_metrics.get('action_dim_mse'))}
lstm_action_dim_6_improves_gru = {str(action6_improves_gru).lower()}
```

## Interpretation

- GRU remains the reference recurrent baseline from Stage 20-I.
- LSTM is evaluated with the same combined dataset and same goal-conditioned feature sequence.
- Model choice should still use best_val_loss, with phase 1 and phase 3 inspected separately.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-J LSTM report")
    parser.add_argument("--mlp_metrics", required=True)
    parser.add_argument("--gru_metrics", required=True)
    parser.add_argument("--lstm_metrics", required=True)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        mlp_metrics=_load(args.mlp_metrics),
        gru_metrics=_load(args.gru_metrics),
        lstm_metrics=_load(args.lstm_metrics),
        dataset_summary=_load(args.dataset_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
