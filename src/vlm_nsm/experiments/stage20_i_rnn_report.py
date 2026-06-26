"""Generate Stage 20-I GRU/LSTM report against MLP baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MLP_PHASE1_BASELINE = 0.0083287518
MLP_PHASE3_BASELINE = 0.0023744537
MLP_BEST_BASELINE = 0.0028632263


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _loss(metrics: dict[str, Any], key: str) -> Any:
    if key == "final_val_loss":
        return metrics.get("final_val_loss", metrics.get("val_loss"))
    return metrics.get(key)


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _action_dim(values: list[Any] | None) -> str:
    if not values:
        return "N/A"
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def _phase_loss(metrics: dict[str, Any], phase: str) -> float | None:
    losses = metrics.get("per_phase_val_loss") or {}
    if phase not in losses:
        return None
    return float(losses[phase])


def _direction_gap(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    if len(losses) < 2:
        return None
    vals = [float(v) for v in losses.values()]
    return max(vals) - min(vals)


def build_report(
    *,
    mlp_metrics: dict[str, Any],
    rnn_metrics: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> str:
    mlp_best = float(mlp_metrics.get("best_val_loss", MLP_BEST_BASELINE))
    rnn_best = float(rnn_metrics.get("best_val_loss", float("inf")))
    rnn_beats_mlp = rnn_best < mlp_best

    rnn_phase1 = _phase_loss(rnn_metrics, "1")
    rnn_phase3 = _phase_loss(rnn_metrics, "3")
    phase1_improved = rnn_phase1 is not None and rnn_phase1 < MLP_PHASE1_BASELINE
    phase3_improved = rnn_phase3 is not None and rnn_phase3 < MLP_PHASE3_BASELINE

    mlp_action = mlp_metrics.get("action_dim_mse") or []
    rnn_action = rnn_metrics.get("action_dim_mse") or []
    action6_improved = (
        len(mlp_action) > 6
        and len(rnn_action) > 6
        and float(rnn_action[6]) < float(mlp_action[6])
    )
    mlp_gap = _direction_gap(mlp_metrics)
    rnn_gap = _direction_gap(rnn_metrics)
    direction_gap_narrows = mlp_gap is not None and rnn_gap is not None and rnn_gap < mlp_gap

    strong_count = sum(
        [
            rnn_beats_mlp,
            phase1_improved,
            phase3_improved,
            action6_improved,
            direction_gap_narrows,
        ]
    )
    if strong_count >= 2:
        recommendation = (
            "Stage 20-I reaches STRONG_PASS. Next train an LSTM variant and then consider a small rollout smoke test, "
            "while keeping phase labeling patch as a prerequisite before ACT-lite."
        )
    elif rnn_best < float(mlp_metrics.get("best_val_loss", float("inf"))) * 1.25:
        recommendation = (
            "Stage 20-I reaches PASS. Try LSTM and a phase-balanced sampler before rollout smoke; "
            "patch phase 4/5 labeling before ACT-lite."
        )
    else:
        recommendation = (
            "Do not move to rollout smoke yet. Inspect sequence feature scaling, try phase-balanced sampling, "
            "and patch phase labeling."
        )

    return f"""# Stage 20-I Goal-Conditioned RNN Report

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

- Phase 4/5 are absent, so release/retreat skill learning is not proven.
- Phase 3 is composite placement from the current place state machine.
- This is offline supervised validation only; no closed-loop success claim is made.

## MLP vs RNN

| Model | Train Loss | Final Val Loss | Best Val Loss | Best Epoch |
|-------|-----------:|---------------:|--------------:|-----------:|
| goal-conditioned MLP combined | {_fmt(mlp_metrics.get('train_loss'))} | {_fmt(_loss(mlp_metrics, 'final_val_loss'))} | {_fmt(mlp_metrics.get('best_val_loss'))} | {mlp_metrics.get('best_epoch')} |
| {rnn_metrics.get('model_type', 'rnn')} combined | {_fmt(rnn_metrics.get('train_loss'))} | {_fmt(_loss(rnn_metrics, 'final_val_loss'))} | {_fmt(rnn_metrics.get('best_val_loss'))} | {rnn_metrics.get('best_epoch')} |

```text
rnn_beats_mlp_best_val_loss = {str(rnn_beats_mlp).lower()}
mlp_best_val_loss = {_fmt(mlp_best)}
rnn_best_val_loss = {_fmt(rnn_best)}
baseline_to_beat = {_fmt(MLP_BEST_BASELINE)}
```

## Per-Direction Validation Loss

```text
MLP: {_losses(mlp_metrics.get('per_direction_val_loss'))}
RNN: {_losses(rnn_metrics.get('per_direction_val_loss'))}
mlp_direction_gap = {_fmt(mlp_gap)}
rnn_direction_gap = {_fmt(rnn_gap)}
direction_gap_narrows = {str(direction_gap_narrows).lower()}
```

## Per-Phase Validation Loss

```text
MLP: {_losses(mlp_metrics.get('per_phase_val_loss'))}
RNN: {_losses(rnn_metrics.get('per_phase_val_loss'))}
phase_1_improves_below_0.0083287518 = {str(phase1_improved).lower()}
phase_3_improves_below_0.0023744537 = {str(phase3_improved).lower()}
```

## Action-Dim MSE

```text
MLP: {_action_dim(mlp_metrics.get('action_dim_mse'))}
RNN: {_action_dim(rnn_metrics.get('action_dim_mse'))}
action_dim_6_improves = {str(action6_improved).lower()}
```

## Interpretation

- GRU/LSTM sequence modeling is evaluated against the Stage 20-H goal-conditioned MLP best checkpoint.
- The comparison uses best_val_loss for model selection and reports final_val_loss separately.
- Phase 1 grasp and phase 3 composite placement remain the key phases to inspect.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-I RNN report")
    parser.add_argument("--mlp_metrics", required=True)
    parser.add_argument("--rnn_metrics", required=True)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        mlp_metrics=_load(args.mlp_metrics),
        rnn_metrics=_load(args.rnn_metrics),
        dataset_summary=_load(args.dataset_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
