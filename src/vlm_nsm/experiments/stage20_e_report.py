"""Generate Stage 20-E goal-conditioned MLP comparison report."""

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


def _dict_table(values: dict[str, Any]) -> str:
    if not values:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(values.items(), key=lambda item: str(item[0])))


def build_report(obs_metrics: dict[str, Any], goal_metrics: dict[str, Any], dataset_summary: dict[str, Any]) -> str:
    obs_val = float(obs_metrics.get("val_loss", float("inf")))
    goal_val = float(goal_metrics.get("val_loss", float("inf")))
    improves = goal_val < obs_val
    if improves:
        recommendation = "Goal-conditioned features improved validation loss; proceed to LSTM/GRU or rollout-smoke preparation after broader teacher collection."
    else:
        recommendation = "Goal-conditioned features did not improve validation loss; inspect feature scaling and consider teacher collection v0_2 before recurrent models."

    return f"""# Stage 20-E Goal-Conditioned MLP Baseline Report

## Dataset

```text
num_windows = {dataset_summary.get('num_windows')}
train_windows = {dataset_summary.get('train_windows')}
val_windows = {dataset_summary.get('val_windows')}
obs_dim = {dataset_summary.get('obs_dim')}
action_dim = {dataset_summary.get('action_dim')}
role_state_dim = {dataset_summary.get('role_state_dim')}
goal_delta_dim = {dataset_summary.get('goal_delta_dim')}
reached_goal_delta_dim = {dataset_summary.get('reached_goal_delta_dim')}
history_len = {dataset_summary.get('history_len')}
can_train_mlp_baseline = {dataset_summary.get('can_train_mlp_baseline')}
recommendation = {dataset_summary.get('recommendation')}
```

## Caveats

- Dataset v0_1 uses existing mixed-teacher Stage 19 data, not new Stage 20 `cube2_robust` teacher collection.
- Dry-run data contains phases 0-3 only; do not claim release/retreat coverage.
- This is offline supervised BC validation of the role-conditioned feature pipeline, not a final controller.

## Comparison

| Model | Train Loss | Val Loss | Best Epoch | Input Dim |
|-------|-----------:|---------:|-----------:|----------:|
| obs_only | {_fmt(obs_metrics.get('train_loss'))} | {_fmt(obs_metrics.get('val_loss'))} | {obs_metrics.get('best_epoch')} | {obs_metrics.get('input_dim')} |
| goal_conditioned | {_fmt(goal_metrics.get('train_loss'))} | {_fmt(goal_metrics.get('val_loss'))} | {goal_metrics.get('best_epoch')} | {goal_metrics.get('input_dim')} |

## Per-Direction Validation Loss

| Model | Losses |
|-------|--------|
| obs_only | {_dict_table(obs_metrics.get('per_direction_val_loss', {}))} |
| goal_conditioned | {_dict_table(goal_metrics.get('per_direction_val_loss', {}))} |

## Per-Phase Validation Loss

| Model | Losses |
|-------|--------|
| obs_only | {_dict_table(obs_metrics.get('per_phase_val_loss', {}))} |
| goal_conditioned | {_dict_table(goal_metrics.get('per_phase_val_loss', {}))} |

## Result

```text
goal_conditioned_improves_val_loss = {str(improves).lower()}
obs_only_val_loss = {_fmt(obs_val)}
goal_conditioned_val_loss = {_fmt(goal_val)}
```

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-E MLP report")
    parser.add_argument("--obs_only_metrics", required=True)
    parser.add_argument("--goal_metrics", required=True)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(_load(args.obs_only_metrics), _load(args.goal_metrics), _load(args.dataset_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
