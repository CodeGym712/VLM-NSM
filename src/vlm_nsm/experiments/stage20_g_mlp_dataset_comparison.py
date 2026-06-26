"""Compare Stage 20 MLP baselines across v0_1 and v0_2 datasets."""

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


def _counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))


def _losses(values: dict[str, Any] | None) -> str:
    if not values:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(values.items(), key=lambda x: str(x[0])))


def _improvement(obs: dict[str, Any], goal: dict[str, Any]) -> dict[str, float | bool]:
    obs_val = float(obs.get("val_loss", float("inf")))
    goal_val = float(goal.get("val_loss", float("inf")))
    abs_delta = obs_val - goal_val
    rel_delta = abs_delta / obs_val if obs_val and obs_val != float("inf") else 0.0
    return {
        "improved": goal_val < obs_val,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }


def _best_improvement(obs: dict[str, Any], goal: dict[str, Any]) -> dict[str, float | bool]:
    obs_val = float(obs.get("best_val_loss", float("inf")))
    goal_val = float(goal.get("best_val_loss", float("inf")))
    abs_delta = obs_val - goal_val
    rel_delta = abs_delta / obs_val if obs_val and obs_val != float("inf") else 0.0
    return {
        "improved": goal_val < obs_val,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }


def _direction_gap(metrics: dict[str, Any]) -> float | None:
    losses = metrics.get("per_direction_val_loss") or {}
    if not losses:
        return None
    vals = [float(v) for v in losses.values()]
    return max(vals) - min(vals) if vals else None


def _action_dim_mse(metrics: dict[str, Any]) -> str:
    vals = metrics.get("action_dim_mse") or []
    if not vals:
        return "N/A"
    return "[" + ", ".join(_fmt(v) for v in vals) + "]"


def _phase_45_present(summary: dict[str, Any]) -> bool:
    phases = {str(k) for k in (summary.get("phase_counts") or {})}
    return "4" in phases or "5" in phases


def build_report(
    *,
    v01_obs: dict[str, Any],
    v01_goal: dict[str, Any],
    v02_obs: dict[str, Any],
    v02_goal: dict[str, Any],
    v01_summary: dict[str, Any],
    v02_summary: dict[str, Any],
    collection_summary: dict[str, Any],
) -> str:
    imp_v01 = _improvement(v01_obs, v01_goal)
    imp_v02 = _improvement(v02_obs, v02_goal)
    best_imp_v01 = _best_improvement(v01_obs, v01_goal)
    best_imp_v02 = _best_improvement(v02_obs, v02_goal)
    v02_goal_better = bool(imp_v02["improved"])
    v02_goal_best_better = bool(best_imp_v02["improved"])
    v02_phase45 = _phase_45_present(v02_summary)
    v02_direction_gap = _direction_gap(v02_goal)

    if v02_goal_better and not v02_phase45:
        recommendation = (
            "Proceed to LSTM/GRU on v0_2 for phases 0-3, and patch phase labeling before "
            "claiming release/retreat skill learning or moving to ACT."
        )
    elif v02_goal_better:
        recommendation = "Proceed to LSTM/GRU on v0_2; goal-conditioned features remain beneficial."
    elif v02_goal_best_better:
        recommendation = (
            "Goal-conditioned v0_2 has the better best checkpoint but not the better final epoch. "
            "Before LSTM/GRU, run a small Stage 20-H check on early stopping, feature scaling, "
            "and the combined v0_1_plus_v0_2 dataset."
        )
    else:
        recommendation = (
            "Do not move to recurrent models yet; inspect v0_2 feature scaling, train on the combined "
            "dataset, or collect more demos."
        )

    return f"""# Stage 20-G MLP v0_1 vs v0_2 Dataset Comparison

## Dataset Comparison

| Metric | v0_1 | v0_2 |
|--------|-----:|-----:|
| trajectories loaded | {v01_summary.get('num_trajectories_loaded')} | {v02_summary.get('num_trajectories_loaded')} |
| windows | {v01_summary.get('num_windows')} | {v02_summary.get('num_windows')} |
| train windows | {v01_summary.get('train_windows')} | {v02_summary.get('train_windows')} |
| val windows | {v01_summary.get('val_windows')} | {v02_summary.get('val_windows')} |
| can train MLP | {v01_summary.get('can_train_mlp_baseline')} | {v02_summary.get('can_train_mlp_baseline')} |

Teacher profiles:

```text
v0_1: {_counts(v01_summary.get('teacher_profile_counts'))}
v0_2: {_counts(v02_summary.get('teacher_profile_counts'))}
```

Direction counts:

```text
v0_1: {_counts(v01_summary.get('direction_counts'))}
v0_2: {_counts(v02_summary.get('direction_counts'))}
```

Phase counts:

```text
v0_1: {_counts(v01_summary.get('phase_counts'))}
v0_2: {_counts(v02_summary.get('phase_counts'))}
```

Missing fields:

```text
v0_1: {_counts(v01_summary.get('missing_field_counts'))}
v0_2: {_counts(v02_summary.get('missing_field_counts'))}
```

Phase 4/5 caveat:

```text
v0_2_phase_4_or_5_present = {str(v02_phase45).lower()}
place_state_machine still records composite placement under phase 3.
```

Teacher collection v0_2:

```text
attempts = {collection_summary.get('total_episodes')}
successes = {collection_summary.get('task_success_count')}
success_rate = {collection_summary.get('task_success_rate')}
unsafe_leaks = {collection_summary.get('unsafe_goal_leak_count')}
runtime_errors = {collection_summary.get('controller_runtime_errors')}
```

## MLP Comparison

| Dataset | Feature Mode | Train Loss | Val Loss | Best Val Loss | Best Epoch |
|---------|--------------|-----------:|---------:|--------------:|-----------:|
| v0_1 | obs_only | {_fmt(v01_obs.get('train_loss'))} | {_fmt(v01_obs.get('val_loss'))} | {_fmt(v01_obs.get('best_val_loss'))} | {v01_obs.get('best_epoch')} |
| v0_1 | goal_conditioned | {_fmt(v01_goal.get('train_loss'))} | {_fmt(v01_goal.get('val_loss'))} | {_fmt(v01_goal.get('best_val_loss'))} | {v01_goal.get('best_epoch')} |
| v0_2 | obs_only | {_fmt(v02_obs.get('train_loss'))} | {_fmt(v02_obs.get('val_loss'))} | {_fmt(v02_obs.get('best_val_loss'))} | {v02_obs.get('best_epoch')} |
| v0_2 | goal_conditioned | {_fmt(v02_goal.get('train_loss'))} | {_fmt(v02_goal.get('val_loss'))} | {_fmt(v02_goal.get('best_val_loss'))} | {v02_goal.get('best_epoch')} |

## Relative Improvement

```text
v0_1_goal_conditioned_improved = {str(imp_v01['improved']).lower()}
v0_1_abs_val_loss_delta = {_fmt(imp_v01['abs_delta'])}
v0_1_relative_val_loss_delta = {_fmt(float(imp_v01['rel_delta']) * 100.0)}%
v0_1_goal_conditioned_best_improved = {str(best_imp_v01['improved']).lower()}
v0_1_abs_best_val_loss_delta = {_fmt(best_imp_v01['abs_delta'])}
v0_1_relative_best_val_loss_delta = {_fmt(float(best_imp_v01['rel_delta']) * 100.0)}%

v0_2_goal_conditioned_improved = {str(imp_v02['improved']).lower()}
v0_2_abs_val_loss_delta = {_fmt(imp_v02['abs_delta'])}
v0_2_relative_val_loss_delta = {_fmt(float(imp_v02['rel_delta']) * 100.0)}%
v0_2_goal_conditioned_best_improved = {str(best_imp_v02['improved']).lower()}
v0_2_abs_best_val_loss_delta = {_fmt(best_imp_v02['abs_delta'])}
v0_2_relative_best_val_loss_delta = {_fmt(float(best_imp_v02['rel_delta']) * 100.0)}%
```

## Per-Direction Validation Loss

| Dataset | Feature Mode | Losses |
|---------|--------------|--------|
| v0_1 | obs_only | {_losses(v01_obs.get('per_direction_val_loss'))} |
| v0_1 | goal_conditioned | {_losses(v01_goal.get('per_direction_val_loss'))} |
| v0_2 | obs_only | {_losses(v02_obs.get('per_direction_val_loss'))} |
| v0_2 | goal_conditioned | {_losses(v02_goal.get('per_direction_val_loss'))} |

Direction gap for v0_2 goal-conditioned:

```text
{_fmt(v02_direction_gap)}
```

## Per-Phase Validation Loss

| Dataset | Feature Mode | Losses |
|---------|--------------|--------|
| v0_1 | obs_only | {_losses(v01_obs.get('per_phase_val_loss'))} |
| v0_1 | goal_conditioned | {_losses(v01_goal.get('per_phase_val_loss'))} |
| v0_2 | obs_only | {_losses(v02_obs.get('per_phase_val_loss'))} |
| v0_2 | goal_conditioned | {_losses(v02_goal.get('per_phase_val_loss'))} |

## Action-Dim MSE

```text
v0_1 obs_only: {_action_dim_mse(v01_obs)}
v0_1 goal_conditioned: {_action_dim_mse(v01_goal)}
v0_2 obs_only: {_action_dim_mse(v02_obs)}
v0_2 goal_conditioned: {_action_dim_mse(v02_goal)}
```

## Interpretation

- v0_2 is larger and cleaner than v0_1: teacher profile is exclusively `cube2_robust`.
- Goal-conditioned features improve final v0_2 validation loss: `{str(v02_goal_better).lower()}`.
- Goal-conditioned features improve best v0_2 validation loss: `{str(v02_goal_best_better).lower()}`.
- Direction losses are finite; inspect the direction gap above before closed-loop claims.
- Phase 1 grasp and phase 3 composite placement should be watched because Stage 20-F failure taxonomy is still dominated by grasp/place issues.
- Phase 4/5 release/retreat are absent, so the current result does not prove full release/retreat skill learning.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-G MLP dataset comparison report")
    parser.add_argument("--v01_obs_metrics", required=True)
    parser.add_argument("--v01_goal_metrics", required=True)
    parser.add_argument("--v02_obs_metrics", required=True)
    parser.add_argument("--v02_goal_metrics", required=True)
    parser.add_argument("--v01_summary", required=True)
    parser.add_argument("--v02_summary", required=True)
    parser.add_argument("--collection_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        v01_obs=_load(args.v01_obs_metrics),
        v01_goal=_load(args.v01_goal_metrics),
        v02_obs=_load(args.v02_obs_metrics),
        v02_goal=_load(args.v02_goal_metrics),
        v01_summary=_load(args.v01_summary),
        v02_summary=_load(args.v02_summary),
        collection_summary=_load(args.collection_summary),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
