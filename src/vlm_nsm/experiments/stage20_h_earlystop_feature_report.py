"""Stage 20-H report: early stopping, feature scaling, and combined dataset check."""

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
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _losses(losses: dict[str, Any] | None) -> str:
    if not losses:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in sorted(losses.items(), key=lambda item: str(item[0])))


def _improvement(obs: dict[str, Any], goal: dict[str, Any], key: str) -> dict[str, Any]:
    obs_val = float(obs.get(key, float("inf")))
    goal_val = float(goal.get(key, float("inf")))
    delta = obs_val - goal_val
    return {
        "improved": goal_val < obs_val,
        "delta": delta,
        "relative": delta / obs_val if obs_val not in (0.0, float("inf")) else 0.0,
    }


def _phase45(summary: dict[str, Any]) -> bool:
    phases = {str(k) for k in (summary.get("phase_counts") or {})}
    return "4" in phases or "5" in phases


def _feature_audit(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section, values in sorted(stats.items()):
        mean = values.get("mean") or []
        std = values.get("std") or []
        std_float = [float(x) for x in std]
        mean_float = [float(x) for x in mean]
        out[section] = {
            "dim": len(mean_float),
            "mean_abs_max": max((abs(x) for x in mean_float), default=0.0),
            "std_min": min(std_float, default=0.0),
            "std_max": max(std_float, default=0.0),
            "std_near_zero_count": sum(1 for x in std_float if x <= 1e-5),
        }
    return out


def _feature_audit_text(label: str, audit: dict[str, dict[str, Any]]) -> str:
    lines = [f"{label}:"]
    for section, values in audit.items():
        lines.append(
            f"  {section}: dim={values['dim']}, "
            f"mean_abs_max={_fmt(values['mean_abs_max'])}, "
            f"std_min={_fmt(values['std_min'])}, "
            f"std_max={_fmt(values['std_max'])}, "
            f"std_near_zero_count={values['std_near_zero_count']}"
        )
    return "\n".join(lines)


def build_report(
    *,
    metrics: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    normalization_stats: dict[str, dict[str, Any]],
) -> str:
    final_improvements = {
        label: _improvement(metrics[f"{label}_obs"], metrics[f"{label}_goal"], "val_loss")
        for label in ("v01", "v02", "combined")
    }
    best_improvements = {
        label: _improvement(metrics[f"{label}_obs"], metrics[f"{label}_goal"], "best_val_loss")
        for label in ("v01", "v02", "combined")
    }
    combined_best_ok = bool(best_improvements["combined"]["improved"])
    combined_final_ok = bool(final_improvements["combined"]["improved"])
    phase45 = {label: _phase45(summaries[label]) for label in ("v01", "v02", "combined")}
    audits = {label: _feature_audit(stats) for label, stats in normalization_stats.items()}

    if combined_final_ok:
        recommendation = (
            "Proceed to Stage 20-I LSTM/GRU on the combined dataset, using best-checkpoint "
            "selection and keeping phase 4/5 caveat explicit."
        )
    elif combined_best_ok:
        recommendation = (
            "Run Stage 20-I LSTM/GRU only with strict early stopping by best_val_loss; "
            "also inspect final-epoch drift before ACT."
        )
    else:
        recommendation = (
            "Do not move to recurrent models yet; inspect feature scaling/masks and consider "
            "phase-label patching or additional teacher demos."
        )

    return f"""# Stage 20-H Early-Stopping, Feature Scaling, and Combined Dataset Check

## Dataset Summary

| Dataset | Trajectories | Windows | Train | Val | Teacher Profiles | Phase 4/5 |
|---------|-------------:|--------:|------:|----:|------------------|-----------|
| v0_1 | {summaries['v01'].get('num_trajectories_loaded')} | {summaries['v01'].get('num_windows')} | {summaries['v01'].get('train_windows')} | {summaries['v01'].get('val_windows')} | {_counts(summaries['v01'].get('teacher_profile_counts'))} | {phase45['v01']} |
| v0_2 | {summaries['v02'].get('num_trajectories_loaded')} | {summaries['v02'].get('num_windows')} | {summaries['v02'].get('train_windows')} | {summaries['v02'].get('val_windows')} | {_counts(summaries['v02'].get('teacher_profile_counts'))} | {phase45['v02']} |
| combined | {summaries['combined'].get('num_trajectories_loaded')} | {summaries['combined'].get('num_windows')} | {summaries['combined'].get('train_windows')} | {summaries['combined'].get('val_windows')} | {_counts(summaries['combined'].get('teacher_profile_counts'))} | {phase45['combined']} |

Direction counts:

```text
v0_1: {_counts(summaries['v01'].get('direction_counts'))}
v0_2: {_counts(summaries['v02'].get('direction_counts'))}
combined: {_counts(summaries['combined'].get('direction_counts'))}
```

Phase counts:

```text
v0_1: {_counts(summaries['v01'].get('phase_counts'))}
v0_2: {_counts(summaries['v02'].get('phase_counts'))}
combined: {_counts(summaries['combined'].get('phase_counts'))}
```

## MLP Metrics

| Dataset | Feature Mode | Train Loss | Final Val Loss | Best Val Loss | Best Epoch |
|---------|--------------|-----------:|---------------:|--------------:|-----------:|
| v0_1 | obs_only | {_fmt(metrics['v01_obs'].get('train_loss'))} | {_fmt(metrics['v01_obs'].get('val_loss'))} | {_fmt(metrics['v01_obs'].get('best_val_loss'))} | {metrics['v01_obs'].get('best_epoch')} |
| v0_1 | goal_conditioned | {_fmt(metrics['v01_goal'].get('train_loss'))} | {_fmt(metrics['v01_goal'].get('val_loss'))} | {_fmt(metrics['v01_goal'].get('best_val_loss'))} | {metrics['v01_goal'].get('best_epoch')} |
| v0_2 | obs_only | {_fmt(metrics['v02_obs'].get('train_loss'))} | {_fmt(metrics['v02_obs'].get('val_loss'))} | {_fmt(metrics['v02_obs'].get('best_val_loss'))} | {metrics['v02_obs'].get('best_epoch')} |
| v0_2 | goal_conditioned | {_fmt(metrics['v02_goal'].get('train_loss'))} | {_fmt(metrics['v02_goal'].get('val_loss'))} | {_fmt(metrics['v02_goal'].get('best_val_loss'))} | {metrics['v02_goal'].get('best_epoch')} |
| combined | obs_only | {_fmt(metrics['combined_obs'].get('train_loss'))} | {_fmt(metrics['combined_obs'].get('val_loss'))} | {_fmt(metrics['combined_obs'].get('best_val_loss'))} | {metrics['combined_obs'].get('best_epoch')} |
| combined | goal_conditioned | {_fmt(metrics['combined_goal'].get('train_loss'))} | {_fmt(metrics['combined_goal'].get('val_loss'))} | {_fmt(metrics['combined_goal'].get('best_val_loss'))} | {metrics['combined_goal'].get('best_epoch')} |

## Early-Stopping Decision

```text
v0_1_final_goal_improves = {str(final_improvements['v01']['improved']).lower()}
v0_1_best_goal_improves = {str(best_improvements['v01']['improved']).lower()}
v0_2_final_goal_improves = {str(final_improvements['v02']['improved']).lower()}
v0_2_best_goal_improves = {str(best_improvements['v02']['improved']).lower()}
combined_final_goal_improves = {str(final_improvements['combined']['improved']).lower()}
combined_best_goal_improves = {str(best_improvements['combined']['improved']).lower()}
```

Relative deltas, goal-conditioned vs obs-only:

```text
v0_1 final={_fmt(float(final_improvements['v01']['relative']) * 100)}%, best={_fmt(float(best_improvements['v01']['relative']) * 100)}%
v0_2 final={_fmt(float(final_improvements['v02']['relative']) * 100)}%, best={_fmt(float(best_improvements['v02']['relative']) * 100)}%
combined final={_fmt(float(final_improvements['combined']['relative']) * 100)}%, best={_fmt(float(best_improvements['combined']['relative']) * 100)}%
```

## Per-Direction Validation Loss

```text
v0_1 obs: {_losses(metrics['v01_obs'].get('per_direction_val_loss'))}
v0_1 goal: {_losses(metrics['v01_goal'].get('per_direction_val_loss'))}
v0_2 obs: {_losses(metrics['v02_obs'].get('per_direction_val_loss'))}
v0_2 goal: {_losses(metrics['v02_goal'].get('per_direction_val_loss'))}
combined obs: {_losses(metrics['combined_obs'].get('per_direction_val_loss'))}
combined goal: {_losses(metrics['combined_goal'].get('per_direction_val_loss'))}
```

## Per-Phase Validation Loss

```text
v0_1 obs: {_losses(metrics['v01_obs'].get('per_phase_val_loss'))}
v0_1 goal: {_losses(metrics['v01_goal'].get('per_phase_val_loss'))}
v0_2 obs: {_losses(metrics['v02_obs'].get('per_phase_val_loss'))}
v0_2 goal: {_losses(metrics['v02_goal'].get('per_phase_val_loss'))}
combined obs: {_losses(metrics['combined_obs'].get('per_phase_val_loss'))}
combined goal: {_losses(metrics['combined_goal'].get('per_phase_val_loss'))}
```

## Feature Scaling Audit

```text
{_feature_audit_text('v0_1', audits['v01'])}

{_feature_audit_text('v0_2', audits['v02'])}

{_feature_audit_text('combined', audits['combined'])}
```

## Interpretation

- Use `best_val_loss` for model selection; final epoch can drift after the best checkpoint.
- Masks and phase one-hot are not normalized; continuous tensors are normalized from dataset-specific stats.
- Phase 4/5 are absent in all current datasets, so release/retreat learning remains unproven.
- Phase 1 grasp and phase 3 composite placement remain the key offline bottlenecks.

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-H early-stop and feature report")
    parser.add_argument("--v01_obs_metrics", required=True)
    parser.add_argument("--v01_goal_metrics", required=True)
    parser.add_argument("--v02_obs_metrics", required=True)
    parser.add_argument("--v02_goal_metrics", required=True)
    parser.add_argument("--combined_obs_metrics", required=True)
    parser.add_argument("--combined_goal_metrics", required=True)
    parser.add_argument("--v01_summary", required=True)
    parser.add_argument("--v02_summary", required=True)
    parser.add_argument("--combined_summary", required=True)
    parser.add_argument("--v01_norm", required=True)
    parser.add_argument("--v02_norm", required=True)
    parser.add_argument("--combined_norm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        metrics={
            "v01_obs": _load(args.v01_obs_metrics),
            "v01_goal": _load(args.v01_goal_metrics),
            "v02_obs": _load(args.v02_obs_metrics),
            "v02_goal": _load(args.v02_goal_metrics),
            "combined_obs": _load(args.combined_obs_metrics),
            "combined_goal": _load(args.combined_goal_metrics),
        },
        summaries={
            "v01": _load(args.v01_summary),
            "v02": _load(args.v02_summary),
            "combined": _load(args.combined_summary),
        },
        normalization_stats={
            "v01": _load(args.v01_norm),
            "v02": _load(args.v02_norm),
            "combined": _load(args.combined_norm),
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
