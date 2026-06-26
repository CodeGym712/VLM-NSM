"""Generate Stage 20-F teacher collection and dataset comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt_counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda item: str(item[0])))


def _success_by_goal(collection: dict[str, Any]) -> str:
    per_goal = collection.get("per_goal_success") or {}
    if not per_goal:
        return "N/A"
    parts = []
    for goal, stats in sorted(per_goal.items()):
        total = int(stats.get("total", 0) or 0)
        success = int(stats.get("success", 0) or 0)
        rate = success / total if total else 0.0
        parts.append(f"{goal}: {success}/{total} ({rate:.1%})")
    return ", ".join(parts)


def build_report(v01: dict[str, Any], v02: dict[str, Any], collection: dict[str, Any]) -> str:
    phases = {str(k) for k in (v02.get("phase_counts") or {}).keys()}
    has_release_retreat = "4" in phases or "5" in phases
    can_train = bool(v02.get("can_train_mlp_baseline"))
    unsafe = int(collection.get("unsafe_goal_leak_count", 0) or 0)
    runtime = int(collection.get("controller_runtime_errors", 0) or 0)
    if not can_train:
        recommendation = "Patch recorder or builder before training; v0_2 is not trainable."
    elif unsafe > 0 or runtime > 0:
        recommendation = "Fix unsafe/runtime issues before using v0_2 for policy learning."
    elif not has_release_retreat:
        recommendation = "v0_2 is trainable, but release/retreat phases are still absent; patch recorder or collect longer traces before LSTM/ACT."
    else:
        recommendation = "Train MLP/LSTM on v0_2; phase coverage includes release/retreat."

    return f"""# Stage 20-F Dataset Report

## Teacher Collection

```text
total_episodes = {collection.get('total_episodes')}
executed_episodes = {collection.get('executed_episodes')}
task_success_count = {collection.get('task_success_count')}
task_success_rate = {collection.get('task_success_rate')}
unsafe_goal_leak_count = {collection.get('unsafe_goal_leak_count')}
controller_runtime_errors = {collection.get('controller_runtime_errors')}
scripted_profile = {collection.get('scripted_profile')}
```

Per-goal success:

```text
{_success_by_goal(collection)}
```

Failure taxonomy:

```text
{_fmt_counts(collection.get('per_failure_reason_counts'))}
```

## v0_1 vs v0_2

| Metric | v0_1 | v0_2 |
|--------|-----:|-----:|
| trajectories loaded | {v01.get('num_trajectories_loaded')} | {v02.get('num_trajectories_loaded')} |
| trajectories used | {v01.get('num_trajectories_used')} | {v02.get('num_trajectories_used')} |
| windows | {v01.get('num_windows')} | {v02.get('num_windows')} |
| train windows | {v01.get('train_windows')} | {v02.get('train_windows')} |
| val windows | {v01.get('val_windows')} | {v02.get('val_windows')} |
| can train MLP | {v01.get('can_train_mlp_baseline')} | {v02.get('can_train_mlp_baseline')} |

Direction counts:

```text
v0_1: {_fmt_counts(v01.get('direction_counts'))}
v0_2: {_fmt_counts(v02.get('direction_counts'))}
```

Phase counts:

```text
v0_1: {_fmt_counts(v01.get('phase_counts'))}
v0_2: {_fmt_counts(v02.get('phase_counts'))}
```

Placement bin counts:

```text
v0_1: {_fmt_counts(v01.get('placement_bin_counts'))}
v0_2: {_fmt_counts(v02.get('placement_bin_counts'))}
```

Teacher profile counts:

```text
v0_1: {_fmt_counts(v01.get('teacher_profile_counts'))}
v0_2: {_fmt_counts(v02.get('teacher_profile_counts'))}
```

Missing field counts:

```text
v0_1: {_fmt_counts(v01.get('missing_field_counts'))}
v0_2: {_fmt_counts(v02.get('missing_field_counts'))}
```

## Phase Coverage

```text
phase_4_or_5_present = {str(has_release_retreat).lower()}
```

## Recommendation

{recommendation}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20-F dataset report")
    parser.add_argument("--v01_summary", required=True)
    parser.add_argument("--v02_summary", required=True)
    parser.add_argument("--collection_summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(_load(args.v01_summary), _load(args.v02_summary), _load(args.collection_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
