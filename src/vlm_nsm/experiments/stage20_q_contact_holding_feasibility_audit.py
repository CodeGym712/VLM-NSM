"""Audit feasibility of contact/holding auxiliary targets for Stage 20-Q.

This script does not create training labels. It reports whether direct fields
exist and how stable weak proxy signals look in the current role-state dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.data.stage20_role_conditioned_dataset import Stage20RoleConditionedDataset


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "p10": None, "median": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": _to_float(values.mean()),
        "std": _to_float(values.std()),
        "p10": _to_float(np.percentile(values, 10)),
        "median": _to_float(np.percentile(values, 50)),
        "p90": _to_float(np.percentile(values, 90)),
    }


def _ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return _to_float(np.mean(mask.astype(np.float64)))


def _count_dict(values: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _mean_by_group(values: np.ndarray, group: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in sorted(set(group.astype(int).tolist())):
        mask = group == key
        if mask.any():
            out[str(int(key))] = _to_float(values[mask].mean())
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Stage20RoleConditionedDataset(
        args.dataset_dir,
        split=args.split,
        feature_mode="goal_conditioned",
        return_sequence=True,
        normalize=False,
    )
    arrays = dataset.arrays
    idx = dataset.indices
    role = arrays["role_state_history"][idx, -1, :]
    reached = arrays["reached_goal_delta_history"][idx, -1, :]
    actions = arrays["actions"][idx]
    phase = np.asarray([int(record.get("phase_id", -1)) for record in dataset.records], dtype=np.int64)
    direction = np.asarray([int(record.get("direction_id", -1)) for record in dataset.records], dtype=np.int64)

    rel_gripper_top = np.linalg.norm(role[:, 10:13], axis=1)
    gripper_open = role[:, 3]
    action_dim6 = actions[:, 6]
    reached_top_z_delta = reached[:, 6]
    near_top = rel_gripper_top <= float(args.contact_distance_threshold)
    gripper_closed = gripper_open <= float(args.closed_gripper_threshold)
    commanded_close = action_dim6 > 0.0
    commanded_open = action_dim6 <= 0.0

    direct_fields = {
        "contact": [k for k in arrays if "contact" in k.lower()],
        "holding": [k for k in arrays if "hold" in k.lower() or "grasp" in k.lower()],
    }
    direct_label_available = bool(direct_fields["contact"] or direct_fields["holding"])

    contact_proxy = near_top
    holding_proxy = near_top & gripper_closed & np.isin(phase, [2, 3])
    release_proxy = near_top & commanded_open & (phase == 4)

    intended_holding_phase = np.isin(phase, [2, 3])
    non_holding_phase = ~intended_holding_phase
    holding_positive_rate = _ratio(holding_proxy[intended_holding_phase])
    holding_negative_rate = _ratio(holding_proxy[non_holding_phase])
    holding_separation = holding_positive_rate - holding_negative_rate
    contact_phase1_rate = _ratio(contact_proxy[phase == 1])
    contact_phase0_rate = _ratio(contact_proxy[phase == 0])
    contact_phase_separation = contact_phase1_rate - contact_phase0_rate
    phase_leak_risk = holding_positive_rate > 0.95 and holding_negative_rate < 0.10

    feasibility = "direct_label_available" if direct_label_available else "proxy_only"
    if not direct_label_available and holding_separation >= 0.75 and contact_phase_separation >= 0.50:
        recommendation = (
            "Current data supports a weak proxy audit, but not a canonical contact/holding training target. "
            "Patch the recorder to save explicit contact/holding or object-in-gripper fields before using this as a supervised auxiliary head."
        )
    else:
        recommendation = (
            "Do not train contact/holding auxiliary loss from the current fields. The proxy is not stable enough or direct labels are missing."
        )

    per_phase: dict[str, Any] = {}
    for p in sorted(set(phase.astype(int).tolist())):
        mask = phase == p
        per_phase[str(int(p))] = {
            "count": int(mask.sum()),
            "relative_gripper_to_top": _stats(rel_gripper_top[mask]),
            "gripper_open": _stats(gripper_open[mask]),
            "action_dim6": _stats(action_dim6[mask]),
            "reached_top_z_delta": _stats(reached_top_z_delta[mask]),
            "near_top_ratio": _ratio(near_top[mask]),
            "gripper_closed_ratio": _ratio(gripper_closed[mask]),
            "commanded_close_ratio": _ratio(commanded_close[mask]),
            "holding_proxy_ratio": _ratio(holding_proxy[mask]),
            "release_proxy_ratio": _ratio(release_proxy[mask]),
        }

    return {
        "stage": "stage20_q_contact_holding_feasibility",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "num_samples": int(len(dataset)),
        "thresholds": {
            "contact_distance_threshold": float(args.contact_distance_threshold),
            "closed_gripper_threshold": float(args.closed_gripper_threshold),
        },
        "direct_fields": direct_fields,
        "direct_label_available": direct_label_available,
        "proxy_only": not direct_label_available,
        "phase_counts": _count_dict(phase.tolist()),
        "direction_counts": _count_dict(direction.tolist()),
        "overall": {
            "relative_gripper_to_top": _stats(rel_gripper_top),
            "gripper_open": _stats(gripper_open),
            "action_dim6": _stats(action_dim6),
            "reached_top_z_delta": _stats(reached_top_z_delta),
            "near_top_ratio": _ratio(near_top),
            "gripper_closed_ratio": _ratio(gripper_closed),
            "commanded_close_ratio": _ratio(commanded_close),
            "holding_proxy_ratio": _ratio(holding_proxy),
            "release_proxy_ratio": _ratio(release_proxy),
        },
        "per_phase": per_phase,
        "per_direction_holding_proxy_ratio": _mean_by_group(holding_proxy.astype(np.float64), direction),
        "per_direction_near_top_ratio": _mean_by_group(near_top.astype(np.float64), direction),
        "feasibility": {
            "status": feasibility,
            "holding_positive_rate_in_phase_2_3": holding_positive_rate,
            "holding_negative_rate_outside_phase_2_3": holding_negative_rate,
            "holding_phase_separation": holding_separation,
            "contact_phase1_rate": contact_phase1_rate,
            "contact_phase0_rate": contact_phase0_rate,
            "contact_phase_separation": contact_phase_separation,
            "phase_leak_risk": phase_leak_risk,
            "recommended_for_training_now": bool(direct_label_available),
        },
        "recommendation": recommendation,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _phase_line(per_phase: dict[str, Any], phase: str) -> str:
    item = per_phase.get(phase, {})
    return (
        f"{phase}: count={item.get('count', 0)}, "
        f"near_top={_fmt(item.get('near_top_ratio'))}, "
        f"closed={_fmt(item.get('gripper_closed_ratio'))}, "
        f"close_cmd={_fmt(item.get('commanded_close_ratio'))}, "
        f"holding_proxy={_fmt(item.get('holding_proxy_ratio'))}, "
        f"release_proxy={_fmt(item.get('release_proxy_ratio'))}"
    )


def build_markdown_report(audit: dict[str, Any]) -> str:
    feasibility = audit["feasibility"]
    per_phase_lines = "\n".join(_phase_line(audit["per_phase"], str(p)) for p in sorted(int(k) for k in audit["per_phase"]))
    return f"""# Stage 20-Q Contact/Holding Auxiliary Target Feasibility Audit

## Status

```text
Stage 20-Q contact/holding audit = PASS
training = not_run
rollout_smoke = not_run
label_policy = no_hard_fabrication
```

## Inputs

```text
dataset = {audit['dataset_dir']}
split = {audit['split']}
num_samples = {audit['num_samples']}
contact_distance_threshold = {audit['thresholds']['contact_distance_threshold']}
closed_gripper_threshold = {audit['thresholds']['closed_gripper_threshold']}
```

## Direct Label Availability

```text
direct_contact_fields = {audit['direct_fields']['contact']}
direct_holding_fields = {audit['direct_fields']['holding']}
direct_label_available = {str(audit['direct_label_available']).lower()}
proxy_only = {str(audit['proxy_only']).lower()}
```

## Proxy Stability

```text
holding_positive_rate_in_phase_2_3 = {_fmt(feasibility['holding_positive_rate_in_phase_2_3'])}
holding_negative_rate_outside_phase_2_3 = {_fmt(feasibility['holding_negative_rate_outside_phase_2_3'])}
holding_phase_separation = {_fmt(feasibility['holding_phase_separation'])}
contact_phase1_rate = {_fmt(feasibility['contact_phase1_rate'])}
contact_phase0_rate = {_fmt(feasibility['contact_phase0_rate'])}
contact_phase_separation = {_fmt(feasibility['contact_phase_separation'])}
phase_leak_risk = {str(feasibility['phase_leak_risk']).lower()}
recommended_for_training_now = {str(feasibility['recommended_for_training_now']).lower()}
```

## Per-Phase Proxy Ratios

```text
{per_phase_lines}
```

## Interpretation

- The current dataset has role-state geometry and actions, but no direct contact or object-in-gripper field.
- The holding proxy is useful for audit because phases 2/3 are close-gripper carrying phases, but it also has phase-leak risk.
- A supervised contact/holding auxiliary head should use recorder-provided labels, not this proxy as ground truth.

## Recommendation

{audit['recommendation']}
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Stage 20-Q contact/holding target feasibility")
    parser.add_argument("--dataset_dir", default="generated/datasets/stage_20/role_conditioned_stack_skill_dataset_v0_2_phase_labeled")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--contact_distance_threshold", type=float, default=0.035)
    parser.add_argument("--closed_gripper_threshold", type=float, default=0.03)
    parser.add_argument("--output_json", default="generated/results/stage_20/stage20_q_contact_holding_feasibility_audit.json")
    parser.add_argument("--output_md", default="generated/results/stage_20/stage20_q_contact_holding_feasibility_audit.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    audit = run_audit(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    _write_json(output_json, audit)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown_report(audit), encoding="utf-8")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
