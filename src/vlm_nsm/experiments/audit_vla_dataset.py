"""D0 Dataset Audit — check data quality, distribution, and training readiness.

Answers:
  1. Are failure episode windows mixed into training data?
  2. What is the phase/action/gripper distribution?
  3. Are there data integrity issues (NaN, shape mismatch, etc.)?
  4. Is green cube under-represented?

Usage:
    nsm/bin/python -m vlm_nsm.experiments.audit_vla_dataset \\
        --dataset_dir generated/datasets/vla_demos/v0 \\
        --output_dir generated/results/vla/d0_audit
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_PHASE_NAMES = [
    "reach_pick", "grasp", "lift", "reach_place", "release", "retreat",
]

_FAILURE_TYPE_MAP = {
    "preplace": "preplace_misaligned",
    "nocontact": "nocontact",
    "horizon": "horizon",
    "horizon_preplace": "horizon_preplace",
    "verify_failed": "verify_failed",
    None: "none",
}


def load_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def check_hdf5_integrity(episodes_dir: Path, manifest: list[dict]) -> dict[str, Any]:
    """Verify all HDF5 files are readable and have correct shapes."""
    issues = []
    total_episodes = 0
    total_steps = 0
    corrupted = 0

    for ep in manifest:
        ep_path = Path(ep.get("episode_path", ""))
        if not ep_path.exists():
            issues.append(f"MISSING: {ep_path}")
            corrupted += 1
            continue

        try:
            with h5py.File(ep_path, "r") as f:
                T_av = f["observations/agentview"].shape[0]
                T_fv = f["observations/frontview"].shape[0]
                T_act = f["actions_4d"].shape[0]
                T_eef = f["proprio/eef_pos"].shape[0]
                T_phase = f["symbolic/phase_id"].shape[0]

                if not (T_av == T_fv == T_act == T_eef == T_phase):
                    issues.append(
                        f"SHAPE_MISMATCH: {ep['episode_id']} "
                        f"av={T_av} fv={T_fv} act={T_act} eef={T_eef} phase={T_phase}"
                    )
                    corrupted += 1
                    continue

                # Check for NaN/inf in actions
                actions = f["actions_4d"][:]
                if np.any(~np.isfinite(actions)):
                    issues.append(f"NAN_ACTION: {ep['episode_id']}")
                    corrupted += 1

                # Check image value range
                av = f["observations/agentview"][:]
                if av.min() < 0 or av.max() > 255:
                    issues.append(f"IMAGE_RANGE: {ep['episode_id']} av=[{av.min()},{av.max()}]")

                # Check phase_id bounds
                phases = f["symbolic/phase_id"][:]
                if phases.min() < 0 or phases.max() >= len(_PHASE_NAMES):
                    issues.append(f"PHASE_BOUNDS: {ep['episode_id']} phase=[{phases.min()},{phases.max()}]")

                # Check attrs match manifest
                if f.attrs.get("success") != ep.get("success"):
                    issues.append(
                        f"SUCCESS_MISMATCH: {ep['episode_id']} "
                        f"hdf5={f.attrs.get('success')} manifest={ep.get('success')}"
                    )

                total_episodes += 1
                total_steps += T_act

        except Exception as e:
            issues.append(f"READ_ERROR: {ep['episode_id']}: {e}")
            corrupted += 1

    return {
        "total_episodes": total_episodes,
        "corrupted": corrupted,
        "total_steps": total_steps,
        "issues": issues,
    }


def compute_distributions(manifest: list[dict], episodes_dir: Path) -> dict[str, Any]:
    """Compute phase, action, gripper distributions across all episodes."""
    success_episodes = 0
    failure_episodes = 0
    success_steps = 0
    failure_steps = 0
    failure_types: dict[str, int] = defaultdict(int)
    task_success: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})

    phase_steps: dict[str, int] = defaultdict(int)
    phase_action_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"dx": [], "dy": [], "dz": [], "gripper": []})

    episode_lengths: list[int] = []

    for ep in manifest:
        ep_path = Path(ep.get("episode_path", ""))
        if not ep_path.exists():
            continue

        task_key = f"{ep.get('pick_object','?')}_to_{ep.get('target_object','?')}"
        task_success[task_key]["total"] += 1

        try:
            with h5py.File(ep_path, "r") as f:
                T = f["actions_4d"].shape[0]
                episode_lengths.append(T)

                phases = f["symbolic/phase_id"][:]
                actions = f["actions_4d"][:]

                for i in range(T):
                    pid = int(phases[i])
                    phase_name = _PHASE_NAMES[pid] if 0 <= pid < len(_PHASE_NAMES) else f"p{pid}"
                    phase_steps[phase_name] += 1
                    phase_action_stats[phase_name]["dx"].append(float(actions[i, 0]))
                    phase_action_stats[phase_name]["dy"].append(float(actions[i, 1]))
                    phase_action_stats[phase_name]["dz"].append(float(actions[i, 2]))
                    phase_action_stats[phase_name]["gripper"].append(float(actions[i, 3]))

                if ep.get("success", False):
                    success_episodes += 1
                    success_steps += T
                    task_success[task_key]["success"] += 1
                else:
                    failure_episodes += 1
                    failure_steps += T
                    ft = ep.get("failure_type") or "unknown"
                    failure_types[_FAILURE_TYPE_MAP.get(ft, ft)] += 1

        except Exception:
            continue

    # Summarize per-phase stats
    phase_summary = {}
    for phase, stats in sorted(phase_action_stats.items()):
        phase_summary[phase] = {
            "steps": phase_steps[phase],
            "dx_mean": float(np.mean(stats["dx"])),
            "dx_std": float(np.std(stats["dx"])),
            "dy_mean": float(np.mean(stats["dy"])),
            "dy_std": float(np.std(stats["dy"])),
            "dz_mean": float(np.mean(stats["dz"])),
            "dz_std": float(np.std(stats["dz"])),
            "gripper_mean": float(np.mean(stats["gripper"])),
            "gripper_std": float(np.std(stats["gripper"])),
            "gripper_close_ratio": float(np.mean(
                [1.0 if g > 0 else 0.0 for g in stats["gripper"]])),
        }

    return {
        "success_episodes": success_episodes,
        "failure_episodes": failure_episodes,
        "success_steps": success_steps,
        "failure_steps": failure_steps,
        "success_step_ratio": success_steps / max(success_steps + failure_steps, 1),
        "failure_types": dict(failure_types),
        "task_breakdown": {
            k: {"total": v["total"], "success": v["success"],
                "success_rate": v["success"] / max(v["total"], 1)}
            for k, v in task_success.items()
        },
        "phase_distribution": dict(phase_steps),
        "phase_action_stats": phase_summary,
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "episode_length_min": int(np.min(episode_lengths)),
        "episode_length_max": int(np.max(episode_lengths)),
    }


def compute_window_counts(
    manifest: list[dict], episodes_dir: Path, stride: int = 4,
    obs_history_len: int = 1, action_chunk_len: int = 1,
) -> dict[str, Any]:
    """Count training windows: success vs failure, per-phase."""
    success_windows = 0
    failure_windows = 0
    per_phase_windows: dict[str, dict[str, int]] = defaultdict(
        lambda: {"success": 0, "failure": 0})

    for ep in manifest:
        ep_path = Path(ep.get("episode_path", ""))
        if not ep_path.exists():
            continue
        is_success = ep.get("success", False)

        try:
            with h5py.File(ep_path, "r") as f:
                T = f["actions_4d"].shape[0]
                phases = f["symbolic/phase_id"][:]
                min_start = obs_history_len - 1
                max_start = T - action_chunk_len - 1

                for t in range(min_start, max_start + 1, stride):
                    pid = int(phases[t])
                    phase_name = _PHASE_NAMES[pid] if 0 <= pid < len(_PHASE_NAMES) else f"p{pid}"

                    if is_success:
                        success_windows += 1
                        per_phase_windows[phase_name]["success"] += 1
                    else:
                        failure_windows += 1
                        per_phase_windows[phase_name]["failure"] += 1
        except Exception:
            continue

    return {
        "success_windows": success_windows,
        "failure_windows": failure_windows,
        "failure_window_ratio": failure_windows / max(success_windows + failure_windows, 1),
        "per_phase": {
            phase: {
                "success": counts["success"],
                "failure": counts["failure"],
                "total": counts["success"] + counts["failure"],
                "failure_ratio": counts["failure"] / max(counts["success"] + counts["failure"], 1),
            }
            for phase, counts in sorted(per_phase_windows.items())
        },
    }


def build_report(
    integrity: dict, distributions: dict, windows: dict, output_dir: Path,
) -> str:
    """Generate markdown audit report."""
    lines = [
        "# D0 Dataset Audit Report",
        "",
        "## 1. Data Integrity",
        "",
        f"- Total episodes: {integrity['total_episodes']}",
        f"- Corrupted: {integrity['corrupted']}",
        f"- Total steps: {integrity['total_steps']:,}",
        f"- Issues found: {len(integrity['issues'])}",
    ]
    if integrity["issues"]:
        lines.append("")
        lines.append("### Issues")
        for issue in integrity["issues"]:
            lines.append(f"- {issue}")
    else:
        lines.append("- ✅ All HDF5 files pass integrity check")

    dist = distributions
    lines.extend([
        "",
        "## 2. Episode Distribution",
        "",
        f"- Success episodes: {dist['success_episodes']}",
        f"- Failure episodes: {dist['failure_episodes']}",
        f"- Success step ratio: {dist['success_step_ratio']:.1%}",
        f"- Episode length: mean={dist['episode_length_mean']:.0f} "
        f"std={dist['episode_length_std']:.0f} "
        f"min={dist['episode_length_min']} max={dist['episode_length_max']}",
    ])

    lines.extend([
        "",
        "## 3. Failure Distribution",
        "",
        "| Failure Type | Count |",
        "|-------------|------:|",
    ])
    for ft, count in sorted(dist["failure_types"].items(), key=lambda x: -x[1]):
        lines.append(f"| {ft} | {count} |")

    lines.extend([
        "",
        "## 4. Task Breakdown",
        "",
        "| Task | Total | Success | Rate |",
        "|------|------:|-------:|-----:|",
    ])
    for task, stats in sorted(dist["task_breakdown"].items()):
        lines.append(
            f"| {task} | {stats['total']} | {stats['success']} | "
            f"{stats['success_rate']:.0%} |"
        )

    lines.extend([
        "",
        "## 5. Phase Distribution",
        "",
        "| Phase | Steps | Ratio | Gripper Mean | Gripper Close% |",
        "|-------|------:|------:|-------------:|---------------:|",
    ])
    total_phase_steps = sum(dist["phase_distribution"].values())
    for phase, count in sorted(dist["phase_distribution"].items()):
        ratio = count / max(total_phase_steps, 1)
        pstats = dist["phase_action_stats"].get(phase, {})
        gm = pstats.get("gripper_mean", 0)
        gc = pstats.get("gripper_close_ratio", 0)
        lines.append(f"| {phase} | {count:,} | {ratio:.1%} | {gm:+.3f} | {gc:.1%} |")

    lines.extend([
        "",
        "## 6. Action Stats by Phase",
        "",
        "| Phase | dx_mean | dx_std | dy_mean | dy_std | dz_mean | dz_std |",
        "|-------|--------:|-------:|--------:|-------:|--------:|-------:|",
    ])
    for phase, stats in sorted(dist["phase_action_stats"].items()):
        lines.append(
            f"| {phase} | {stats['dx_mean']:.4f} | {stats['dx_std']:.4f} | "
            f"{stats['dy_mean']:.4f} | {stats['dy_std']:.4f} | "
            f"{stats['dz_mean']:.4f} | {stats['dz_std']:.4f} |"
        )

    w = windows
    lines.extend([
        "",
        "## 7. Training Windows (stride=4)",
        "",
        f"- Success windows: {w['success_windows']:,}",
        f"- Failure windows: {w['failure_windows']:,}",
        f"- Failure window ratio: {w['failure_window_ratio']:.1%}",
        "",
        "| Phase | Success | Failure | Total | Failure% |",
        "|-------|--------:|--------:|------:|---------:|",
    ])
    for phase, counts in sorted(w["per_phase"].items()):
        lines.append(
            f"| {phase} | {counts['success']:,} | {counts['failure']:,} | "
            f"{counts['total']:,} | {counts['failure_ratio']:.1%} |"
        )

    # Recommendations
    lines.extend([
        "",
        "## 8. Recommendations",
        "",
    ])
    if w["failure_window_ratio"] > 0.05:
        lines.append(
            f"- ⚠️ Failure windows make up {w['failure_window_ratio']:.1%} of training data. "
            f"Recommend success-only training."
        )
    else:
        lines.append("- ✅ Failure window ratio is low. Data is clean for training.")

    phase_dist = dist["phase_distribution"]
    max_phase = max(phase_dist, key=phase_dist.get)
    min_phase = min(phase_dist, key=phase_dist.get)
    imbalance = phase_dist[max_phase] / max(phase_dist[min_phase], 1)
    if imbalance > 5:
        lines.append(
            f"- ⚠️ Phase imbalance: {max_phase} has {imbalance:.0f}x more steps than {min_phase}. "
            f"Consider phase-aware sampling."
        )

    green_tasks = [k for k in dist["task_breakdown"] if "cube2" in k and "pick" not in k.split("_to_")[0]]
    if not green_tasks:
        green_pick = [k for k in dist["task_breakdown"] if k.startswith("cube2")]
        for gt in green_pick:
            rate = dist["task_breakdown"][gt]["success_rate"]
            if rate < 0.7:
                lines.append(
                    f"- ⚠️ Green cube pick ({gt}) success rate is {rate:.0%}. "
                    f"Prioritize green cube grasp profile."
                )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="D0 VLA Dataset Audit")
    parser.add_argument("--dataset_dir", default="generated/datasets/vla_demos/v0")
    parser.add_argument("--output_dir", default="generated/results/vla/d0_audit")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(dataset_dir)
    episodes_dir = dataset_dir / "episodes"

    print(f"Loaded {len(manifest)} manifest entries")

    # 1. Integrity
    print("Checking HDF5 integrity...")
    integrity = check_hdf5_integrity(episodes_dir, manifest)
    print(f"  {integrity['total_episodes']} valid, {integrity['corrupted']} corrupted, "
          f"{len(integrity['issues'])} issues")

    # 2. Distributions
    print("Computing distributions...")
    distributions = compute_distributions(manifest, episodes_dir)
    print(f"  {distributions['success_episodes']} success, "
          f"{distributions['failure_episodes']} failure episodes")

    # 3. Window counts
    print("Counting training windows...")
    windows = compute_window_counts(manifest, episodes_dir, stride=args.stride)
    print(f"  {windows['success_windows']:,} success, "
          f"{windows['failure_windows']:,} failure windows "
          f"({windows['failure_window_ratio']:.1%})")

    # 4. Save outputs
    (output_dir / "integrity.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "distributions.json").write_text(
        json.dumps(distributions, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "windows.json").write_text(
        json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_report(integrity, distributions, windows, output_dir)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nReport saved to {output_dir / 'report.md'}")
    print(report)


if __name__ == "__main__":
    main()
