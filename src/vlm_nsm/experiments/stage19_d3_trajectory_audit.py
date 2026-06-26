"""Stage 19-D3-1: Trajectory availability and format audit.

Audits forward and reverse demonstration trajectories to determine whether
a balanced trainable SkillDataset can be built.

CLI:
  python -m vlm_nsm.experiments.stage19_d3_trajectory_audit \
    --balanced_index generated/datasets/stage19_d2_balanced_skill_dataset.json \
    --forward_dataset generated/datasets/skill_demos_v0_1_smoke \
    --reverse_episodes generated/results/stage_19/stage19_d2_reverse_collection_episodes.jsonl \
    --output generated/results/stage_19/stage19_d3_trajectory_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_forward_dataset(dataset_dir: Path, index_items: list[dict]) -> dict:
    """Audit forward (cube1->cube2) trajectories."""
    manifest_path = dataset_dir / "manifest.jsonl"
    trajectories_dir = dataset_dir / "trajectories"

    manifest_exists = manifest_path.exists()
    traj_dir_exists = trajectories_dir.exists()

    indexed_ids = {item["episode_id"] for item in index_items}

    manifest_entries: dict[str, dict] = {}
    if manifest_exists:
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                eid = entry.get("episode_id", "")
                if eid:
                    manifest_entries[eid] = entry

    # Match indexed items to manifest entries
    matched: list[dict] = []
    missing_manifest: list[str] = []
    missing_traj: list[str] = []
    trajectory_files: list[str] = []

    for item in index_items:
        eid = item["episode_id"]
        if eid not in manifest_entries:
            missing_manifest.append(eid)
            continue
        entry = manifest_entries[eid]
        traj_path = entry.get("trajectory_path", "")
        full_path = dataset_dir / traj_path if traj_path else None

        if full_path and full_path.exists():
            matched.append({"episode_id": eid, "manifest_entry": entry, "trajectory_file": str(full_path)})
            trajectory_files.append(str(full_path))
        else:
            missing_traj.append(eid)

    # Sample obs/action dims from first available trajectory
    obs_dim = None
    action_dim = None
    npz_keys: list[str] = []
    total_steps = 0
    has_phase_id = False
    has_skill_labels = False

    for tf_path in trajectory_files[:3]:
        try:
            data = np.load(tf_path, allow_pickle=True)
            if obs_dim is None:
                obs_dim = int(data["filtered_obs"].shape[1]) if "filtered_obs" in data else int(data["obs"].shape[1])
            if action_dim is None:
                action_dim = int(data["action"].shape[1])
            if not npz_keys:
                npz_keys = sorted(data.keys())
            total_steps += int(data["action"].shape[0])
            has_phase_id = has_phase_id or "phase_id" in data
            has_skill_labels = has_skill_labels or "skill_object_id" in data
        except Exception:
            continue

    return {
        "dataset_dir": str(dataset_dir),
        "manifest_exists": manifest_exists,
        "trajectories_dir_exists": traj_dir_exists,
        "manifest_entries": len(manifest_entries),
        "trajectory_files_count": len(list(trajectories_dir.iterdir())) if traj_dir_exists else 0,
        "indexed_count": len(index_items),
        "matched_trajectories": len(matched),
        "missing_from_manifest": missing_manifest,
        "missing_trajectory_file": missing_traj,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "npz_keys": npz_keys,
        "total_steps_sampled_3": total_steps,
        "has_phase_id": has_phase_id,
        "has_skill_labels": has_skill_labels,
        "format": "per_step_npz_with_manifest",
        "compatible_with_skill_dataset": True,
    }


def audit_reverse_episodes(episodes_path: Path, index_items: list[dict]) -> dict:
    """Audit reverse (cube2->cube1) trajectory availability."""
    episodes: dict[str, dict] = {}
    if episodes_path.exists():
        with open(episodes_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                eid = ep.get("episode_id", "")
                if eid:
                    episodes[eid] = ep

    indexed_ids = {item["episode_id"] for item in index_items}

    matched: list[dict] = []
    missing_episode: list[str] = []
    no_trajectory: list[str] = []
    success_count = 0
    steps_list: list[int] = []

    for item in index_items:
        eid = item["episode_id"]
        if eid not in episodes:
            missing_episode.append(eid)
            continue
        ep = episodes[eid]
        if ep.get("task_success"):
            success_count += 1
        steps_list.append(ep.get("steps", 0))

        traj_path = ep.get("trajectory_path", "")
        if traj_path:
            matched.append({"episode_id": eid, "trajectory_path": traj_path})
        else:
            no_trajectory.append(eid)

    return {
        "episodes_file": str(episodes_path),
        "episodes_in_file": len(episodes),
        "indexed_count": len(index_items),
        "matched_in_episodes": len(index_items) - len(missing_episode),
        "missing_from_episodes": missing_episode,
        "has_trajectory_files": len(matched),
        "missing_trajectory_files": len(no_trajectory),
        "trajectory_available": len(matched) > 0,
        "success_count": success_count,
        "total_steps_episodes": sum(steps_list),
        "format": "episode_summary_only" if not matched else "mixed",
    }


def check_format_compatibility(forward: dict, reverse: dict) -> dict:
    """Check whether forward and reverse formats are compatible."""
    issues: list[str] = []

    if not forward.get("compatible_with_skill_dataset"):
        issues.append("Forward dataset is not SkillDataset-compatible")
    if not reverse.get("trajectory_available"):
        issues.append("Reverse trajectories are not available — need D3-2 collection")

    if forward.get("obs_dim") and forward.get("obs_dim") != 11:
        issues.append(f"Forward obs_dim={forward['obs_dim']} (expected 11)")
    if forward.get("action_dim") and forward.get("action_dim") != 7:
        issues.append(f"Forward action_dim={forward['action_dim']} (expected 7)")

    compatible = len(issues) == 0
    return {
        "formats_compatible": compatible,
        "issues": issues,
        "can_build_directly": compatible and reverse.get("trajectory_available"),
        "needs_reverse_collection": not reverse.get("trajectory_available"),
    }


def run_audit(
    balanced_index: dict,
    forward_dataset_dir: Path,
    reverse_episodes_path: Path,
) -> dict:
    """Run full trajectory audit."""
    forward_items = balanced_index.get("forward_items", [])
    reverse_items = balanced_index.get("reverse_items", [])

    forward_audit = audit_forward_dataset(forward_dataset_dir, forward_items)
    reverse_audit = audit_reverse_episodes(reverse_episodes_path, reverse_items)
    compatibility = check_format_compatibility(forward_audit, reverse_audit)

    forward_available = forward_audit["matched_trajectories"]
    reverse_available = reverse_audit["has_trajectory_files"]

    recommendation: str
    if forward_available >= 40 and reverse_available >= 40:
        recommendation = "READY — sufficient trajectories available for balanced dataset"
    elif not reverse_available:
        recommendation = (
            "REVERSE_TRAJECTORIES_UNAVAILABLE — "
            "re-run successful reverse seeds with trajectory saving enabled (D3-2)"
        )
    elif forward_available < 40:
        recommendation = f"INSUFFICIENT_FORWARD — only {forward_available} forward trajectories available"
    else:
        recommendation = "PARTIAL — some trajectories missing, check audit details"

    return {
        "audit_date": "2026-05-27",
        "balanced_index": {
            "file": str(getattr(balanced_index, "__path__", "")),
            "forward_count": len(forward_items),
            "reverse_count": len(reverse_items),
        },
        "forward_audit": forward_audit,
        "reverse_audit": reverse_audit,
        "format_compatibility": compatibility,
        "recommendation": recommendation,
        "summary": {
            "forward_trajectories_available": forward_available,
            "reverse_trajectories_available": reverse_available,
            "ready_to_build_dataset": compatibility["can_build_directly"],
            "needs_reverse_collection": compatibility["needs_reverse_collection"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="D3-1: Trajectory availability audit")
    parser.add_argument("--balanced_index", required=True,
                        help="Path to D2 balanced dataset index JSON")
    parser.add_argument("--forward_dataset", required=True,
                        help="Path to forward skill dataset directory")
    parser.add_argument("--reverse_episodes", required=True,
                        help="Path to D2-3 reverse collection episodes JSONL")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    parser.add_argument("--markdown_output", default=None,
                        help="Optional markdown report output")
    args = parser.parse_args()

    balanced_index = _load_json(Path(args.balanced_index))
    forward_dir = Path(args.forward_dataset)
    reverse_path = Path(args.reverse_episodes)

    if not forward_dir.exists():
        print(f"ERROR: Forward dataset not found: {forward_dir}")
        return
    if not reverse_path.exists():
        print(f"ERROR: Reverse episodes not found: {reverse_path}")
        return

    result = run_audit(balanced_index, forward_dir, reverse_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit written to: {output_path}")

    print(f"\n=== D3-1 Trajectory Audit ===")
    print(f"Forward trajectories available: {result['summary']['forward_trajectories_available']}")
    print(f"Reverse trajectories available: {result['summary']['reverse_trajectories_available']}")
    print(f"Ready to build dataset: {result['summary']['ready_to_build_dataset']}")
    print(f"Needs reverse collection: {result['summary']['needs_reverse_collection']}")
    print(f"Recommendation: {result['recommendation']}")

    for issue in result["format_compatibility"]["issues"]:
        print(f"  ISSUE: {issue}")

    # Write markdown report if requested
    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = _generate_markdown(result)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown report: {md_path}")

    return result


def _generate_markdown(result: dict) -> str:
    fwd = result["forward_audit"]
    rev = result["reverse_audit"]
    compat = result["format_compatibility"]

    lines = [
        "# Stage 19-D3-1: Trajectory Availability Audit",
        "",
        f"**Date**: {result['audit_date']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Forward trajectories | {result['summary']['forward_trajectories_available']} |",
        f"| Reverse trajectories | {result['summary']['reverse_trajectories_available']} |",
        f"| Ready to build | {result['summary']['ready_to_build_dataset']} |",
        f"| Needs reverse collection | {result['summary']['needs_reverse_collection']} |",
        "",
        f"**Recommendation**: {result['recommendation']}",
        "",
        "## Forward Dataset",
        "",
        f"- Path: {fwd['dataset_dir']}",
        f"- Manifest entries: {fwd['manifest_entries']}",
        f"- Trajectory files: {fwd['trajectory_files_count']}",
        f"- Matched trajectories: {fwd['matched_trajectories']}/{fwd['indexed_count']}",
        f"- obs_dim: {fwd['obs_dim']}, action_dim: {fwd['action_dim']}",
        f"- has_phase_id: {fwd['has_phase_id']}",
        f"- has_skill_labels: {fwd['has_skill_labels']}",
        f"- NPZ keys: {fwd['npz_keys']}",
        "",
        "## Reverse Dataset",
        "",
        f"- Episodes file: {rev['episodes_file']}",
        f"- Episodes in file: {rev['episodes_in_file']}",
        f"- Matched in episodes: {rev['matched_in_episodes']}/{rev['indexed_count']}",
        f"- Has trajectory files: {rev['has_trajectory_files']}",
        f"- Trajectory available: {rev['trajectory_available']}",
        "",
        "## Format Compatibility",
        "",
    ]

    if compat["issues"]:
        for issue in compat["issues"]:
            lines.append(f"- ISSUE: {issue}")
    else:
        lines.append("- No compatibility issues")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
