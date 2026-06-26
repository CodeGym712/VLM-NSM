"""Stage 19-D3-3: Build balanced trainable SkillDataset.

Merges forward and reverse demonstration trajectories into a combined
manifest-based dataset compatible with existing SkillDataset training code.

CLI:
  python -m vlm_nsm.experiments.stage19_build_balanced_train_dataset \
    --forward_dataset generated/datasets/skill_demos_v0_1_smoke \
    --reverse_dataset generated/datasets/stage_19/reverse_stack_demos_v0_1_trajectories \
    --output_dir generated/datasets/stage_19/balanced_stack_skill_dataset_v0_1 \
    --max_per_direction 44 \
    --val_ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.learning.skill_dataset import load_manifest


def _resolve_manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "manifest.jsonl"


def load_direction_entries(dataset_dir: Path, direction: str, max_count: int) -> list[dict]:
    """Load successful entries from a dataset directory, labelling with direction."""
    manifest_path = _resolve_manifest_path(dataset_dir)
    if not manifest_path.exists():
        print(f"  WARNING: manifest not found: {manifest_path}")
        return []

    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("success") is not True:
                continue
            traj_path = entry.get("trajectory_path", "")
            if not traj_path:
                continue
            full_path = dataset_dir / traj_path
            if not full_path.exists():
                continue
            entry["_trajectory_source_file"] = str(full_path)
            entry["direction"] = direction
            entries.append(entry)

    if len(entries) > max_count:
        entries = entries[:max_count]

    return entries


def copy_trajectory(src_path: str, dst_dir: Path, episode_id: str) -> str | None:
    """Copy one trajectory NPZ and return relative path."""
    src = Path(src_path)
    if not src.exists():
        return None
    dst = dst_dir / "trajectories" / f"{episode_id}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"trajectories/{dst.name}"


def stratified_split(entries: list[dict], val_ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split entries into train/val stratified by direction."""
    fwd = [e for e in entries if e.get("direction") == "forward"]
    rev = [e for e in entries if e.get("direction") == "reverse"]

    rng = random.Random(seed)
    rng.shuffle(fwd)
    rng.shuffle(rev)

    fwd_val_n = max(1, int(len(fwd) * val_ratio))
    rev_val_n = max(1, int(len(rev) * val_ratio))

    val_entries = fwd[:fwd_val_n] + rev[:rev_val_n]
    train_entries = fwd[fwd_val_n:] + rev[rev_val_n:]

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)

    return train_entries, val_entries


def build_manifest_entry(entry: dict, traj_path: str | None, episode_index: int) -> dict:
    """Create a manifest entry for the balanced dataset."""
    return _json_safe({
        "episode_id": entry.get("episode_id", f"ep_{episode_index:06d}"),
        "skill_name": entry.get("skill_name", "pick_place"),
        "object_id": entry.get("object_id", ""),
        "target_object_id": entry.get("target_object_id", ""),
        "relation": entry.get("relation", "on"),
        "source": entry.get("source", "table"),
        "direction": entry.get("direction", ""),
        "success": entry.get("success", True),
        "num_steps": entry.get("num_steps", 0),
        "failure_type": entry.get("failure_type"),
        "trajectory_path": traj_path,
        "seed": entry.get("seed"),
        "teacher_mode": entry.get("teacher_mode", ""),
        "initial_scene": entry.get("initial_scene", {}),
        "final_scene": entry.get("final_scene", {}),
    })


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_balanced_dataset(
    forward_dir: Path,
    reverse_dir: Path,
    output_dir: Path,
    *,
    max_per_direction: int = 44,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a balanced trainable SkillDataset from forward and reverse demos."""

    if not reverse_dir.exists():
        raise FileNotFoundError(
            f"Reverse dataset not found: {reverse_dir}\n"
            "Run D3-2 first to collect reverse trajectories, then re-run D3-3."
        )

    print(f"Loading forward entries from: {forward_dir}")
    forward_entries = load_direction_entries(forward_dir, "forward", max_per_direction)
    print(f"  Forward: {len(forward_entries)}")

    print(f"Loading reverse entries from: {reverse_dir}")
    reverse_entries = load_direction_entries(reverse_dir, "reverse", max_per_direction)
    print(f"  Reverse: {len(reverse_entries)}")

    if len(reverse_entries) < 10:
        print(f"  WARNING: Only {len(reverse_entries)} reverse entries. Dataset may be too small.")

    all_entries = forward_entries + reverse_entries

    # Copy trajectories
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    for idx, entry in enumerate(all_entries):
        src_file = entry.pop("_trajectory_source_file", None)
        episode_id = entry.get("episode_id", f"ep_{idx:06d}")

        traj_rel_path = None
        if src_file:
            traj_rel_path = copy_trajectory(src_file, output_dir, episode_id)

        manifest_entry = build_manifest_entry(entry, traj_rel_path, idx)
        manifest_entries.append(manifest_entry)

    print(f"  Trajectories copied: {sum(1 for m in manifest_entries if m['trajectory_path'])}")

    # Stratified split
    train_entries, val_entries = stratified_split(manifest_entries, val_ratio, seed)

    # Write manifests
    write_manifest(manifest_entries, output_dir / "manifest.jsonl")
    write_manifest(train_entries, output_dir / "train_manifest.jsonl")
    write_manifest(val_entries, output_dir / "val_manifest.jsonl")

    # Validate
    fwd_count = sum(1 for e in manifest_entries if e.get("direction") == "forward")
    rev_count = sum(1 for e in manifest_entries if e.get("direction") == "reverse")
    train_fwd = sum(1 for e in train_entries if e.get("direction") == "forward")
    train_rev = sum(1 for e in train_entries if e.get("direction") == "reverse")
    val_fwd = sum(1 for e in val_entries if e.get("direction") == "forward")
    val_rev = sum(1 for e in val_entries if e.get("direction") == "reverse")

    validations: list[dict] = [
        {"check": "forward_count >= 40", "pass": fwd_count >= 40, "value": fwd_count},
        {"check": "reverse_count >= 40", "pass": rev_count >= 40, "value": rev_count},
        {"check": "train_has_both_directions", "pass": train_fwd > 0 and train_rev > 0,
         "value": f"fwd={train_fwd}, rev={train_rev}"},
        {"check": "val_has_both_directions", "pass": val_fwd > 0 and val_rev > 0,
         "value": f"fwd={val_fwd}, rev={val_rev}"},
    ]

    # Summary
    obs_dim = 11
    action_dim = 7

    summary = {
        "dataset_name": "balanced_stack_skill_dataset_v0_1",
        "dataset_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_episodes": len(manifest_entries),
        "forward_count": fwd_count,
        "reverse_count": rev_count,
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "val_ratio": val_ratio,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "manifest_path": "manifest.jsonl",
        "train_manifest_path": "train_manifest.jsonl",
        "val_manifest_path": "val_manifest.jsonl",
        "trajectory_dir": "trajectories",
        "source_datasets": {
            "forward": str(forward_dir),
            "reverse": str(reverse_dir),
        },
        "validations": validations,
        "all_validations_pass": all(v["pass"] for v in validations),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # README
    readme = (
        "# Balanced Stack Skill Dataset v0.1\n\n"
        f"Created: {summary['created_at']}\n\n"
        "## Contents\n\n"
        f"- Forward (cube1->cube2): {fwd_count} episodes\n"
        f"- Reverse (cube2->cube1): {rev_count} episodes\n"
        f"- Train: {len(train_entries)} episodes\n"
        f"- Val: {len(val_entries)} episodes\n\n"
        "## Format\n\n"
        "- manifest.jsonl: All episodes\n"
        "- train_manifest.jsonl: Training split (stratified by direction)\n"
        "- val_manifest.jsonl: Validation split (stratified by direction)\n"
        "- trajectories/: Per-episode NPZ files\n\n"
        f"- obs_dim: {obs_dim}\n"
        f"- action_dim: {action_dim}\n"
        "- Compatible with `SkillDataset` from `vlm_nsm.learning.skill_dataset`\n\n"
        "## Direction Labels\n\n"
        'Each manifest entry includes a "direction" field:\n'
        '- "forward" = cube1 on cube2 (red on green)\n'
        '- "reverse" = cube2 on cube1 (green on red)\n'
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nBalanced dataset built: {output_dir}")
    print(f"  Forward: {fwd_count}, Reverse: {rev_count}")
    print(f"  Train: {len(train_entries)} (fwd={train_fwd}, rev={train_rev})")
    print(f"  Val: {len(val_entries)} (fwd={val_fwd}, rev={val_rev})")
    print(f"  Validations: {'ALL PASS' if summary['all_validations_pass'] else 'SOME FAIL'}")

    for v in validations:
        status = "PASS" if v["pass"] else "FAIL"
        print(f"    [{status}] {v['check']}: {v['value']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="D3-3: Build balanced trainable SkillDataset")
    parser.add_argument("--forward_dataset", required=True,
                        help="Path to forward skill demos directory")
    parser.add_argument("--reverse_dataset", required=True,
                        help="Path to reverse skill demos directory")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for balanced dataset")
    parser.add_argument("--max_per_direction", type=int, default=44,
                        help="Maximum episodes per direction (default: 44)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42)")
    args = parser.parse_args()

    forward_dir = Path(args.forward_dataset)
    reverse_dir = Path(args.reverse_dataset)
    output_dir = Path(args.output_dir)

    if not forward_dir.exists():
        print(f"ERROR: Forward dataset not found: {forward_dir}")
        return

    try:
        build_balanced_dataset(
            forward_dir=forward_dir,
            reverse_dir=reverse_dir,
            output_dir=output_dir,
            max_per_direction=args.max_per_direction,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return


if __name__ == "__main__":
    main()
