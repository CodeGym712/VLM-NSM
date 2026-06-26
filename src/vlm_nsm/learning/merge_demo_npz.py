"""Merge per-episode demo npz files into a single merged npz.

Reads all ``episode_*.npz`` files from a directory, concatenates trajectory
arrays, and writes a single ``trajectories.npz`` with consistent episode_id.

Usage:
    python3 -m vlm_nsm.learning.merge_demo_npz \\
        --demo_dir path/to/strict_only \\
        --output path/to/trajectories_strict_only.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def merge_demo_npz(
    demo_dir: str,
    output_path: str,
    episode_prefix: str = "episode_",
    required_keys: list[str] | None = None,
) -> dict:
    """Merge per-episode npz files into a single npz.

    Args:
        demo_dir: Directory containing episode_*.npz files.
        output_path: Output .npz path.
        episode_prefix: File name prefix for episode files.
        required_keys: Keys that must be present in each file.

    Returns:
        Statistics dict.
    """
    if required_keys is None:
        required_keys = ["filtered_obs", "action", "phase_id", "episode_id",
                         "step_id", "skill_id", "done", "success"]

    demo_dir = Path(demo_dir)
    episode_files = sorted(demo_dir.glob(f"{episode_prefix}*.npz"))

    if not episode_files:
        raise FileNotFoundError(f"No {episode_prefix}*.npz files found in {demo_dir}")

    # Collect all trajectory arrays
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in required_keys}
    total_steps = 0
    episode_count = 0

    for ep_idx, fpath in enumerate(episode_files):
        data = np.load(fpath)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  WARNING: {fpath.name} missing keys {missing}, skipping")
            continue
        for k in required_keys:
            arrays[k].append(data[k])
        total_steps += len(data[required_keys[0]])
        episode_count += 1

    # Reassign episode_id to be globally unique (0..episode_count-1)
    episode_id_reindexed: list[np.ndarray] = []
    for ep_idx, arr in enumerate(arrays["episode_id"]):
        episode_id_reindexed.append(np.full_like(arr, ep_idx, dtype=np.int32))
    arrays["episode_id"] = episode_id_reindexed

    # Concatenate
    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}

    # Validate
    n = len(merged[required_keys[0]])
    for k in required_keys:
        assert len(merged[k]) == n, f"Length mismatch for {k}: {len(merged[k])} vs {n}"

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    # Add metadata
    meta = {
        "input_dir": str(demo_dir),
        "num_episodes": episode_count,
        "total_steps": int(total_steps),
        "output_path": str(output_path),
        "keys": required_keys,
        "episode_files": len(episode_files),
    }
    meta_path = output_path.with_name(output_path.stem + ".meta.json")
    import json
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Merge per-episode demo npz files")
    parser.add_argument("--demo_dir", type=str, required=True,
                        help="Directory of episode_*.npz files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    args = parser.parse_args()

    meta = merge_demo_npz(args.demo_dir, args.output)
    print(f"Merged {meta['num_episodes']} episodes, {meta['total_steps']} steps")
    print(f"Output: {meta['output_path']}")
    print(f"Keys: {meta['keys']}")


if __name__ == "__main__":
    main()
