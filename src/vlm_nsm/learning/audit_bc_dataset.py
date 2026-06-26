"""Audit BC demonstration dataset schema and statistics.

Loads a demonstration npz file (or directory of per-episode npz files)
and reports:
  - Available keys and their shapes/dtypes
  - Observation and action dimension
  - Value ranges for each key
  - Number of episodes and total steps
  - Action statistics (for normalization)
  - Consistency checks

Usage:
    # Single npz (combined trajectories)
    python3 -m vlm_nsm.learning.audit_bc_dataset \\
        generated/demos/stage_7B-1/robosuite_stage7b1_trajectories.npz

    # Directory of per-episode npz files
    python3 -m vlm_nsm.learning.audit_bc_dataset \\
        generated/demos/stage_7A-12/all_semantic_success
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _describe_array(key: str, arr: np.ndarray) -> dict:
    """Produce a descriptive summary of an array."""
    info: dict = {
        "key": key,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": arr.ndim,
        "size": int(np.prod(arr.shape)) if arr.size > 0 else 0,
        "n_bytes": int(arr.nbytes),
    }
    if arr.ndim == 1:
        info["length"] = int(arr.shape[0])
        if np.issubdtype(arr.dtype, np.floating):
            info["min"] = float(np.min(arr))
            info["max"] = float(np.max(arr))
            info["mean"] = float(np.mean(arr))
            info["std"] = float(np.std(arr))
            info["has_nan"] = bool(np.any(np.isnan(arr)))
            info["has_inf"] = bool(np.any(np.isinf(arr)))
        elif np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
            info["min"] = int(np.min(arr)) if arr.size > 0 else None
            info["max"] = int(np.max(arr)) if arr.size > 0 else None
            info["n_unique"] = int(len(np.unique(arr)))
    elif arr.ndim == 2:
        info["n_rows"] = int(arr.shape[0])
        info["n_cols"] = int(arr.shape[1])
        if np.issubdtype(arr.dtype, np.floating):
            info["col_mins"] = [float(v) for v in np.min(arr, axis=0)[:10]]
            info["col_maxs"] = [float(v) for v in np.max(arr, axis=0)[:10]]
            info["col_means"] = [float(v) for v in np.mean(arr, axis=0)[:10]]
            info["has_nan"] = bool(np.any(np.isnan(arr)))
            info["has_inf"] = bool(np.any(np.isinf(arr)))
    return info


def _find_npz_files(path: Path) -> list[Path]:
    """Find npz files in path (single file or directory)."""
    if path.is_file() and path.suffix == ".npz":
        return [path]
    elif path.is_dir():
        return sorted(path.glob("*.npz"))
    return []


def audit_single_npz(npz_path: Path) -> dict:
    """Audit a single combined npz file."""
    data = np.load(npz_path)
    report: dict = {
        "path": str(npz_path),
        "file_size_bytes": npz_path.stat().st_size,
        "load_mode": "single_npz",
        "keys": list(data.keys()),
        "arrays": {},
    }

    for key in data.keys():
        arr = data[key]
        report["arrays"][key] = _describe_array(key, arr)

    # Episode-level stats
    if "episode_id" in data:
        ep_ids = data["episode_id"]
        n_eps = len(np.unique(ep_ids))
        report["num_episodes"] = int(n_eps)
        report["total_steps"] = int(len(ep_ids))
        if "obs" in data:
            report["obs_dim"] = int(data["obs"].shape[-1]) if data["obs"].ndim >= 2 else None
        if "action" in data:
            report["action_dim"] = int(data["action"].shape[-1]) if data["action"].ndim >= 2 else None

    return report


def audit_directory(demo_dir: Path) -> dict:
    """Audit a directory of per-episode npz files."""
    npz_files = _find_npz_files(demo_dir)
    report: dict = {
        "path": str(demo_dir),
        "load_mode": "directory",
        "num_files": len(npz_files),
        "total_file_size_bytes": sum(f.stat().st_size for f in npz_files),
        "files": [],
        "keys_across_files": {},
        "consistency": {},
    }

    if not npz_files:
        return report

    # Audit first file for schema
    first = npz_files[0]
    first_data = np.load(first)
    report["first_file"] = str(first)
    report["schema_keys"] = list(first_data.keys())

    # Collect key statistics across files
    key_dims: dict[str, set[int]] = {}
    key_dtypes: dict[str, set[str]] = {}
    total_steps = 0

    for f in npz_files:
        data = np.load(f)
        fsize = f.stat().st_size
        finfo: dict = {"path": str(f), "size_bytes": fsize}
        for k in data.keys():
            arr = data[k]
            if k not in key_dims:
                key_dims[k] = set()
                key_dtypes[k] = set()
            key_dims[k].add(arr.ndim)
            key_dtypes[k].add(str(arr.dtype))
            if k == "action":
                total_steps += len(arr)
                finfo["action_shape"] = list(arr.shape)
            elif k in ("obs", "filtered_obs"):
                finfo[f"{k}_shape"] = list(arr.shape)
        report["files"].append(finfo)

    # Determine obs_dim and action_dim from first file
    for obs_key in ("obs", "filtered_obs"):
        if obs_key in first_data and first_data[obs_key].ndim >= 2:
            report["obs_dim"] = int(first_data[obs_key].shape[-1])
            break
    if "action" in first_data and first_data["action"].ndim >= 2:
        report["action_dim"] = int(first_data["action"].shape[-1])

    report["total_steps"] = total_steps
    report["key_dimensions"] = {k: list(v) for k, v in key_dims.items()}
    report["key_dtypes"] = {k: list(v) for k, v in key_dtypes.items()}

    # Consistency checks
    all_keys_set = [set(np.load(f).keys()) for f in npz_files]
    common_keys = set.intersection(*all_keys_set) if all_keys_set else set()
    all_keys = set.union(*all_keys_set) if all_keys_set else set()
    inconsistent_keys = all_keys - common_keys

    report["consistency"] = {
        "all_files_have_same_keys": len(inconsistent_keys) == 0,
        "common_keys": sorted(common_keys),
        "files_with_extra_keys": sorted(inconsistent_keys) if inconsistent_keys else [],
        "num_files_with_obs": sum(1 for f in npz_files if "obs" in np.load(f)),
        "num_files_with_action": sum(1 for f in npz_files if "action" in np.load(f)),
        "num_files_with_filtered_obs": sum(1 for f in npz_files if "filtered_obs" in np.load(f)),
    }

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m vlm_nsm.learning.audit_bc_dataset <npz_path_or_directory>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    npz_files = _find_npz_files(path)
    if not npz_files:
        print(f"ERROR: No .npz files found at {path}", file=sys.stderr)
        sys.exit(1)

    if path.is_file():
        report = audit_single_npz(path)
    else:
        report = audit_directory(path)

    print(json.dumps(report, indent=2))

    # Save audit report alongside the data
    out_dir = path if path.is_dir() else path.parent
    out_path = out_dir / "dataset_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit report saved to {out_path}")

    # Print summary
    print("\n=== SCHEMA SUMMARY ===")
    print(f"Load mode: {report['load_mode']}")
    if report["load_mode"] == "single_npz":
        print(f"Keys: {report['keys']}")
        print(f"Episodes: {report.get('num_episodes', 'N/A')}")
        print(f"Total steps: {report.get('total_steps', 'N/A')}")
    else:
        print(f"Num files: {report.get('num_files', 0)}")
        print(f"Schema keys: {report.get('schema_keys', [])}")
        print(f"Total steps: {report.get('total_steps', 'N/A')}")
        c = report.get("consistency", {})
        print(f"Consistent: {c.get('all_files_have_same_keys', 'N/A')}")
    print(f"Obs dim: {report.get('obs_dim', 'N/A')}")
    print(f"Action dim: {report.get('action_dim', 'N/A')}")


if __name__ == "__main__":
    main()
