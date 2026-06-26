"""Analyze SymbolConditionedObservationFilter statistics from collected demos.

Computes observation dimensionality reduction and basic statistics.

Usage:
    python -m vlm_nsm.experiments.analyze_symbol_filter \\
        --demo_path generated/demos/stacking_3_scripted_lowdim.npz \\
        --meta_path generated/demos/stacking_3_scripted_lowdim_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir


def analyze_filter_stats(
    demo_path: str | Path,
    meta_path: str | Path | None = None,
) -> dict:
    """Analyze symbol-filtered observation statistics.

    Detects full_obs / filtered_obs arrays in the npz if available,
    otherwise falls back to metadata estimates.

    Args:
        demo_path: Path to npz demonstration file.
        meta_path: Optional path to metadata JSON.

    Returns:
        Dict with filter analysis stats.
    """
    demo_path = Path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_path}")

    data = np.load(demo_path)
    meta = None
    if meta_path:
        meta_path = Path(meta_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

    obs = data.get("obs")
    actions = data.get("action")

    if obs is None:
        raise ValueError(f"No 'obs' array found in {demo_path}")

    num_transitions = obs.shape[0] if obs.ndim >= 1 else 0
    filtered_obs_dim = obs.shape[1] if obs.ndim == 2 else (obs.shape[0] if obs.ndim == 1 else 0)

    # Detect full_obs and filtered_obs arrays
    full_obs_available = "full_obs" in data
    full_obs_arr = data.get("full_obs")
    filtered_obs_arr = data.get("filtered_obs")

    full_obs_dim = None
    if full_obs_available and full_obs_arr is not None:
        full_obs_dim = full_obs_arr.shape[1] if full_obs_arr.ndim == 2 else full_obs_arr.shape[0]
        filtered_obs_dim = filtered_obs_arr.shape[1] if filtered_obs_arr is not None and filtered_obs_arr.ndim == 2 else filtered_obs_dim
    elif meta and "obs_dim" in meta:
        full_obs_dim = meta["obs_dim"]
    else:
        full_obs_dim = filtered_obs_dim  # estimate as same

    # Reduction ratio
    reduction_ratio = None
    if full_obs_available and full_obs_dim and filtered_obs_dim and full_obs_dim > 0:
        reduction_ratio = (full_obs_dim - filtered_obs_dim) / full_obs_dim
    elif meta and meta.get("reduction_ratio") is not None:
        reduction_ratio = meta["reduction_ratio"]

    # Step distribution (from skill_id)
    skill_ids = data.get("skill_id")
    step_distribution: dict[str, int] = {}
    if skill_ids is not None:
        unique, counts = np.unique(skill_ids, return_counts=True)
        step_distribution = {f"step_{int(u)}": int(c) for u, c in zip(unique, counts)}

    # Observation statistics
    obs_flat = obs.reshape(-1) if obs.ndim >= 2 else obs
    obs_mean = float(np.mean(obs_flat)) if len(obs_flat) > 0 else 0.0
    obs_std = float(np.std(obs_flat)) if len(obs_flat) > 0 else 0.0

    stats = {
        "full_obs_available": full_obs_available,
        "full_obs_dim": full_obs_dim,
        "filtered_obs_dim": filtered_obs_dim,
        "reduction_ratio": reduction_ratio,
        "num_transitions": num_transitions,
        "step_distribution": step_distribution,
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_dim": actions.shape[1] if actions is not None and actions.ndim == 2 else None,
    }

    if not full_obs_available:
        stats["full_obs_dim_note"] = "estimated from metadata; raw full obs not saved in demo npz"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze symbol filter stats")
    parser.add_argument("--demo_path", required=True, help="Path to demo npz")
    parser.add_argument("--meta_path", default=None, help="Path to metadata JSON")
    args = parser.parse_args()

    stats = analyze_filter_stats(
        demo_path=args.demo_path,
        meta_path=args.meta_path,
    )

    print(json.dumps(stats, indent=2))

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    stats_path = results_dir / "symbol_filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
