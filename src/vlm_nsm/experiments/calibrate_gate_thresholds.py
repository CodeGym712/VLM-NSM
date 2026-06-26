"""Diagnostic 3: Per-Phase Gate Calibration.

Uses S0 Shadow data (VLA predicts alongside scripted, no control) to compute
data-driven per-phase gate thresholds. For each phase, sets threshold at the
P95 of ||VLA_xyz - scripted_xyz||, so the gate accepts 95% of in-distribution
VLA actions.

No GPU required — reads existing S0 or training data.

Usage:
    nsm/bin/python -m vlm_nsm.experiments.calibrate_gate_thresholds \\
        --dataset_dir generated/datasets/vla_demos/d1_small_200ep \\
        --output_dir generated/results/vla/gate_calibration
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


_PHASE_NAMES = {0: "reach_pick", 1: "grasp", 2: "lift",
                3: "reach_place", 4: "release", 5: "retreat"}


def compute_phase_deviations_from_training(
    dataset_dir: Path,
    n_eps: int = 50,
) -> dict[str, list[float]]:
    """Compute per-phase action magnitude distribution from training data.

    For each phase, collect the per-step action delta magnitudes.
    The P95 of this distribution serves as a data-driven gate threshold:
    VLA actions within P95 of training action magnitude are considered
    in-distribution.

    This is a PROXY method — ideally we'd use S0 shadow data where we have
    actual VLA predictions, but the training data gives us the teacher's
    per-phase action distribution for calibration.
    """
    phase_magnitudes: dict[str, list[float]] = defaultdict(list)

    ep_dir = dataset_dir / "episodes"
    hdf5_files = sorted(ep_dir.glob("ep_*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files in {ep_dir}")
        return {}

    for ep_path in hdf5_files[:n_eps]:
        try:
            with h5py.File(ep_path, "r") as f:
                if not bool(f.attrs.get("success", False)):
                    continue
                actions = f["actions_4d"][:]
                phase_ids = f["symbolic/phase_id"][:]
                T = min(len(actions), len(phase_ids))
                for t in range(T):
                    pid = int(phase_ids[t])
                    pname = _PHASE_NAMES.get(pid, "unknown")
                    if pname == "unknown":
                        continue
                    xyz = actions[t, :3]
                    mag = float(np.linalg.norm(xyz))
                    phase_magnitudes[pname].append(mag)
        except Exception:
            continue

    return phase_magnitudes


def calibrate(phase_magnitudes: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Compute per-phase gate thresholds from magnitude distributions."""
    result = {}
    for pname in sorted(phase_magnitudes.keys()):
        mags = np.array(phase_magnitudes[pname])
        if len(mags) < 10:
            continue
        result[pname] = {
            "count": int(len(mags)),
            "mean": float(mags.mean()),
            "std": float(mags.std()),
            "p50": float(np.median(mags)),
            "p75": float(np.percentile(mags, 75)),
            "p90": float(np.percentile(mags, 90)),
            "p95": float(np.percentile(mags, 95)),
            "p99": float(np.percentile(mags, 99)),
            "max": float(mags.max()),
            # Recommended gate threshold (P95, with 1.5x safety factor
            # since VLA may have higher variance than teacher)
            "recommended_gate_xyz_threshold": float(np.percentile(mags, 95) * 1.5),
        }
    return result


def print_calibration(calib: dict[str, dict[str, float]]):
    """Print calibration table."""
    print("\n" + "=" * 80)
    print("Per-Phase Gate Calibration (from training data action distributions)")
    print("=" * 80)
    header = (f"{'Phase':<16} {'N':>7} {'Mean':>8} {'P50':>8} "
              f"{'P95':>8} {'Curr':>8} {'Calib':>8}")
    print(header)
    print("-" * 80)

    current_global = 0.025  # current global max_xyz_deviation
    for pname, stats in sorted(calib.items()):
        print(f"{pname:<16} {stats['count']:>7} {stats['mean']:>8.4f} "
              f"{stats['p50']:>8.4f} {stats['p95']:>8.4f} "
              f"{current_global:>8.4f} {stats['recommended_gate_xyz_threshold']:>8.4f}")

    print("-" * 80)
    print(f"\nCurrent gate: max_xyz_deviation = {current_global:.4f} (global)")
    print(f"Calibrated gate: per-phase P95 × 1.5 safety factor")

    # Check which phases would change most
    for pname, stats in sorted(calib.items()):
        new_val = stats["recommended_gate_xyz_threshold"]
        if new_val > current_global * 1.5:
            print(f"  {pname}: {current_global:.4f} → {new_val:.4f} "
                  f"(relax significantly, +{new_val/current_global:.1f}x)")
        elif new_val < current_global * 0.5:
            print(f"  {pname}: {current_global:.4f} → {new_val:.4f} "
                  f"(tighten significantly, {new_val/current_global:.1f}x)")
        elif abs(new_val - current_global) < 0.003:
            print(f"  {pname}: {current_global:.4f} → {new_val:.4f} (similar)")
        else:
            change = "loosen" if new_val > current_global else "tighten"
            print(f"  {pname}: {current_global:.4f} → {new_val:.4f} ({change})")


def main():
    parser = argparse.ArgumentParser(description="Gate calibration")
    parser.add_argument("--dataset_dir", default="generated/datasets/vla_demos/d1_small_200ep")
    parser.add_argument("--output_dir", default="generated/results/vla/gate_calibration")
    parser.add_argument("--n_eps", type=int, default=50,
                       help="Max training episodes to scan for calibration")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Computing per-phase action distributions from {args.dataset_dir}...")
    phase_magnitudes = compute_phase_deviations_from_training(
        Path(args.dataset_dir), n_eps=args.n_eps)
    calib = calibrate(phase_magnitudes)
    print_calibration(calib)

    # Save
    (output / "calibration.json").write_text(
        json.dumps(calib, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output / 'calibration.json'}")

    # Generate per-phase gate config
    gate_config = {
        "use_confidence": False,
        "per_phase_max_xyz_deviation": {
            phase: float(stats["recommended_gate_xyz_threshold"])
            for phase, stats in calib.items()
        },
        "global_max_xyz_deviation": 0.025,
        "max_gripper_deviation": 0.5,
        "calibration_source": str(args.dataset_dir),
        "calibration_method": "p95 × 1.5 safety factor from training data",
    }
    (output / "gate_config.json").write_text(
        json.dumps(gate_config, indent=2, ensure_ascii=False))
    print(f"Gate config saved to {output / 'gate_config.json'}")


if __name__ == "__main__":
    main()
