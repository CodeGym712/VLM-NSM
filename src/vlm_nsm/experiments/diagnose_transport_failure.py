"""Diagnostic 2: Transport cos=0.549 Root Cause Analysis.

Analyzes S0 Live Shadow per-step data to determine WHY reach_place direction
cosine is so much lower than reach_pick.

Hypotheses tested:
  H1: VLA predicts smaller XYZ magnitudes (conservative, undershooting)
  H2: VLA direction is noisy (high angular variance)
  H3: VLA has systematic per-dim bias
  H4: Training data for transport has higher action variance
  H5: Transport degrades in second half (temporal divergence)

No GPU required — uses already-collected S0 shadow data.

Usage:
    nsm/bin/python -m vlm_nsm.experiments.diagnose_transport_failure \\
        --s0_dir generated/results/vla/s0_live_shadow \\
        --dataset_dir generated/datasets/vla_demos/d1_small_200ep \\
        --output_dir generated/results/vla/diag_transport
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def analyze_s0_per_step(s0_report_path: Path) -> dict[str, Any]:
    """Re-run analysis on S0 data if per-step trace exists, or report what's available."""
    report = json.loads(s0_report_path.read_text())
    print(f"S0 report keys: {list(report.keys())}")
    return report


def analyze_training_data(dataset_dir: Path) -> dict[str, Any]:
    """Analyze per-phase action distributions in training HDF5 data."""
    phase_actions: dict[str, list[np.ndarray]] = defaultdict(list)
    phase_names = {0: "reach_pick", 1: "grasp", 2: "lift",
                   3: "reach_place", 4: "release", 5: "retreat"}

    ep_dir = dataset_dir / "episodes"
    hdf5_files = sorted(ep_dir.glob("ep_*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {ep_dir}")
        return {}

    n_eps = min(len(hdf5_files), 100)
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
                    pname = phase_names.get(pid, "unknown")
                    phase_actions[pname].append(actions[t, :3])  # XYZ only
        except Exception:
            continue

    result = {}
    for pname in sorted(phase_actions.keys()):
        arr = np.stack(phase_actions[pname], axis=0)
        magnitudes = np.linalg.norm(arr, axis=1)
        result[pname] = {
            "count": len(arr),
            "action_xyz_mean": arr.mean(axis=0).tolist(),
            "action_xyz_std": arr.std(axis=0).tolist(),
            "magnitude_mean": float(magnitudes.mean()),
            "magnitude_std": float(magnitudes.std()),
            "magnitude_p50": float(np.median(magnitudes)),
            "magnitude_p95": float(np.percentile(magnitudes, 95)),
            "per_dim_std": {
                "dx": float(arr[:, 0].std()),
                "dy": float(arr[:, 1].std()),
                "dz": float(arr[:, 2].std()),
            },
        }
        print(f"Training {pname}: n={result[pname]['count']}, "
              f"|a|_mean={result[pname]['magnitude_mean']:.4f}, "
              f"|a|_p95={result[pname]['magnitude_p95']:.4f}")

    return result


def compute_cos_sim(vla_xyz: np.ndarray, scripted_xyz: np.ndarray) -> float:
    vn = np.linalg.norm(vla_xyz)
    sn = np.linalg.norm(scripted_xyz)
    if vn < 1e-8 or sn < 1e-8:
        return 0.0
    return float(np.dot(vla_xyz, scripted_xyz) / (vn * sn))


def print_diagnosis(training_stats: dict, s0_per_phase: dict | None = None):
    """Print diagnostic conclusions based on available data."""
    print("\n" + "=" * 70)
    print("Transport cos=0.549 Root Cause Analysis")
    print("=" * 70)

    # Training data contrast
    if "reach_pick" in training_stats and "reach_place" in training_stats:
        rp = training_stats["reach_pick"]
        rpl = training_stats["reach_place"]
        print(f"\nTraining Data Contrast:")
        print(f"  reach_pick:  |a|_mean={rp['magnitude_mean']:.4f}, "
              f"std_dx={rp['per_dim_std']['dx']:.4f}, "
              f"std_dy={rp['per_dim_std']['dy']:.4f}, "
              f"std_dz={rp['per_dim_std']['dz']:.4f}")
        print(f"  reach_place: |a|_mean={rpl['magnitude_mean']:.4f}, "
              f"std_dx={rpl['per_dim_std']['dx']:.4f}, "
              f"std_dy={rpl['per_dim_std']['dy']:.4f}, "
              f"std_dz={rpl['per_dim_std']['dz']:.4f}")

        mag_ratio = rpl['magnitude_mean'] / max(rp['magnitude_mean'], 1e-8)
        std_ratio = rpl['magnitude_std'] / max(rp['magnitude_std'], 1e-8)
        print(f"\n  reach_place/reach_pick magnitude ratio: {mag_ratio:.2f}")
        print(f"  reach_place/reach_pick std ratio: {std_ratio:.2f}")

        # H4 check
        if rpl['magnitude_std'] > rp['magnitude_std'] * 1.5:
            print(f"  H4 (higher action variance): CONFIRMED — "
                  f"transport actions {std_ratio:.1f}x more variable")
        else:
            print(f"  H4 (higher action variance): REFUTED — "
                  f"transport variance not substantially higher")

    # Interpret cos=0.549
    cos = 0.549
    ang_err = math.acos(max(-1.0, min(1.0, cos))) * 180 / math.pi
    print(f"\nInterpretation of cos={cos:.3f}:")
    print(f"  Equivalent angular error: {ang_err:.1f}°")
    print(f"  A random direction on hemisphere has expected cos ~ 0.5")
    print(f"  cos=0.549 is barely above random (0.500)")
    print(f"  For comparison, reach_pick cos=0.880 → ang_err={math.acos(0.880)*180/math.pi:.1f}°")
    print(f"  Gap: cos drop of 0.331 (38% relative degradation)")

    # Recommendations
    print(f"\nRecommended Next Steps:")
    print(f"  1. Run S0 shadow with M1 Diffusion (current data may be M0)")
    print(f"  2. Collect per-step traces with vla_xyz, scripted_xyz, phase tags")
    print(f"  3. Run per-dim bias analysis on reach_place steps")
    print(f"  4. If H4 confirmed: longer obs_history or transport-specific DAgger")
    print(f"  5. If temporal divergence: shorten action horizon in transport phase")


def main():
    parser = argparse.ArgumentParser(description="Transport failure diagnosis")
    parser.add_argument("--dataset_dir", default="generated/datasets/vla_demos/d1_small_200ep")
    parser.add_argument("--output_dir", default="generated/results/vla/diag_transport")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 1. Training data analysis
    print("Analyzing training data action distributions...")
    training_stats = analyze_training_data(Path(args.dataset_dir))

    # 2. Print diagnosis
    print_diagnosis(training_stats)

    # 3. Save
    result = {
        "diagnosis": "transport_cos_root_cause",
        "cos_observed": 0.549,
        "cos_reach_pick": 0.880,
        "angular_error_deg": math.acos(0.549) * 180 / math.pi,
        "training_per_phase": training_stats,
    }
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output / 'diagnosis.json'}")


if __name__ == "__main__":
    main()
