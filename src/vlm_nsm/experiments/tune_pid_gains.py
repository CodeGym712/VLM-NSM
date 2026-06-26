"""Grid search PID gains using control_audit episodes for offline evaluation.

For each phase, sweep kp/ki/kd over reasonable ranges and select combination
that minimizes reach error (dist to target) while maximizing convergence rate.

Usage:
    python -m vlm_nsm.experiments.tune_pid_gains \
        --dataset_dir generated/datasets/vla_demos/control_audit_200ep \
        --output generated/configs/control/pid_gain_schedule_tuned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID, GainSchedule, PIDGain,
)

_PHASE_ID_MAP = {0: "reach_pick", 1: "grasp", 2: "lift", 3: "reach_place"}

_SEARCH_RANGES: dict[str, dict] = {
    "reach_pick":   {"kp": [0.04, 0.06, 0.08, 0.10], "ki": [0.0, 0.001, 0.002, 0.005], "kd": [0.0, 0.005, 0.01, 0.02], "max_delta": [0.02]},
    "grasp":        {"kp": [0.03, 0.05, 0.07], "ki": [0.0, 0.001, 0.002], "kd": [0.0, 0.005, 0.008, 0.01], "max_delta": [0.01]},
    "lift":         {"kp": [0.04, 0.06, 0.08, 0.10], "ki": [0.0, 0.001, 0.002], "kd": [0.0, 0.005, 0.01], "max_delta": [0.02]},
    "reach_place":  {"kp": [0.04, 0.06, 0.08], "ki": [0.0, 0.001, 0.002, 0.005], "kd": [0.0, 0.005, 0.01, 0.015, 0.02], "max_delta": [0.015, 0.02]},
}


def replay_episode_offline(episode_path: str, test_gains: dict) -> tuple[float, int]:
    """Replay one episode with given PID gains, evaluating convergence.

    Returns:
        (mean_reach_error, convergence_failures)
    """
    total_error = 0.0
    total_steps = 0
    conv_failures = 0

    with h5py.File(episode_path, "r") as h5:
        eef_pos = h5["proprio/eef_pos"][:]
        phase_ids = h5["symbolic/phase_id"][:]
        n_steps = eef_pos.shape[0]

        # Find phase boundaries
        phase_starts = [0]
        for t in range(1, n_steps):
            if phase_ids[t] != phase_ids[t - 1]:
                phase_starts.append(t)
        phase_starts.append(n_steps)

        pid = OperationalSpacePID()
        for phase_name, g in test_gains.items():
            pid._gain_schedule.gains[phase_name] = PIDGain(**g)

        for ps_idx in range(len(phase_starts) - 1):
            t_start = phase_starts[ps_idx]
            t_end = phase_starts[ps_idx + 1]
            phase_name = _PHASE_ID_MAP.get(int(phase_ids[t_start]), "reach_pick")

            target_eef = eef_pos[min(t_end - 1, n_steps - 1)]
            pid.set_target(pos=target_eef)
            pid.set_phase(phase_name)

            phase_steps = min(t_end - t_start, 500)
            for t in range(t_start, t_start + phase_steps):
                pid.set_current(eef_pos[t])
                v_des = pid.compute_pid_velocity()
                pred_next = eef_pos[t] + v_des * 0.02
                err = float(np.linalg.norm(pred_next - target_eef))
                total_error += err
                total_steps += 1

            final_dist = float(np.linalg.norm(
                eef_pos[min(t_end - 1, n_steps - 1)] - target_eef))
            if final_dist > 0.02:
                conv_failures += 1

    mean_error = total_error / max(total_steps, 1)
    return mean_error, conv_failures


def _generate_combinations(ranges: dict) -> list[dict]:
    """Generate all combinations from per-parameter search ranges."""
    keys = list(ranges.keys())
    values = list(ranges.values())
    combos = [[]]
    for v_list in values:
        new_combos = []
        for combo in combos:
            for v in v_list:
                new_combos.append(combo + [v])
        combos = new_combos
    return [{keys[i]: combo[i] for i in range(len(keys))} for combo in combos]


def grid_search(dataset_dir: str, max_episodes: int = 50) -> dict:
    """Run grid search over per-phase gain ranges."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        episodes = [json.loads(l) for l in f.readlines()]

    success_eps = [e for e in episodes if e.get("success", True)][:max_episodes]
    validate_eps = success_eps[:min(10, len(success_eps))]
    print(f"[tune] {len(success_eps)} episodes, validating on {len(validate_eps)}")

    default_gains = GainSchedule()
    best_gains = {}

    for phase_name, ranges in _SEARCH_RANGES.items():
        print(f"\n[tune] Tuning {phase_name}...")
        candidates = _generate_combinations(ranges)
        print(f"  {len(candidates)} candidates")
        phase_best = None
        phase_best_score = float("inf")

        for candidate in candidates:
            test_gains = {}
            for pn in _SEARCH_RANGES:
                if pn == phase_name:
                    test_gains[pn] = candidate
                else:
                    dg = default_gains.get(pn)
                    test_gains[pn] = {"kp": dg.kp, "ki": dg.ki, "kd": dg.kd,
                                      "max_delta": dg.max_delta, "i_max": dg.i_max}

            total_err = 0.0
            total_cf = 0
            for ep in validate_eps:
                err, cf = replay_episode_offline(ep["episode_path"], test_gains)
                total_err += err
                total_cf += cf

            score = total_err + total_cf * 0.1
            if score < phase_best_score:
                phase_best_score = score
                phase_best = candidate

        print(f"  Best: kp={phase_best['kp']}, ki={phase_best['ki']}, "
              f"kd={phase_best['kd']}, score={phase_best_score:.4f}")
        best_gains[phase_name] = phase_best

    return best_gains


def main():
    parser = argparse.ArgumentParser(description="Tune PID gains via grid search")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_episodes", type=int, default=50)
    args = parser.parse_args()

    best_gains = grid_search(args.dataset_dir, args.max_episodes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(best_gains, f, indent=2)
    print(f"\n[tune] Saved to {out}")


if __name__ == "__main__":
    main()
