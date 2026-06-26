"""Aggregate DAgger round 1 data with expert v2 strict_only data.

Combines reach-only steps (phase_id in {0, 3}) from both sources into a
single npz for DAgger reach BC training. Also computes residual action
targets for residual BC training.

Usage:
    python3 -m vlm_nsm.learning.aggregate_reach_dataset \\
        --expert_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only \\
        --dagger_npz generated/demos/stage_7B-4/dagger_round1_reach/trajectories_dagger_round1_reach.npz \\
        --output_dir generated/demos/stage_7B-4/dagger_reach_aggregated
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vlm_nsm.core.utils import ensure_dir

REACH_PHASE_IDS = {0, 3}  # reach_pick, reach_place
PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def load_expert_reach_data(demo_dir: str) -> dict:
    """Load expert v2 strict_only data and filter to reach phases.

    Returns:
        dict with arrays: filtered_obs, action, phase_id, episode_id
    """
    demo_path = Path(demo_dir)
    npz_files = sorted(demo_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {demo_dir}")

    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    phase_list: list[np.ndarray] = []
    ep_id_list: list[np.ndarray] = []

    for ep_idx, f in enumerate(npz_files):
        data = np.load(f)
        obs = data.get("filtered_obs", data.get("obs"))
        action = data["action"]
        phase_id = data.get("phase_id", data.get("skill_id"))

        if obs is None or phase_id is None:
            print(f"  Warning: {f.name} missing required keys, skipping")
            continue

        # Filter to reach phases only
        reach_mask = np.isin(phase_id, list(REACH_PHASE_IDS))

        obs_list.append(obs[reach_mask])
        action_list.append(action[reach_mask])
        phase_list.append(phase_id[reach_mask].astype(np.int64))
        ep_id_arr = np.full(np.sum(reach_mask), ep_idx, dtype=np.int32)
        ep_id_list.append(ep_id_arr)

    if not obs_list:
        raise ValueError(f"No reach-phase data found in {demo_dir}")

    return {
        "filtered_obs": np.concatenate(obs_list, axis=0).astype(np.float32),
        "action": np.concatenate(action_list, axis=0).astype(np.float32),
        "phase_id": np.concatenate(phase_list, axis=0).astype(np.int64),
        "episode_id": np.concatenate(ep_id_list, axis=0).astype(np.int32),
    }


def load_dagger_data(dagger_npz_path: str) -> dict:
    """Load DAgger round 1 data.

    Returns:
        dict with arrays: filtered_obs, action, bc_action, phase_id, episode_id
    """
    data = np.load(dagger_npz_path)
    result = {
        "filtered_obs": data["filtered_obs"].astype(np.float32),
        "action": data["oracle_scripted_action"].astype(np.float32),
        "bc_action": data["bc_action"].astype(np.float32) if "bc_action" in data else np.zeros_like(data["oracle_scripted_action"]),
        "phase_id": data["phase_id"].astype(np.int64),
        "episode_id": data["episode_id"].astype(np.int32),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Aggregate reach dataset")
    parser.add_argument("--expert_dir", type=str, required=True,
                        help="Expert v2 strict_only directory")
    parser.add_argument("--dagger_npz", type=str, required=True,
                        help="DAgger round 1 npz file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for aggregated npz")
    args = parser.parse_args()

    print("Loading expert v2 strict_only reach data...")
    expert = load_expert_reach_data(args.expert_dir)
    print(f"  Expert: {len(expert['filtered_obs'])} reach steps, "
          f"{len(np.unique(expert['episode_id']))} episodes")
    for pid in sorted(REACH_PHASE_IDS):
        n = np.sum(expert["phase_id"] == pid)
        print(f"    {PHASE_NAMES[pid]}: {n} steps")

    print("Loading DAgger round 1 data...")
    dagger = load_dagger_data(args.dagger_npz)
    print(f"  DAgger: {len(dagger['filtered_obs'])} steps, "
          f"{len(np.unique(dagger['episode_id']))} episodes")
    for pid in sorted(REACH_PHASE_IDS):
        n = np.sum(dagger["phase_id"] == pid)
        print(f"    {PHASE_NAMES[pid]}: {n} steps")

    # Compute residual = oracle_action - bc_action for DAgger data
    dagger_residual = dagger["action"] - dagger["bc_action"]  # (T_d, 7)

    # Expert data: bc_action is not applicable → set residual = 0
    expert_bc = np.zeros_like(expert["action"])
    expert_residual = np.zeros_like(expert["action"])

    # Combine
    combined_obs = np.concatenate([expert["filtered_obs"], dagger["filtered_obs"]], axis=0)
    combined_action = np.concatenate([expert["action"], dagger["action"]], axis=0)
    combined_bc = np.concatenate([expert_bc, dagger["bc_action"]], axis=0)
    combined_residual = np.concatenate([expert_residual, dagger_residual], axis=0)
    combined_phase = np.concatenate([expert["phase_id"], dagger["phase_id"]], axis=0)
    combined_episode = np.concatenate([expert["episode_id"], dagger["episode_id"]], axis=0)

    # Source: 0=expert, 1=dagger
    source_expert = np.zeros(len(expert["filtered_obs"]), dtype=np.uint8)
    source_dagger = np.ones(len(dagger["filtered_obs"]), dtype=np.uint8)
    combined_source = np.concatenate([source_expert, source_dagger], axis=0)

    Total = len(combined_obs)
    print(f"\nAggregated dataset: {Total} total steps")
    print(f"  Expert: {len(expert['filtered_obs'])} ({len(expert['filtered_obs'])/Total*100:.1f}%)")
    print(f"  DAgger: {len(dagger['filtered_obs'])} ({len(dagger['filtered_obs'])/Total*100:.1f}%)")
    for pid in sorted(REACH_PHASE_IDS):
        n = np.sum(combined_phase == pid)
        print(f"    {PHASE_NAMES[pid]}: {n} steps")

    # Validation: action dims
    for name, arr in [("filtered_obs", combined_obs), ("action", combined_action),
                       ("residual_action", combined_residual)]:
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Residual statistics
    res_mean = np.mean(combined_residual, axis=0)
    res_std = np.std(combined_residual, axis=0)
    res_abs_max = np.max(np.abs(combined_residual), axis=0)
    print(f"\nResidual action stats:")
    print(f"  Mean: {np.array2string(res_mean, precision=4, floatmode='fixed')}")
    print(f"  Std:  {np.array2string(res_std, precision=4, floatmode='fixed')}")
    print(f"  Abs max: {np.array2string(res_abs_max, precision=4, floatmode='fixed')}")

    # Save aggregated npz
    output_dir = ensure_dir(Path(args.output_dir))
    output_path = output_dir / "aggregated_reach_dataset.npz"
    np.savez_compressed(
        output_path,
        filtered_obs=combined_obs,
        action=combined_action,  # oracle target for DAgger BC
        bc_action=combined_bc,  # BC action for residual computation
        residual_action=combined_residual,  # oracle - bc for residual BC
        phase_id=combined_phase,
        episode_id=combined_episode,
        source=combined_source,
    )
    print(f"\nAggregated dataset saved to {output_path}")


if __name__ == "__main__":
    main()
