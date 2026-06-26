"""BC diagnostics: per-phase MSE, action distribution, and endpoint error analysis.

Computes per-phase MSE for train and validation sets, compares
policy output distribution vs dataset action distribution, and
computes endpoint (EEF-to-target) distance errors.

Usage:
    python3 -m vlm_nsm.learning.bc_diagnostics \\
        --checkpoint path/to/best.pt \\
        --demo_npz path/to/trajectories.npz \\
        --output_dir generated/report_tables/stage_7B-3

Outputs:
    - bc_per_phase_mse.md
    - bc_action_distribution.md
    - bc_endpoint_errors.md (with --action_norm_config)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.dataset import RobosuiteDemoDataset
from vlm_nsm.learning.train_bc import MLPBCPolicy


PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    bar = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [sep, bar]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    return "\n".join(lines)


@torch.no_grad()
def compute_per_phase_mse(
    model: nn.Module,
    dataset: RobosuiteDemoDataset,
    device: str = "cpu",
    loss_fn: nn.Module = nn.MSELoss(),
) -> dict:
    """Compute MSE per phase.

    Returns dict mapping phase_name -> {"n_samples": int, "mse": float}.
    """
    model.eval()
    # Phase mapping requires phase_id in dataset
    if not dataset.condition_on_phase or dataset.phase_id is None:
        # Fallback: compute overall MSE only
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        total_loss = 0.0
        n = 0
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            pred = model(obs)
            loss = loss_fn(pred, action)
            total_loss += loss.item() * len(obs)
            n += len(obs)
        return {"overall": {"n_samples": n, "mse": total_loss / max(n, 1)}}

    # Per-phase computation
    phase_samples: dict[int, list[float]] = {p: [] for p in range(6)}
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        pred = model(obs)

        # Compute per-sample MSE
        diff = (pred - action).pow(2).mean(dim=1).cpu().numpy()  # (B,)

        # We need to recover global steps to look up phase_id
        # Since DataLoader shuffles, we need a non-shuffled loader
        # Use the flat_indices to track phase
        pass

    # Instead: iterate through dataset directly for phase accuracy
    phase_squared_errors: dict[int, list[float]] = {p: [] for p in range(6)}
    n_total = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        obs = sample["obs"].unsqueeze(0).to(device)
        action = sample["action"].unsqueeze(0).to(device)
        pred = model(obs)
        mse = (pred - action).pow(2).mean().item()

        # Find phase from global step
        global_step = dataset._flat_indices[idx]
        if dataset.phase_id is not None and global_step < len(dataset.phase_id):
            phase = int(dataset.phase_id[global_step])
            if 0 <= phase < 6:
                phase_squared_errors[phase].append(mse)
        n_total += 1

    result = {}
    for phase_idx in range(6):
        errors = phase_squared_errors[phase_idx]
        name = PHASE_NAMES[phase_idx] if phase_idx < len(PHASE_NAMES) else f"phase_{phase_idx}"
        if errors:
            result[name] = {
                "n_samples": len(errors),
                "mse": float(np.mean(errors)),
                "std": float(np.std(errors)),
            }
        else:
            result[name] = {"n_samples": 0, "mse": 0.0, "std": 0.0}

    return result


@torch.no_grad()
def compute_action_distribution(
    model: nn.Module,
    dataset: RobosuiteDemoDataset,
    device: str = "cpu",
) -> dict:
    """Compare dataset action distribution vs policy output distribution."""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_dataset_actions: list[np.ndarray] = []
    all_policy_actions: list[np.ndarray] = []

    for batch in loader:
        obs = batch["obs"].to(device)
        action = batch["action"].cpu().numpy()
        pred = model(obs).cpu().numpy()
        all_dataset_actions.append(action)
        all_policy_actions.append(pred)

    dataset_actions = np.concatenate(all_dataset_actions, axis=0)
    policy_actions = np.concatenate(all_policy_actions, axis=0)

    action_dim = dataset_actions.shape[1]
    result = {}
    for d in range(action_dim):
        ds_mean = float(np.mean(dataset_actions[:, d]))
        ds_std = float(np.std(dataset_actions[:, d]))
        pol_mean = float(np.mean(policy_actions[:, d]))
        pol_std = float(np.std(policy_actions[:, d]))
        note = ""
        if abs(ds_std) < 1e-4:
            note = "near-zero dataset variance"
        elif abs(pol_std) < 1e-4 and ds_std > 1e-2:
            note = "policy collapses to constant"
        elif abs(pol_mean - ds_mean) > 2 * ds_std:
            note = "policy mean >> dataset std"

        result[f"dim_{d}"] = {
            "dataset_mean": ds_mean,
            "dataset_std": ds_std,
            "policy_mean": pol_mean,
            "policy_std": pol_std,
            "note": note,
        }

    return result


@torch.no_grad()
def compute_endpoint_errors(
    model: nn.Module,
    dataset: RobosuiteDemoDataset,
    device: str = "cpu",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
) -> dict:
    """Compute EEF-to-target endpoint errors per phase.

    For each phase in each episode:
      - Rollout BC policy from phase start using predicted actions
      - Compute distance from final predicted EEF position to phase target
      - Compare with ground-truth final distance

    Phase targets:
      - reach_pick / grasp: cube1 position (obs[5:8])
      - lift: cube1 position + clearance_z
      - reach_place / release: cube2 position + clearance_z
      - retreat: N/A (no target, returns 0)

    Returns:
        dict mapping phase_name -> { endpoint_error: {dataset, policy}, ... }
    """
    model.eval()
    CLEARANCE = 0.15  # lift / place clearance height

    # Default normalization (identity)
    am = np.zeros(7, dtype=np.float32) if action_mean is None else action_mean
    as_ = np.ones(7, dtype=np.float32) if action_std is None else action_std

    # dataset.obs_dim is the raw obs dim (11); phase one-hot is only in __getitem__
    raw_obs_dim = dataset.obs_dim
    PHASE_NAMES_LOCAL = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
    phase_ds_errors: dict[int, list[float]] = {p: [] for p in range(6)}
    phase_policy_errors: dict[int, list[float]] = {p: [] for p in range(6)}
    phase_end_samples: dict[int, int] = {p: 0 for p in range(6)}

    # Iterate episodes to find phase endpoints
    for ep_idx in dataset._episode_indices:
        start, end = dataset._episodes[ep_idx]
        T = end - start
        if T < 2:
            continue

        # Get phase ids for this episode
        if dataset.phase_id is not None and start < len(dataset.phase_id):
            ep_phase_ids = dataset.phase_id[start:end]
        else:
            continue  # need phase ids

        # Get episode obs and actions
        ep_obs = dataset.obs[start:end]  # (T, raw_obs_dim) or (T, raw_obs_dim+phase)
        ep_action = dataset.action[start:end]  # (T, 7) unnormalized

        # Find phase boundaries
        boundaries: list[tuple[int, int, int]] = []  # (phase_id, seg_start, seg_end)
        cur_phase = int(ep_phase_ids[0])
        seg_start = 0
        for t in range(1, T):
            p = int(ep_phase_ids[t])
            if p != cur_phase:
                boundaries.append((cur_phase, seg_start, t))
                cur_phase = p
                seg_start = t
        boundaries.append((cur_phase, seg_start, T))

        for phase_id, seg_start, seg_end in boundaries:
            seg_len = seg_end - seg_start
            if seg_len < 1:
                continue

            # Ground truth: last step of phase
            last_idx = seg_end - 1
            gt_obs = ep_obs[last_idx]  # raw obs (without phase one-hot)
            if dataset.condition_on_phase:
                gt_obs = gt_obs[:raw_obs_dim]

            gt_eef = gt_obs[0:3]

            # Determine target position
            target = _get_phase_target(phase_id, gt_obs, CLEARANCE)
            if target is None:
                continue

            # Dataset endpoint distance
            ds_dist = float(np.linalg.norm(gt_eef - target))
            phase_ds_errors[phase_id].append(ds_dist)

            # Policy rollout from phase start
            sim_eef = ep_obs[seg_start, 0:3].copy()  # initial EEF position
            for t in range(seg_start, seg_end):
                # Build obs for model: raw obs + phase one-hot if needed
                raw_obs_t = ep_obs[t]
                if dataset.condition_on_phase:
                    raw_obs_t = raw_obs_t[:raw_obs_dim]
                model_obs = torch.from_numpy(raw_obs_t).float().unsqueeze(0).to(device)
                if dataset.condition_on_phase:
                    oh = torch.zeros(dataset._n_phases, dtype=torch.float32, device=device)
                    oh[int(ep_phase_ids[t])] = 1.0
                    model_obs = torch.cat([model_obs, oh.unsqueeze(0)], dim=1)

                pred_action_norm = model(model_obs).squeeze(0).cpu().numpy()
                # Denormalize delta xyz
                delta_xyz = pred_action_norm[:3] * as_[:3] + am[:3]
                sim_eef = sim_eef + delta_xyz

                # For next step, get cube positions from dataset obs
                # (important: cube positions may change mid-phase)

            # Policy endpoint distance
            policy_dist = float(np.linalg.norm(sim_eef - target))
            phase_policy_errors[phase_id].append(policy_dist)
            phase_end_samples[phase_id] += 1

    # Build result
    result = {}
    for pid in range(6):
        name = PHASE_NAMES_LOCAL[pid]
        ds_errs = np.array(phase_ds_errors[pid]) if phase_ds_errors[pid] else np.array([0.0])
        pol_errs = np.array(phase_policy_errors[pid]) if phase_policy_errors[pid] else np.array([0.0])
        n = phase_end_samples[pid]
        result[name] = {
            "n_phase_endpoints": n,
            "dataset_endpoint_error": {
                "mean": float(np.mean(ds_errs)),
                "std": float(np.std(ds_errs)),
                "p50": float(np.median(ds_errs)),
                "p95": float(np.percentile(ds_errs, 95)),
            },
            "policy_endpoint_error": {
                "mean": float(np.mean(pol_errs)),
                "std": float(np.std(pol_errs)),
                "p50": float(np.median(pol_errs)),
                "p95": float(np.percentile(pol_errs, 95)),
            },
        }
    return result


def _get_phase_target(
    phase_id: int, obs: np.ndarray, clearance: float = 0.15
) -> np.ndarray | None:
    """Return target position for a given phase, or None if not applicable."""
    cube1_pos = obs[5:8]
    cube2_pos = obs[8:11]
    if phase_id == 0:        # reach_pick
        return cube1_pos.copy()
    elif phase_id == 1:      # grasp
        return cube1_pos.copy()
    elif phase_id == 2:      # lift
        return cube1_pos + np.array([0.0, 0.0, clearance])
    elif phase_id == 3:      # reach_place
        return cube2_pos + np.array([0.0, 0.0, clearance])
    elif phase_id == 4:      # release
        return cube2_pos + np.array([0.0, 0.0, clearance])
    elif phase_id == 5:      # retreat
        return None
    return None


def export_endpoint_errors(errors: dict, output_path: Path):
    """Export endpoint errors as markdown table."""
    headers = ["Phase", "N Endpoints", "DS Err Mean", "DS Err P95",
               "Policy Err Mean", "Policy Err P95", "Delta Mean"]
    rows = []
    for phase in ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]:
        d = errors.get(phase)
        if d is None or d["n_phase_endpoints"] == 0:
            continue
        ds = d["dataset_endpoint_error"]
        ps = d["policy_endpoint_error"]
        delta = ps["mean"] - ds["mean"]
        rows.append([
            phase,
            str(d["n_phase_endpoints"]),
            f"{ds['mean']:.4f}",
            f"{ds['p95']:.4f}",
            f"{ps['mean']:.4f}",
            f"{ps['p95']:.4f}",
            f"{delta:+.4f}",
        ])

    sections = [
        "# BC Endpoint Errors (EEF-to-Target Distance)\n",
        "Distance in meters at phase endpoints. Policy rollout uses chained "
        "BC predictions within each phase.\n",
        _md_table(headers, rows),
        "\n## Interpretation\n",
        "- **DS Err** = ground-truth dataset distance (how close the demonstrator got)",
        "- **Policy Err** = BC policy rollout distance (how close BC gets)",
        "- **Delta** = Policy - Dataset (positive = BC worse than demonstrator)",
        "- Large delta on reach phases indicates the policy fails to converge "
        "to the target object.\n",
    ]
    output_path.write_text("\n".join(sections))
    return output_path


def export_per_phase_mse(
    train_mse: dict,
    val_mse: dict,
    output_path: Path,
):
    """Export per-phase MSE as markdown table."""
    all_phases = sorted(set(list(train_mse.keys()) + list(val_mse.keys())))
    headers = ["Phase", "Train N", "Train MSE", "Val N", "Val MSE", "Main Error Dim(s)"]
    rows = []
    for phase in all_phases:
        t = train_mse.get(phase, {})
        v = val_mse.get(phase, {})
        rows.append([
            phase,
            str(t.get("n_samples", 0)),
            f"{t.get('mse', 0):.6f}",
            str(v.get("n_samples", 0)),
            f"{v.get('mse', 0):.6f}",
            "all" if phase == "overall" else "—",
        ])

    sections = [
        "# BC Per-Phase MSE\n",
        _md_table(headers, rows),
    ]
    output_path.write_text("\n".join(sections))
    return output_path


def export_action_distribution(
    dist: dict,
    output_path: Path,
):
    """Export action distribution comparison as markdown table."""
    headers = ["Action Dim", "Dataset Mean", "Dataset Std", "Policy Mean", "Policy Std", "Note"]
    rows = []
    for dim_key in sorted(dist.keys()):
        d = dist[dim_key]
        rows.append([
            dim_key,
            f"{d['dataset_mean']:.6f}",
            f"{d['dataset_std']:.6f}",
            f"{d['policy_mean']:.6f}",
            f"{d['policy_std']:.6f}",
            d.get("note", ""),
        ])

    sections = [
        "# BC Action Distribution\n",
        _md_table(headers, rows),
    ]
    output_path.write_text("\n".join(sections))
    return output_path


def main():
    parser = argparse.ArgumentParser(description="BC diagnostics")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC model checkpoint")
    parser.add_argument("--demo_npz", type=str, default=None,
                        help="Path to demonstration npz (for dataset)")
    parser.add_argument("--demo_dir", type=str, default=None,
                        help="Directory of per-episode npz files")
    parser.add_argument("--obs_key", type=str, default="filtered_obs")
    parser.add_argument("--action_key", type=str, default="action")
    parser.add_argument("--phase_key", type=str, default="phase_id")
    parser.add_argument("--condition_on_phase", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--min_action_std", type=float, default=1e-3)
    # Endpoint diagnostics
    parser.add_argument("--action_norm_config", type=str, default=None,
                        help="Path to action_norm_stats.json for denormalizing BC outputs")
    parser.add_argument("--skip_endpoint", action="store_true", default=False,
                        help="Skip endpoint error computation (saves time)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load checkpoint to get model dimensions
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        obs_dim = checkpoint.get("obs_dim", 11)
        action_dim = checkpoint.get("action_dim", 7)
        hidden_dims = checkpoint.get("hidden_dims", [256, 256])
        condition_on_phase = checkpoint.get("condition_on_phase", args.condition_on_phase)
    else:
        state_dict = checkpoint
        # Infer dims from state_dict weights
        obs_dim = state_dict["net.0.weight"].shape[1]
        action_dim = state_dict["net.4.weight"].shape[0]
        hidden_dims = [state_dict["net.0.weight"].shape[0]]
        if "net.2.weight" in state_dict:
            hidden_dims.append(state_dict["net.2.weight"].shape[0])
        condition_on_phase = args.condition_on_phase

    model = MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Load dataset
    if args.demo_npz is not None:
        train_ds, val_ds = RobosuiteDemoDataset.train_val(
            args.demo_npz, val_ratio=0.2, seed=42,
            obs_key=args.obs_key,
            condition_on_phase=args.condition_on_phase,
            phase_key=args.phase_key,
            min_action_std=args.min_action_std,
        )
    elif args.demo_dir is not None:
        train_ds, val_ds = RobosuiteDemoDataset.from_directory(
            args.demo_dir, val_ratio=0.2, seed=42,
            obs_key=args.obs_key,
            condition_on_phase=args.condition_on_phase,
            phase_key=args.phase_key,
            min_action_std=args.min_action_std,
        )
    else:
        parser.error("Either --demo_npz or --demo_dir required")

    print(f"Train episodes: {train_ds.num_episodes}, Val episodes: {val_ds.num_episodes}")

    # Per-phase MSE
    print("Computing per-phase MSE...")
    train_mse = compute_per_phase_mse(model, train_ds, device)
    val_mse = compute_per_phase_mse(model, val_ds, device)
    train_overall = train_mse.get("overall", {})
    if "overall" not in train_mse and train_mse:
        # Compute overall from per-phase
        n_total = sum(v.get("n_samples", 0) for v in train_mse.values())
        mse_total = sum(v.get("n_samples", 0) * v.get("mse", 0) for v in train_mse.values())
        train_overall = {"n_samples": n_total, "mse": mse_total / max(n_total, 1)}
    val_overall = val_mse.get("overall", {})
    if "overall" not in val_mse and val_mse and "overall" not in val_mse:
        n_total = sum(v.get("n_samples", 0) for v in val_mse.values())
        mse_total = sum(v.get("n_samples", 0) * v.get("mse", 0) for v in val_mse.values())
        val_overall = {"n_samples": n_total, "mse": mse_total / max(n_total, 1)}
    print(f"  Train overall MSE: {train_overall.get('mse', 0):.6f}")
    print(f"  Val overall MSE:   {val_overall.get('mse', 0):.6f}")
    for phase in PHASE_NAMES:
        if phase in train_mse:
            print(f"  {phase}: train={train_mse[phase]['mse']:.6f} (n={train_mse[phase]['n_samples']})"
                  f" val={val_mse.get(phase, {}).get('mse', 0):.6f}")

    # Action distribution
    print("Computing action distribution...")
    action_dist = compute_action_distribution(model, train_ds, device)

    # Endpoint errors (optional, requires action norm config)
    endpoint_errors: dict | None = None
    if not args.skip_endpoint and args.action_norm_config is not None:
        config_path = Path(args.action_norm_config)
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            action_mean_ep = np.array(cfg.get("action_mean", [0.0]*7), dtype=np.float32)
            action_std_ep = np.array(cfg.get("action_std", [1.0]*7), dtype=np.float32)
            print(f"Computing endpoint errors (action norm from {config_path})...")
            endpoint_errors = compute_endpoint_errors(
                model, val_ds, device,
                action_mean=action_mean_ep, action_std=action_std_ep,
            )
        else:
            print(f"Warning: action_norm_config not found at {config_path}, skipping endpoint errors")

    # Output directory
    output_dir = ensure_dir(Path(args.output_dir)) if args.output_dir else ensure_dir(
        PROJECT_ROOT / "generated" / "report_tables" / "stage_7B-2")

    # Export per-phase MSE
    mse_path = output_dir / "bc_per_phase_mse.md"
    export_per_phase_mse(train_mse, val_mse, mse_path)
    print(f"Per-phase MSE -> {mse_path}")

    # Export action distribution
    dist_path = output_dir / "bc_action_distribution.md"
    export_action_distribution(action_dist, dist_path)
    print(f"Action distribution -> {dist_path}")

    # Export endpoint errors if computed
    if endpoint_errors is not None:
        ep_path = output_dir / "bc_endpoint_errors.md"
        export_endpoint_errors(endpoint_errors, ep_path)
        print(f"Endpoint errors -> {ep_path}")
        for phase, data in endpoint_errors.items():
            if data["n_phase_endpoints"] > 0:
                print(f"  {phase}: DS_err={data['dataset_endpoint_error']['mean']:.4f} "
                      f"Policy_err={data['policy_endpoint_error']['mean']:.4f}")

    # Also save JSON
    json_path = output_dir / "bc_diagnostics.json"
    json_data: dict = {
        "per_phase_mse": {"train": {k: v for k, v in train_mse.items()},
                          "val": {k: v for k, v in val_mse.items()}},
        "action_distribution": action_dist,
    }
    if endpoint_errors is not None:
        json_data["endpoint_errors"] = endpoint_errors
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Diagnostics JSON -> {json_path}")


if __name__ == "__main__":
    main()
