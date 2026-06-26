"""Action space audit: compare dataset, scripted, and BC action distributions.

Quantifies per-dimension and per-phase statistics for:
  1. Dataset actions (raw stored normalized [-1, 1] values)
  2. Dataset actions after BC normalization ((x - mean) / std)
  3. BC model predictions (normalized output space)
  4. BC model predictions after denormalization
  5. Scripted actions estimated from dataset obs (proportional controller)

Usage:
    python3 -m vlm_nsm.learning.action_space_audit \\
        --checkpoint generated/results/stage_7B-3/bc_main/best.pt \\
        --demo_dir generated/demos/stage_7A-14/robosuite_expert_v2_100strict/strict_only \\
        --action_norm_stats generated/results/stage_7B-3/action_norm_stats.json \\
        --output_dir generated/report_tables/stage_7B-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir
from vlm_nsm.learning.dataset import RobosuiteDemoDataset
from vlm_nsm.learning.train_bc import MLPBCPolicy

PHASE_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]
ACTION_DIM_NAMES = ["dx", "dy", "dz", "dOri0", "dOri1", "dOri2", "gripper"]
N_DIMS = 7
N_PHASES = 6

# Proportional controller params (defaults from run_embodied_stacking_robosuite.py)
KP_DEFAULT = 0.05
MAX_DELTA_DEFAULT = 0.02
OSC_POSE_MAX_DELTA = 0.05
LIFT_HEIGHT = 0.05
REACH_CLEARANCE = 0.10


def _norm_action_scripted(dx: float, dy: float, dz: float, gripper: float) -> np.ndarray:
    """Simulate _norm_action from run_embodied_stacking_robosuite.py."""
    return np.array([
        np.clip(dx / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dy / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dz / OSC_POSE_MAX_DELTA, -1.0, 1.0),
        0.0, 0.0, 0.0,
        gripper,
    ], dtype=np.float32)


def _estimate_scripted_action(
    obs: np.ndarray,
    phase_id: int,
    kp: float = KP_DEFAULT,
    max_delta: float = MAX_DELTA_DEFAULT,
) -> np.ndarray:
    """Estimate proportional controller action for given obs and phase.

    Args:
        obs: 11-dim observation [eef_xyz, gripper, has_object, cube1_xyz, cube2_xyz]
        phase_id: 0-5 phase index
        kp: Proportional gain
        max_delta: Max position delta per step (physical)

    Returns:
        7-dim normalized action vector.
    """
    eef = obs[0:3]
    cube1 = obs[5:8]
    cube2 = obs[8:11]
    GRIPPER_OPEN = -1.0
    GRIPPER_CLOSE = 1.0

    if phase_id == 0:  # reach_pick
        target = cube1 + np.array([0.0, 0.0, REACH_CLEARANCE])
        gripper = GRIPPER_OPEN
    elif phase_id == 1:  # grasp
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE], dtype=np.float32)
    elif phase_id == 2:  # lift
        target = cube1 + np.array([0.0, 0.0, REACH_CLEARANCE + LIFT_HEIGHT])
        gripper = GRIPPER_CLOSE
    elif phase_id == 3:  # reach_place
        target = cube2 + np.array([0.0, 0.0, REACH_CLEARANCE])
        gripper = GRIPPER_CLOSE
    elif phase_id == 4:  # release
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN], dtype=np.float32)
    elif phase_id == 5:  # retreat
        target = eef + np.array([0.0, 0.0, 0.15])
        gripper = GRIPPER_OPEN
    else:
        return np.zeros(7, dtype=np.float32)

    dx = np.clip(kp * (target[0] - eef[0]), -max_delta, max_delta)
    dy = np.clip(kp * (target[1] - eef[1]), -max_delta, max_delta)
    dz = np.clip(kp * (target[2] - eef[2]), -max_delta, max_delta)
    return _norm_action_scripted(dx, dy, dz, gripper)


@torch.no_grad()
def _collect_action_samples(
    dataset: RobosuiteDemoDataset,
    model: MLPBCPolicy | None = None,
    device: str = "cpu",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    kp: float = KP_DEFAULT,
    max_delta: float = MAX_DELTA_DEFAULT,
) -> dict:
    """Collect action samples from dataset, BC model, and simulated scripted controller.

    Returns:
        Dict with structure:
          phase_name -> {
            "dataset_raw": (N, 7) array,
            "dataset_normalized": (N, 7) array,
            "bc_normalized": (N, 7) array or None,
            "bc_denormalized": (N, 7) array or None,
            "scripted": (N, 7) array,
          }
    """
    phase_samples: dict[str, dict[str, np.ndarray | None]] = {
        name: {"dataset_raw": [], "dataset_normalized": [], "bc_normalized": [], "bc_denormalized": [], "scripted": []}
        for name in PHASE_NAMES
    }

    # Collect per-phase samples from flat indices
    for idx in range(len(dataset)):
        global_step = dataset._flat_indices[idx]
        raw_obs = dataset.obs[global_step].copy()  # (11,) or (11+phase,)
        if dataset.condition_on_phase and raw_obs.shape[0] > dataset.obs_dim:
            raw_obs = raw_obs[:dataset.obs_dim]

        raw_action = dataset.action[global_step].copy()  # (7,)

        # Normalize action
        if action_mean is not None and action_std is not None:
            normalized_action = (raw_action - action_mean) / action_std
        else:
            normalized_action = raw_action.copy()

        # Phase
        phase = 0
        if dataset.phase_id is not None and global_step < len(dataset.phase_id):
            phase = int(dataset.phase_id[global_step])
        phase_name = PHASE_NAMES[phase] if phase < N_PHASES else PHASE_NAMES[0]

        phase_samples[phase_name]["dataset_raw"].append(raw_action)
        phase_samples[phase_name]["dataset_normalized"].append(normalized_action)

        # Scripted
        scripted = _estimate_scripted_action(raw_obs, phase, kp=kp, max_delta=max_delta)
        phase_samples[phase_name]["scripted"].append(scripted)

        # BC predictions
        if model is not None:
            model_obs = torch.from_numpy(raw_obs).float().unsqueeze(0).to(device)
            if dataset.condition_on_phase:
                oh = torch.zeros(N_PHASES, dtype=torch.float32, device=device)
                oh[phase] = 1.0
                model_obs = torch.cat([model_obs, oh.unsqueeze(0)], dim=1)
            pred_norm = model(model_obs).squeeze(0).cpu().numpy()  # normalized
            phase_samples[phase_name]["bc_normalized"].append(pred_norm)
            # Denormalize
            if action_mean is not None and action_std is not None:
                pred_denorm = pred_norm * action_std + action_mean
                pred_denorm = np.clip(pred_denorm, -1.0, 1.0)
            else:
                pred_denorm = pred_norm.copy()
            phase_samples[phase_name]["bc_denormalized"].append(pred_denorm)

    # Convert lists to arrays
    result = {}
    for pname in PHASE_NAMES:
        entry = {}
        for key in ["dataset_raw", "dataset_normalized", "scripted"]:
            arr = np.array(phase_samples[pname][key], dtype=np.float32)
            entry[key] = arr if len(arr) > 0 else np.empty((0, N_DIMS), dtype=np.float32)
        for key in ["bc_normalized", "bc_denormalized"]:
            arr = np.array(phase_samples[pname][key], dtype=np.float32) if phase_samples[pname][key] else np.empty((0, N_DIMS), dtype=np.float32)
            entry[key] = arr if len(arr) > 0 else None
        result[pname] = entry
    return result


def _dist_stats(arr: np.ndarray) -> dict:
    """Compute distribution statistics per dimension."""
    if arr.shape[0] == 0:
        return {f"dim_{d}": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p5": 0.0, "p95": 0.0} for d in range(N_DIMS)}
    stats = {}
    for d in range(min(arr.shape[1], N_DIMS)):
        vals = arr[:, d]
        stats[f"dim_{d}"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        }
    return stats


def _export_audit_table(audit_data: dict, overall_only: bool = False) -> str:
    """Export action space audit as markdown table.

    Produces a table per action source with per-dimension stats.
    """
    lines = ["# Action Space Audit\n"]
    lines.append("Per-dimension statistics for dataset actions, BC model predictions, ")
    lines.append("and scripted (proportional controller) actions.\n")

    sources = [
        ("dataset_raw", "Dataset (Raw)"),
        ("dataset_normalized", "Dataset (Normalized)"),
        ("bc_normalized", "BC Output (Normalized)"),
        ("bc_denormalized", "BC Output (Denormalized)"),
        ("scripted", "Scripted (Proportional Ctrl)"),
    ]

    if overall_only or True:
        # Combine all phases for overall stats
        combined: dict[str, list[np.ndarray]] = {key: [] for key, _ in sources}
        for pname in PHASE_NAMES:
            entry = audit_data.get(pname, {})
            for key, _ in sources:
                arr = entry.get(key)
                if arr is not None and isinstance(arr, np.ndarray) and arr.shape[0] > 0:
                    combined[key].append(arr)

        lines.append("## Overall (All Phases)\n")
        for key, label in sources:
            if not combined[key]:
                lines.append(f"### {label}\n\nNo data available.\n")
                continue
            arr = np.concatenate(combined[key], axis=0)
            stats = _dist_stats(arr)
            headers = ["Dim", "Mean", "Std", "Min", "Max", "p5", "p95"]
            rows = []
            for d in range(N_DIMS):
                s = stats[f"dim_{d}"]
                rows.append([
                    f"{ACTION_DIM_NAMES[d]} (dim{d})",
                    f"{s['mean']:.4f}", f"{s['std']:.4f}",
                    f"{s['min']:.4f}", f"{s['max']:.4f}",
                    f"{s['p5']:.4f}", f"{s['p95']:.4f}",
                ])
            lines.append(f"### {label} (N={arr.shape[0]})\n")
            lines.append(__md_table(headers, rows))
            lines.append("")

    # Per-phase tables
    lines.append("## Per-Phase Breakdown\n")
    for pname in PHASE_NAMES:
        entry = audit_data.get(pname, {})
        lines.append(f"### {pname}\n")
        for key, label in sources:
            arr = entry.get(key)
            if arr is None or (isinstance(arr, np.ndarray) and arr.shape[0] == 0):
                continue
            stats = _dist_stats(arr)
            headers = ["Dim", "Mean", "Std", "Min", "Max", "p5", "p95"]
            rows = []
            for d in range(N_DIMS):
                s = stats[f"dim_{d}"]
                rows.append([
                    f"{ACTION_DIM_NAMES[d]} (dim{d})",
                    f"{s['mean']:.4f}", f"{s['std']:.4f}",
                    f"{s['min']:.4f}", f"{s['max']:.4f}",
                    f"{s['p5']:.4f}", f"{s['p95']:.4f}",
                ])
            lines.append(f"**{label}** (N={arr.shape[0]})\n\n{__md_table(headers, rows)}\n")

    return "\n".join(lines)


def _export_summary_table(audit_data: dict) -> str:
    """Export compact summary table comparing action sources per dim."""
    lines = ["# Action Space Audit Summary\n"]
    lines.append("Mean ± Std per dimension across all action sources.\n")

    # Overall comparison table
    combined: dict[str, np.ndarray] = {}
    for key in ["dataset_raw", "dataset_normalized", "bc_normalized", "bc_denormalized", "scripted"]:
        arrs = []
        for pname in PHASE_NAMES:
            a = audit_data.get(pname, {}).get(key)
            if a is not None and isinstance(a, np.ndarray) and a.shape[0] > 0:
                arrs.append(a)
        combined[key] = np.concatenate(arrs, axis=0) if arrs else np.empty((0, N_DIMS))

    for d in range(N_DIMS):
        lines.append(f"### {ACTION_DIM_NAMES[d]} (dim {d})\n")
        headers = ["Source", "Mean", "Std", "Min", "Max", "p5", "p95"]
        rows = []
        for key, label in [
            ("dataset_raw", "Dataset (Raw)"),
            ("dataset_normalized", "Dataset (Normalized)"),
            ("bc_normalized", "BC Output (Norm)"),
            ("bc_denormalized", "BC Output (Denorm)"),
            ("scripted", "Scripted"),
        ]:
            arr = combined[key]
            if arr.shape[0] == 0:
                continue
            vals = arr[:, d]
            rows.append([
                label,
                f"{float(np.mean(vals)):.4f}",
                f"{float(np.std(vals)):.4f}",
                f"{float(np.min(vals)):.4f}",
                f"{float(np.max(vals)):.4f}",
                f"{float(np.percentile(vals, 5)):.4f}",
                f"{float(np.percentile(vals, 95)):.4f}",
            ])
        lines.append(__md_table(headers, rows))
        lines.append("")

    return "\n".join(lines)


def __md_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    bar = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [sep, bar]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Action space audit")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to BC model checkpoint")
    parser.add_argument("--demo_dir", type=str, required=True,
                        help="Directory of per-episode npz files")
    parser.add_argument("--action_norm_stats", type=str, required=True,
                        help="Path to action_norm_stats.json")
    parser.add_argument("--obs_key", type=str, default="filtered_obs")
    parser.add_argument("--phase_key", type=str, default="phase_id")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--kp", type=float, default=KP_DEFAULT,
                        help="Proportional gain for scripted action estimate")
    parser.add_argument("--max_delta", type=float, default=MAX_DELTA_DEFAULT,
                        help="Max position delta for scripted action estimate")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        obs_dim = checkpoint.get("obs_dim", 11)
        action_dim = checkpoint.get("action_dim", 7)
        hidden_dims = checkpoint.get("hidden_dims", [256, 256])
        condition_on_phase = checkpoint.get("condition_on_phase", False)
    else:
        state_dict = checkpoint
        obs_dim = state_dict["net.0.weight"].shape[1]
        action_dim = state_dict["net.4.weight"].shape[0]
        hidden_dims = [state_dict["net.0.weight"].shape[0]]
        if "net.2.weight" in state_dict:
            hidden_dims.append(state_dict["net.2.weight"].shape[0])
        condition_on_phase = False

    # Infer condition_on_phase from obs_dim (17 = 11 raw + 6 phase)
    if not condition_on_phase and obs_dim > 11:
        condition_on_phase = True
        print(f"  Inferred condition_on_phase=True from obs_dim={obs_dim}")

    model = MLPBCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}, hidden={hidden_dims}, "
          f"condition_on_phase={condition_on_phase}")

    # Load action norm stats
    norm_path = Path(args.action_norm_stats)
    if not norm_path.exists():
        print(f"Error: action_norm_stats not found at {norm_path}")
        return
    norm_cfg = json.loads(norm_path.read_text())
    action_mean = np.array(norm_cfg.get("action_mean", [0.0] * N_DIMS), dtype=np.float32)
    action_std = np.array(norm_cfg.get("action_std", [1.0] * N_DIMS), dtype=np.float32)
    print(f"Action mean: {action_mean}")
    print(f"Action std:  {action_std}")

    # Load dataset
    train_ds, val_ds = RobosuiteDemoDataset.from_directory(
        args.demo_dir, val_ratio=0.2, seed=42,
        obs_key=args.obs_key,
        condition_on_phase=True,
        phase_key=args.phase_key,
    )
    print(f"Train episodes: {train_ds.num_episodes}, steps: {len(train_ds)}")
    print(f"Val episodes:   {val_ds.num_episodes}, steps: {len(val_ds)}")

    # Collect action samples (use val set to avoid overfitting bias in stats)
    print("Collecting action samples...")
    audit_data = _collect_action_samples(
        val_ds, model=model, device=device,
        action_mean=action_mean, action_std=action_std,
        kp=args.kp, max_delta=args.max_delta,
    )

    # Print summary stats
    print("\n--- Per-Phase Sample Counts ---")
    for pname in PHASE_NAMES:
        n = audit_data[pname]["dataset_raw"].shape[0]
        n_bc = audit_data[pname]["bc_normalized"].shape[0] if audit_data[pname]["bc_normalized"] is not None else 0
        print(f"  {pname}: {n} dataset steps, {n_bc} BC predictions")

    # Print key findings
    print("\n--- Mean ± Std Per Dimension (Overall) ---")
    for key, label in [
        ("dataset_raw", "Dataset (Raw)"),
        ("dataset_normalized", "Dataset (Normalized)"),
        ("bc_normalized", "BC (Normalized)"),
        ("bc_denormalized", "BC (Denormalized)"),
        ("scripted", "Scripted"),
    ]:
        all_arrs = []
        for pname in PHASE_NAMES:
            a = audit_data[pname].get(key)
            if a is not None and isinstance(a, np.ndarray) and a.shape[0] > 0:
                all_arrs.append(a)
        if not all_arrs:
            continue
        full = np.concatenate(all_arrs, axis=0)
        means = np.mean(full, axis=0)
        stds = np.std(full, axis=0)
        print(f"  {label:25s}: mean={np.array2string(means, precision=4, floatmode='fixed')}  "
              f"std={np.array2string(stds, precision=4, floatmode='fixed')}")

    # Export markdown tables
    output_dir = ensure_dir(Path(args.output_dir)) if args.output_dir else ensure_dir(
        Path(args.checkpoint).parent.parent / "report_tables" / "stage_7B-4")

    detail_path = output_dir / "action_space_audit_detail.md"
    detail_path.write_text(_export_audit_table(audit_data))
    print(f"\nDetailed audit -> {detail_path}")

    summary_path = output_dir / "action_space_audit_summary.md"
    summary_path.write_text(_export_summary_table(audit_data))
    print(f"Summary table -> {summary_path}")

    # Save JSON
    json_path = output_dir / "action_space_audit.json"
    json_data: dict = {"per_phase_stats": {}}
    for pname in PHASE_NAMES:
        json_data["per_phase_stats"][pname] = {}
        for key in ["dataset_raw", "dataset_normalized", "scripted"]:
            arr = audit_data[pname][key]
            json_data["per_phase_stats"][pname][key] = _dist_stats(arr)
        for key in ["bc_normalized", "bc_denormalized"]:
            arr = audit_data[pname].get(key)
            if arr is not None:
                json_data["per_phase_stats"][pname][key] = _dist_stats(arr)
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"Audit JSON -> {json_path}")


if __name__ == "__main__":
    main()
