"""VLA Shadow Evaluation — compare VLA predictions against scripted teacher.

Runs scripted episodes while VLA predicts actions in parallel (shadow mode).
Measures per-phase action MSE, gripper sign accuracy, and chunk consistency.

Usage:
    MUJOCO_GL=egl nsm/bin/python -m vlm_nsm.experiments.eval_vla_shadow \\
        --checkpoint generated/results/vla/vla_m0_smoke/best.pt \\
        --num_episodes 10 --device cpu
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlm_nsm.neuro_symbolic.symbolic_condition import SymbolicConditionEncoder
from vlm_nsm.vla.models.vision_encoder import VLABCModel

_PHASE_NAMES = [
    "reach_pick", "grasp", "lift", "reach_place", "release", "retreat",
]


def load_model(checkpoint_path: str, device: torch.device) -> VLABCModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = VLABCModel(
        backbone=config.get("backbone", "resnet18"),
        pretrained=False,
        freeze_bn=False,
        vision_output_dim=config.get("vision_dim", 256),
        symbolic_dim=config.get("symbolic_dim", 42),
        hidden_dims=tuple(config.get("hidden_dims", [512, 256])),
        action_dim=4,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_shadow_eval(
    checkpoint_path: str,
    dataset_dir: str,
    num_episodes: int = 10,
    device: str = "cpu",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run shadow evaluation: collect per-phase action prediction metrics.

    This loads the VLA dataset (pre-collected episodes), runs inference
    on each sample, and compares VLA predictions to teacher actions.
    """
    device_obj = torch.device(device)
    model = load_model(checkpoint_path, device_obj)

    # Load normalization stats
    stats_path = Path(dataset_dir) / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if action_mean is None:
            action_mean = np.array(stats["action_mean"], dtype=np.float32)
        if action_std is None:
            action_std = np.array(stats["action_std"], dtype=np.float32)

    # Load dataset
    from vlm_nsm.vla.data.vla_dataset import VLADataset
    ds = VLADataset(
        dataset_dir, split="val",
        obs_history_len=1, action_chunk_len=1,
        stride=1, success_only=False,
    )

    per_phase_errors: dict[str, list[float]] = defaultdict(list)
    per_phase_counts: dict[str, int] = defaultdict(int)
    gripper_correct: list[float] = []
    all_mse: list[float] = []
    phase_name_map = _PHASE_NAMES

    count = 0
    max_samples = min(len(ds), num_episodes * 100)  # ~100 steps per episode
    stride = max(1, len(ds) // max_samples)

    with torch.no_grad():
        for idx in range(0, len(ds), stride):
            if count >= max_samples:
                break
            sample = ds[idx]
            av = sample["agentview"].unsqueeze(0).to(device_obj)
            fv = sample["frontview"].unsqueeze(0).to(device_obj)
            prop = sample["proprio"].unsqueeze(0).to(device_obj)
            sym = sample["symbolic_cond"].unsqueeze(0).to(device_obj)
            target = sample["action_chunk"]
            if target.dim() == 3:
                target = target[0, :]  # (chunk, 4) if multi-action
            target = target.to(device_obj)

            pred = model(av, fv, prop, sym)
            pred_np = pred.cpu().numpy().ravel()
            target_np = target.cpu().numpy().ravel()

            # Denormalize if stats available
            if action_mean is not None and action_std is not None:
                pred_np = pred_np * action_std + action_mean
                target_np = target_np * action_std + action_mean

            # Per-dim MSE
            mse = float(np.mean((pred_np - target_np) ** 2))
            all_mse.append(mse)

            phase_id = sample["phase_id"]
            if isinstance(phase_id, torch.Tensor):
                phase_id = int(phase_id.item())
            elif isinstance(phase_id, np.ndarray):
                phase_id = int(phase_id.item())
            else:
                phase_id = int(phase_id)

            phase_name = phase_name_map[phase_id] if 0 <= phase_id < len(phase_name_map) else f"phase_{phase_id}"
            per_phase_errors[phase_name].append(mse)
            per_phase_counts[phase_name] += 1

            # Gripper sign accuracy (dim 3 in 4D action = gripper)
            pred_gripper = pred_np[3]
            target_gripper = target_np[3]
            pred_sign = 1 if pred_gripper > 0 else -1
            target_sign = 1 if target_gripper > 0 else -1
            gripper_correct.append(1.0 if pred_sign == target_sign else 0.0)

            count += 1

    if count == 0:
        return {"error": "no_samples"}

    per_phase_summary = {}
    for phase, errors in sorted(per_phase_errors.items()):
        per_phase_summary[phase] = {
            "mse": float(np.mean(errors)),
            "count": per_phase_counts[phase],
        }

    return {
        "num_samples": count,
        "overall_mse": float(np.mean(all_mse)),
        "gripper_sign_accuracy": float(np.mean(gripper_correct)),
        "per_phase": per_phase_summary,
        "device": str(device_obj),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VLA Shadow Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_shadow_eval(
        checkpoint_path=args.checkpoint,
        dataset_dir=args.dataset_dir,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
