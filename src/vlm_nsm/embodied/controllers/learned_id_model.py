"""Learned Inverse Dynamics Model (Layer 2 of hierarchical ID controller).

A lightweight MLP (~25K params) that predicts target velocity from
state transitions. Trained on HDF5 data to learn:
    (eef_pos, gripper, object_poses, phase, desired_next_eef) → [dx, dy, dz, gripper]

Key properties:
  - ~1000x smaller than VLA Diffusion Policy (25K vs 26.8M params)
  - No visual input required — uses proprioceptive state + phase
  - Ensemble of 3 models for uncertainty estimation
  - ~5 min training on CPU, <1ms inference per step
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ── Model ─────────────────────────────────────────────────────────────────────

class LearnedIDModel(nn.Module):
    """Lightweight MLP for inverse dynamics prediction.

    Args:
        input_dim: Input dimension (default: 38).
        output_dim: Output dimension (default: 4 for [dx, dy, dz, gripper]).
        hidden_dim: Hidden layer size.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        input_dim: int = 28,
        output_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict action [dx, dy, dz, gripper] from state vector.

        Args:
            x: (B, input_dim) state vector.

        Returns:
            (B, output_dim) predicted action.
        """
        return self.net(x)

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Predict action from numpy array (single sample or batch).

        Args:
            x: (input_dim,) or (B, input_dim) numpy array.

        Returns:
            (output_dim,) or (B, output_dim) numpy array.
        """
        self.eval()
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32))
            y = self.forward(x_t).cpu().numpy()
        return y[0] if single else y


# ── Ensemble with Uncertainty ────────────────────────────────────────────────

@dataclass
class IDPrediction:
    """Output from Learned ID model inference.

    Attributes:
        action: Mean predicted action [dx, dy, dz, gripper].
        velocity: Target velocity [vx, vy, vz] (same as action[:3] for OSC).
        gripper: Gripper command (-1 open, 1 close).
        confidence: Ensemble agreement (1 = perfect, 0 = total disagreement).
        std: Per-dimension standard deviation across ensemble.
    """

    action: np.ndarray       # (4,)
    velocity: np.ndarray     # (3,)
    gripper: float
    confidence: float
    std: np.ndarray          # (4,)


class IDModelEnsemble:
    """Ensemble of LearnedIDModel for uncertainty-aware prediction.

    Args:
        model_paths: List of paths to trained model checkpoints.
        input_dim: Input dimension.
        output_dim: Output dimension.
    """

    def __init__(
        self,
        model_paths: list[str | Path] | None = None,
        input_dim: int = 28,
        output_dim: int = 4,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models: list[LearnedIDModel] = []
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None

        if model_paths:
            for path in model_paths:
                self.add_model(path)

    def add_model(self, checkpoint_path: str | Path):
        """Load a model from checkpoint and add to ensemble."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = LearnedIDModel(
            input_dim=ckpt.get("input_dim", self.input_dim),
            output_dim=ckpt.get("output_dim", self.output_dim),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.models.append(model)

        # Use stats from first model
        if self.action_mean is None:
            self.action_mean = np.array(ckpt.get("action_mean", [0, 0, 0, 0]), dtype=np.float32)
            self.action_std = np.array(ckpt.get("action_std", [1, 1, 1, 1]), dtype=np.float32)

    def predict(self, x: np.ndarray, confidence_threshold: float = 0.7) -> IDPrediction:
        """Predict action with uncertainty from ensemble.

        Args:
            x: (input_dim,) state vector.
            confidence_threshold: Below this, prediction flagged as low confidence.

        Returns:
            IDPrediction with action, velocity, gripper, confidence, std.
        """
        if not self.models:
            return IDPrediction(
                action=np.zeros(self.output_dim, dtype=np.float32),
                velocity=np.zeros(3, dtype=np.float32),
                gripper=-1.0,
                confidence=0.0,
                std=np.ones(self.output_dim, dtype=np.float32),
            )

        preds = np.stack([m.predict_numpy(x) for m in self.models], axis=0)  # (N_models, 4)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)

        # De-normalize
        if self.action_mean is not None and self.action_std is not None:
            mean = mean * self.action_std + self.action_mean

        confidence = 1.0 / (1.0 + float(np.linalg.norm(std)))

        return IDPrediction(
            action=mean.astype(np.float32),
            velocity=mean[:3].astype(np.float32),
            gripper=float(mean[3]),
            confidence=confidence,
            std=std.astype(np.float32),
        )


# ── Layer 1 + Layer 2 Fusion ─────────────────────────────────────────────────

def hybrid_pid_learned_action(
    pid_action: np.ndarray,
    learned_pred: IDPrediction,
    pid_weight: float = 0.3,
    confidence_threshold: float = 0.7,
) -> tuple[np.ndarray, str]:
    """Fuse PID (Layer 1, safe) and Learned ID (Layer 2, performant).

    Args:
        pid_action: 7-dim action from OperationalSpacePID.
        learned_pred: Prediction from IDModelEnsemble.
        pid_weight: Minimum weight for PID component (0-1).
        confidence_threshold: Below this, use pure PID.

    Returns:
        (action, source) where action is 7-dim and source is
        "pid_fallback", "hybrid", or "no_learned".
    """
    if learned_pred.confidence < confidence_threshold:
        return pid_action, "pid_fallback"

    # Blend: PID floor ensures stability, Learned ID adds performance
    learned_action_7d = np.zeros(7, dtype=np.float32)
    learned_action_7d[:3] = learned_pred.velocity
    learned_action_7d[6] = learned_pred.gripper

    blend = pid_weight * pid_action + (1.0 - pid_weight) * learned_action_7d
    return blend.astype(np.float32), "hybrid"
