"""Stage 20-U1: Learned executor wrapper for Stage 20-S LSTM checkpoint.

Loads a goal-conditioned LSTM checkpoint and provides step-wise action inference
with history buffering and feature construction matching the training pipeline.

Usage:
    executor = Stage20LearnedExecutor(checkpoint_path, dataset_dir, device="cpu")
    action = executor.predict(obs, phase_id)  # step-wise
    executor.reset()  # clear history for new episode
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlm_nsm.training.train_stage20_goal_conditioned_rnn import GoalConditionedRNN

# Stage 20 feature dimensions (matching builder constants)
ROLE_STATE_DIM = 20
GOAL_DELTA_DIM = 16
REACHED_GOAL_DELTA_DIM = 20

# OBS index constants (matching robosuite_schema 11-dim obs)
_OBS_IDX = {
    "eef_x": 0, "eef_y": 1, "eef_z": 2,
    "gripper_open": 3, "gripper_has_object": 4,
    "cube1_x": 5, "cube1_y": 6, "cube1_z": 7,
    "cube2_x": 8, "cube2_y": 9, "cube2_z": 10,
}
_OBS_DIM = 11
_ACTION_DIM = 7
_NUM_PHASES = 6
_HISTORY_LEN = 4


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class Stage20LearnedExecutor:
    """Wraps a Stage 20 goal-conditioned LSTM checkpoint for step-wise inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        dataset_dir: str | Path,
        device: str = "cpu",
        history_len: int = _HISTORY_LEN,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.dataset_dir = Path(dataset_dir)
        self.device = _resolve_device(device)
        self.history_len = history_len

        # Load checkpoint
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
        self._config = ckpt.get("config", {})
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Detect heads
        use_gripper_head = any(k.startswith("gripper_head.") for k in state_dict)
        use_gripper_mag = any(k.startswith("gripper_magnitude_head.") for k in state_dict)
        use_holding = any(k.startswith("holding_head.") for k in state_dict)

        input_dim = int(self._config.get("input_dim", 110))
        action_dim = int(self._config.get("action_dim", _ACTION_DIM))

        self.model = GoalConditionedRNN(
            input_dim=input_dim,
            action_dim=action_dim,
            model_type=self._config.get("model_type", "lstm"),
            hidden_dim=int(self._config.get("hidden_dim", 256)),
            num_layers=int(self._config.get("num_layers", 1)),
            dropout=float(self._config.get("dropout", 0.0)),
            use_gripper_head=use_gripper_head,
            use_gripper_magnitude_head=use_gripper_mag,
            use_holding_head=use_holding,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.has_holding_head = use_holding
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Load normalization stats
        norm_path = self.dataset_dir / "normalization_stats.json"
        if norm_path.exists():
            self._norm_stats = json.loads(norm_path.read_text())
        else:
            self._norm_stats = {}

        # Load dataset summary for constants
        summary_path = self.dataset_dir / "summary.json"
        self._top_object = "cube1"
        self._bottom_object = "cube1"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            # Default cube sizes
            self._top_size = np.array([0.04, 0.04, 0.04], dtype=np.float32)
            self._bottom_size = np.array([0.04, 0.04, 0.04], dtype=np.float32)

        # History buffer
        self._obs_history: list[np.ndarray] = []
        self._phase_history: list[int] = []

    def reset(self):
        """Clear history buffer for a new episode."""
        self._obs_history.clear()
        self._phase_history.clear()

    def predict(self, obs: np.ndarray, phase_id: int, top_object: str = "cube1",
                bottom_object: str = "cube1",
                return_features: bool = False) -> dict[str, Any]:
        """Predict action for a single step, maintaining internal history.

        Args:
            obs: 11-dim observation vector matching robosuite_schema.
            phase_id: Current phase ID (0-5).
            top_object: Which cube is the top (pick target).
            bottom_object: Which cube is the bottom (place target).
            return_features: If True, include the 110-dim normalized feature sequence
                and raw obs/phase history in the result dict.

        Returns:
            dict with keys: action, action_norm, finite, holding_logit (if head present).
            If return_features=True, also includes: features (normalized), obs_padded,
            phase_padded, history_warm, n_avail.
        """
        obs_f32 = np.asarray(obs, dtype=np.float32).ravel()
        if len(obs_f32) < _OBS_DIM:
            raise ValueError(f"obs must be at least {_OBS_DIM}-dim, got {len(obs_f32)}")

        self._obs_history.append(obs_f32[:_OBS_DIM])
        self._phase_history.append(phase_id)

        # Keep only last history_len steps
        if len(self._obs_history) > self.history_len:
            self._obs_history = self._obs_history[-self.history_len:]
            self._phase_history = self._phase_history[-self.history_len:]

        # Build padded sequence if not enough history
        n_avail = len(self._obs_history)
        obs_padded = np.zeros((self.history_len, _OBS_DIM), dtype=np.float32)
        phase_padded = np.zeros(self.history_len, dtype=np.int64)

        for i in range(n_avail):
            obs_padded[self.history_len - n_avail + i] = self._obs_history[i]
            phase_padded[self.history_len - n_avail + i] = self._phase_history[i]
        # Fill missing history by repeating first available observation
        if n_avail < self.history_len:
            first_obs = self._obs_history[0]
            first_phase = self._phase_history[0]
            for i in range(self.history_len - n_avail):
                obs_padded[i] = first_obs
                phase_padded[i] = first_phase

        # Build features (goal_conditioned mode)
        features = self._build_features(obs_padded, phase_padded, top_object, bottom_object)

        # Forward pass
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, H, D)
            pred = self.model(x).squeeze(0).cpu().numpy()  # (7,)

        result: dict[str, Any] = {
            "action": pred.astype(np.float32),
            "action_norm": float(np.linalg.norm(pred)),
            "finite": bool(np.isfinite(pred).all()),
        }

        # Stage 20-Y: expose features for DAgger recording
        if return_features:
            result["features"] = features.astype(np.float32)  # (H, 110) normalized
            result["obs_padded"] = obs_padded.astype(np.float32)  # (H, 11)
            result["phase_padded"] = phase_padded.astype(np.int64)  # (H,)
            result["history_warm"] = n_avail >= self.history_len
            result["n_avail"] = n_avail

        # Holding head (optional, for diagnostics only)
        if self.has_holding_head:
            with torch.no_grad():
                logits = self.model.predict_holding_logits(x)
                result["holding_logit"] = float(logits.squeeze().cpu().item())

        return result

    def _build_features(
        self,
        obs_seq: np.ndarray,  # (H, 11)
        phase_seq: np.ndarray,  # (H,)
        top_object: str,
        bottom_object: str,
    ) -> np.ndarray:
        """Build goal-conditioned feature sequence matching training format."""
        H = self.history_len
        features_parts = []

        # obs_history: (H, 11) — raw
        obs_norm = self._norm_array("obs_history", obs_seq)
        features_parts.append(obs_norm)

        # role_state: (H, 20)
        role = np.zeros((H, ROLE_STATE_DIM), dtype=np.float32)
        for t in range(H):
            role[t] = self._role_state_vector(obs_seq[t], top_object, bottom_object)
        role_norm = self._norm_array("role_state_history", role)
        features_parts.append(role_norm)

        # goal_delta + mask: simplified proxy (H, 16 each)
        goal = np.zeros((H, GOAL_DELTA_DIM), dtype=np.float32)
        goal_mask = np.zeros((H, GOAL_DELTA_DIM), dtype=np.float32)
        # Fill reach-to-grasp delta from gripper to top cube
        for t in range(H):
            top_idx = _OBS_IDX[f"{top_object}_x"]
            gripper_xyz = obs_seq[t, :3]
            top_xyz = obs_seq[t, top_idx:top_idx + 3]
            goal[t, :3] = top_xyz - gripper_xyz
            goal_mask[t, :3] = 1.0
        goal_norm = self._norm_array("goal_delta_history", goal)
        features_parts.append(goal_norm)
        features_parts.append(goal_mask)

        # reached_goal_delta + mask: (H, 20 each)
        reached = np.zeros((H, REACHED_GOAL_DELTA_DIM), dtype=np.float32)
        reached_mask = np.zeros((H, REACHED_GOAL_DELTA_DIM), dtype=np.float32)
        for t in range(H):
            reached[t, :3] = obs_seq[t, :3]  # gripper xyz
            reached_mask[t, :3] = 1.0
        reached_norm = self._norm_array("reached_goal_delta_history", reached)
        features_parts.append(reached_norm)
        features_parts.append(reached_mask)

        # progress_scalar: (H, 1) — proxy: 1.0 if phase > 0 else 0.0
        progress = np.zeros((H, 1), dtype=np.float32)
        for t in range(H):
            progress[t, 0] = 1.0 if phase_seq[t] > 0 else 0.0
        features_parts.append(progress)

        # phase one-hot: (H, 6)
        phase_oh = np.zeros((H, _NUM_PHASES), dtype=np.float32)
        for t in range(H):
            if 0 <= phase_seq[t] < _NUM_PHASES:
                phase_oh[t, phase_seq[t]] = 1.0
        features_parts.append(phase_oh)

        features = np.concatenate(features_parts, axis=1)  # (H, D)
        return features.astype(np.float32)

    def _role_state_vector(self, obs: np.ndarray, top_object: str, bottom_object: str) -> np.ndarray:
        """Build 20-dim role state from a single observation."""
        top_idx = _OBS_IDX[f"{top_object}_x"]
        bottom_idx = _OBS_IDX[f"{bottom_object}_x"]
        vec = np.zeros(ROLE_STATE_DIM, dtype=np.float32)
        vec[0:3] = obs[0:3]  # gripper xyz
        vec[3] = obs[3]  # gripper_open
        vec[4:7] = obs[top_idx:top_idx + 3]  # top xyz
        vec[7:10] = obs[bottom_idx:bottom_idx + 3]  # bottom xyz
        vec[10:13] = obs[top_idx:top_idx + 3] - obs[0:3]  # rel gripper-to-top
        vec[13:16] = obs[top_idx:top_idx + 3] - obs[bottom_idx:bottom_idx + 3]  # rel top-to-bottom
        vec[16:19] = self._top_size  # top size xyz
        vec[19] = float(self._bottom_size[0])  # bottom size scalar
        return vec

    def _norm_array(self, key: str, arr: np.ndarray) -> np.ndarray:
        """Apply normalization stats if available."""
        stats = self._norm_stats.get(key)
        if stats is None:
            return arr
        mean = np.array(stats.get("mean", 0.0), dtype=np.float32)
        std = np.array(stats.get("std", 1.0), dtype=np.float32)
        # Broadcast to match arr shape
        while mean.ndim < arr.ndim:
            mean = mean[np.newaxis, :]
            std = std[np.newaxis, :]
        return (arr - mean) / np.maximum(std, 1e-8)

    @property
    def is_warm(self) -> bool:
        return len(self._obs_history) >= self.history_len

    def __repr__(self) -> str:
        return (f"Stage20LearnedExecutor(checkpoint={self.checkpoint_path.name}, "
                f"device={self.device}, input_dim={self.input_dim}, "
                f"action_dim={self.action_dim}, has_holding={self.has_holding_head})")
