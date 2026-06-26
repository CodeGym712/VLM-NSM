"""Stage 22-A: Fast RL environment — optimized feature building without executor.predict()."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector

_HISTORY_LEN = 4
_FEATURE_DIM = 110
_ACTION_DIM = 7
_OBS_DIM = 11
_NUM_PHASES = 6

# Feature layout offsets
# obs(11) + role(20) + goal(16) + goal_mask(16) + reached(20) + reached_mask(20) + progress(1) + phase_oh(6) = 110
_OBS_END = 11
_ROLE_END = 31
_GOAL_END = 47
_GMASK_END = 63
_REACHED_END = 83
_RMASK_END = 103
_PROG_END = 104  # +1
_PHASE_END = 110  # +6


def _build_features_fast(
    obs_seq: np.ndarray,      # (H, 11)
    phase_ids: np.ndarray,     # (H,)
    norm_stats: dict,          # normalization stats from dataset
    top_object: str = "cube1",
    bottom_object: str = "cube2",
    history_len: int = _HISTORY_LEN,
) -> np.ndarray:
    """Build 110-dim features without executor.predict() overhead.

    Uses cached normalization stats for fast feature construction.
    """
    H = len(obs_seq)
    H_out = history_len

    # Pad if needed
    if H < H_out:
        obs_padded = np.zeros((H_out, _OBS_DIM), dtype=np.float32)
        phase_padded = np.zeros(H_out, dtype=np.int64)
        obs_padded[H_out - H:] = obs_seq
        phase_padded[H_out - H:] = phase_ids
        obs_padded[:H_out - H] = obs_seq[0]
        phase_padded[:H_out - H] = phase_ids[0]
    else:
        obs_padded = obs_seq[-H_out:].astype(np.float32)
        phase_padded = phase_ids[-H_out:].astype(np.int64)

    # Object indices
    top_idx = 5 if top_object == "cube1" else 8
    bottom_idx = 8 if top_object == "cube1" else 5

    features = np.zeros((H_out, _FEATURE_DIM), dtype=np.float32)

    for t in range(H_out):
        o = obs_padded[t]
        # obs: (11,) — raw
        features[t, :11] = o

        # role_state: (20,)
        # gripper_xyz(3) + gripper_open(1) + top_xyz(3) + bottom_xyz(3) +
        # rel_gripper_to_top(3) + rel_top_to_bottom(3) + top_size(3) + bottom_size(1)
        top_size = np.array([0.04, 0.04, 0.04], dtype=np.float32)
        bottom_size = 0.04
        features[t, 11:14] = o[0:3]                          # gripper xyz
        features[t, 14] = o[3]                                # gripper_open
        features[t, 15:18] = o[top_idx:top_idx+3]             # top xyz
        features[t, 18:21] = o[bottom_idx:bottom_idx+3]       # bottom xyz
        features[t, 21:24] = o[top_idx:top_idx+3] - o[0:3]   # rel gripper-to-top
        features[t, 24:27] = o[top_idx:top_idx+3] - o[bottom_idx:bottom_idx+3]  # rel top-to-bottom
        features[t, 27:30] = top_size                          # top size
        features[t, 30] = bottom_size                          # bottom size

        # goal_delta: (16,) — gripper-to-top delta
        features[t, 31:34] = o[top_idx:top_idx+3] - o[0:3]   # gripper_delta_xyz
        features[t, 34] = 0.0                                  # gripper_open_delta
        features[t, 35] = 0.0                                  # target_top_z_delta

        # goal_delta_mask: (16,) — mask for gripper delta
        features[t, 47:50] = 1.0  # gripper delta mask

        # reached_goal_delta: (20,)
        features[t, 63:66] = o[0:3]   # gripper xyz delta
        features[t, 66] = o[3]         # gripper_open delta
        features[t, 67:70] = o[top_idx:top_idx+3]    # top xyz delta
        features[t, 70:73] = o[bottom_idx:bottom_idx+3]  # bottom xyz delta

        # reached_goal_delta_mask: (20,)
        features[t, 83:86] = 1.0   # gripper mask
        features[t, 86:89] = 1.0   # top mask
        features[t, 89:92] = 1.0   # bottom mask

        # progress_scalar: (1,)
        features[t, 103] = 1.0 if phase_padded[t] > 0 else 0.0

        # phase_one_hot: (6,)
        p = phase_padded[t]
        if 0 <= p < _NUM_PHASES:
            features[t, 104 + p] = 1.0

    # Apply normalization
    features = _apply_norm(features, norm_stats)
    return features.astype(np.float32)


def _apply_norm(features: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Apply per-key normalization matching training pipeline."""
    # The training uses these keys for normalization
    key_map = {
        "obs_history": (0, 11),
        "role_state_history": (11, 31),
        "goal_delta_history": (31, 47),
        "reached_goal_delta_history": (63, 83),
    }
    for key, (start, end) in key_map.items():
        stats = norm_stats.get(key)
        if stats is None:
            continue
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.maximum(np.array(stats["std"], dtype=np.float32), 1e-8)
        # Broadcast: (D,) → (H, D)
        features[:, start:end] = (features[:, start:end] - mean) / std
    return features


class FastStackRLEnv:
    """Fast RL env — builds features directly without executor.predict()."""

    def __init__(
        self,
        dataset_dir: str,
        exec_config_path: str,
        goal_id: str = "red_on_green",
        seed: int = 0,
        max_steps: int = 500,
    ):
        self.goal_id = goal_id
        self.seed = seed
        self.max_steps = max_steps
        self.direction_id = 0 if goal_id == "red_on_green" else 1
        self.top_object = "cube1" if goal_id == "red_on_green" else "cube2"
        self.bottom_object = "cube2" if goal_id == "red_on_green" else "cube1"

        # Load normalization stats once
        norm_path = Path(dataset_dir) / "normalization_stats.json"
        self._norm_stats = json.loads(norm_path.read_text()) if norm_path.exists() else {}

        # Load exec config
        config = json.loads(Path(exec_config_path).read_text())
        self._kp = config.get("kp", 0.05)
        self._max_delta = config.get("max_delta", 0.02)

        self._env: RobosuiteEnvWrapper | None = None
        self._step_count = 0
        self._obs_history: list[np.ndarray] = []
        self._phase_history: list[int] = []
        self._ever_lift = False
        self._init_top_z = 0.0
        self._init_bottom_z = 0.0
        self._prev_gripper_open = 0.05

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed = seed
        if self._env is not None:
            try: self._env.env.close()
            except: pass

        self._env = RobosuiteEnvWrapper(
            env_name="Stack", robots="Panda",
            cube_names=["cube1", "cube2"], render_mode="none")
        self._env.reset()
        self._step_count = 0
        self._obs_history.clear()
        self._phase_history.clear()
        self._ever_lift = False
        self._prev_gripper_open = 0.05

        obs = self._get_obs()
        self._init_top_z = obs[7] if self.top_object == "cube1" else obs[10]
        self._init_bottom_z = obs[10] if self.top_object == "cube1" else obs[7]

        # Warm up history
        for _ in range(_HISTORY_LEN):
            self._obs_history.append(obs)
            self._phase_history.append(0)

        features = _build_features_fast(
            np.array(self._obs_history[-_HISTORY_LEN:]),
            np.array(self._phase_history[-_HISTORY_LEN:], dtype=np.int64),
            self._norm_stats, self.top_object, self.bottom_object)
        return features, self._get_info(False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._env is None:
            raise RuntimeError("Not reset")

        action = np.clip(np.asarray(action, dtype=np.float64).ravel()[:7], -1.5, 1.5)
        if not np.isfinite(action).all():
            action = np.zeros(7, dtype=np.float64)

        self._step_count += 1

        # Execute
        self._env.step(action)
        obs = self._get_obs()
        self._obs_history.append(obs)

        # Phase detection
        phase = self._detect_phase(obs)
        self._phase_history.append(phase)

        # Build features
        features = _build_features_fast(
            np.array(self._obs_history[-_HISTORY_LEN:]),
            np.array(self._phase_history[-_HISTORY_LEN:], dtype=np.int64),
            self._norm_stats, self.top_object, self.bottom_object)

        # Reward
        reward = self._compute_reward(obs, action)

        # Termination
        task_success = self._check_success(obs)
        terminated = task_success
        truncated = self._step_count >= self.max_steps
        if task_success:
            reward += 20.0

        self._prev_gripper_open = obs[3]
        return features, reward, terminated, truncated, self._get_info(task_success)

    def _get_obs(self) -> np.ndarray:
        try:
            return build_robosuite_obs_vector(
                self._env.get_eef_pose(), self._env.get_gripper_state(),
                self._env.get_all_object_poses(), cubes=["cube1", "cube2"])
        except:
            return np.zeros(_OBS_DIM, dtype=np.float32)

    def _detect_phase(self, obs: np.ndarray) -> int:
        has_obj = obs[4] > 0.5
        top_z = obs[7] if self.top_object == "cube1" else obs[10]
        top_bottom_z = obs[10] if self.top_object == "cube1" else obs[7]
        gripper_open = obs[3]

        if has_obj and top_z > self._init_top_z + 0.03:
            self._ever_lift = True
            return 2  # lift
        if has_obj and top_z > top_bottom_z + 0.02:
            return 3  # reach_place/transport
        if has_obj:
            return 1  # grasp (holding)
        if gripper_open > 0.03:
            return 0  # reach_pick
        return 1  # grasping

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        r = 0.0
        has_obj = obs[4] > 0.5
        top_z = obs[7] if self.top_object == "cube1" else obs[10]
        top_bottom_z = obs[10] if self.top_object == "cube1" else obs[7]

        # Holding
        if has_obj:
            r += 0.5
            # Lift height
            lift = max(0.0, top_z - self._init_top_z - 0.02)
            r += min(lift * 20.0, 1.0)
            # XY alignment
            top_x, top_y = obs[5:7] if self.top_object == "cube1" else obs[8:10]
            bottom_x, bottom_y = obs[8:10] if self.top_object == "cube1" else obs[5:7]
            xy_dist = np.sqrt((top_x - bottom_x)**2 + (top_y - bottom_y)**2)
            r += max(0.0, 0.5 - xy_dist * 2.0)

        # Gripper closure reward (encourage closing near object)
        if obs[4] < 0.5 and obs[3] < 0.03:
            r += 0.1  # gripper closing

        # Action smoothness
        r -= 0.001 * np.sum(action**2)

        return float(r)

    def _check_success(self, obs: np.ndarray) -> bool:
        top_i = 5 if self.top_object == "cube1" else 8
        bot_i = 8 if self.top_object == "cube1" else 5
        top_z = obs[top_i + 2]
        bot_z = obs[bot_i + 2]
        xy = np.sqrt((obs[top_i]-obs[bot_i])**2 + (obs[top_i+1]-obs[bot_i+1])**2)
        return top_z > bot_z + 0.02 and xy < 0.05

    def _get_info(self, success: bool) -> dict:
        return {"success": success, "step": self._step_count,
                "direction_id": self.direction_id, "ever_lift": self._ever_lift,
                "timeout": self._step_count >= self.max_steps,
                "unsafe_leak": False, "runtime_error": None}

    def close(self):
        if self._env is not None:
            try: self._env.env.close()
            except: pass

    @property
    def feature_dim(self) -> int: return _FEATURE_DIM
    @property
    def action_dim(self) -> int: return _ACTION_DIM
    @property
    def history_len(self) -> int: return _HISTORY_LEN
