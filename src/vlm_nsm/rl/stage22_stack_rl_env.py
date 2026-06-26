"""Stage 22-A1: RL environment wrapper for Robosuite stacking.

Wraps the existing Robosuite execution pipeline as a Gymnasium-like RL env.
Provides 110-dim normalized features as observations and shaped rewards.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vlm_nsm.core.data_types import Action
from vlm_nsm.embodied.stage20_learned_executor import Stage20LearnedExecutor, _OBS_DIM
from vlm_nsm.embodied.stage20_learned_executor_gate import LearnedExecutorGate, GateConfig
from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector
from vlm_nsm.skills.skill_types import SkillCommand

_PHASE_TO_ID = {"reach_pick": 0, "grasp": 1, "lift": 2, "reach_place": 3, "release": 4, "retreat": 5}
_HISTORY_LEN = 4
_FEATURE_DIM = 110
_ACTION_DIM = 7


class StackRLEnv:
    """RL environment wrapping Robosuite Stack task.

    Observation: normalized 110-dim feature sequence (4, 110)
    Action: 7-dim continuous action matching Stage 20 executor
    Reward: sparse success + dense progress shaping

    Uses scripted controller for execution with learned policy override.
    """

    def __init__(
        self,
        executor_checkpoint: str,
        dataset_dir: str,
        exec_config_path: str,
        goal_id: str = "red_on_green",
        seed: int = 0,
        device: str = "cpu",
        max_steps: int = 500,
        reward_config: dict[str, float] | None = None,
    ):
        self.goal_id = goal_id
        self.seed = seed
        self.max_steps = max_steps
        self.direction_id = 0 if goal_id == "red_on_green" else 1
        self.top_object = "cube1" if goal_id == "red_on_green" else "cube2"
        self.bottom_object = "cube2" if goal_id == "red_on_green" else "cube1"

        # Reward weights
        self._rw = reward_config or {
            "success": 20.0,
            "holding": 0.5,
            "lift_height": 1.0,
            "xy_align": 0.3,
            "action_smooth": 0.01,
            "timeout": -1.0,
            "unsafe": -10.0,
        }

        # Load executor for feature pipeline (warm start checkpoint)
        self.executor = Stage20LearnedExecutor(executor_checkpoint, dataset_dir, device=device)
        self.gate = LearnedExecutorGate(GateConfig())

        # Load execution config
        config = json.loads(Path(exec_config_path).read_text())
        self._exec_kwargs = {}
        for k in ["kp", "max_delta", "use_place_state_machine", "horizon_aware",
                   "place_retry", "grasp_close_steps", "grasp_max_retries",
                   "place_max_retries", "place_mode", "close_steps_min",
                   "close_steps_max", "close_stabilize_steps", "release_steps_min",
                   "release_steps_max", "post_release_stable_steps", "close_mode",
                   "close_steps_fast", "reach_threshold", "lift_height",
                   "release_open_steps", "terminate_after_safe_release"]:
            if k in config:
                self._exec_kwargs[k] = config[k]

        self._env: RobosuiteEnvWrapper | None = None
        self._step_count = 0
        self._prev_obs: np.ndarray | None = None
        self._prev_action: np.ndarray | None = None
        self._episode_reward_components: dict[str, float] = {}
        self._ever_lift = False
        self._ever_transport = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.seed = seed

        if self._env is not None:
            try:
                self._env.env.close()
            except Exception:
                pass

        self._env = RobosuiteEnvWrapper(
            env_name="Stack", robots="Panda",
            cube_names=["cube1", "cube2"], render_mode="none",
        )
        self._env.reset()
        self.executor.reset()
        self.gate.reset_counts()
        self._step_count = 0
        self._prev_obs = None
        self._prev_action = None
        self._episode_reward_components = {k: 0.0 for k in self._rw}
        self._ever_lift = False
        self._ever_transport = False

        obs_vec = self._get_obs()
        self._prev_obs = obs_vec

        # Build initial features
        features = self._build_features(obs_vec, phase_id=0)
        return features, self._get_info(False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step with the given action."""
        if self._env is None:
            raise RuntimeError("Environment not reset")

        action = np.asarray(action, dtype=np.float64).ravel()[:7]
        self._step_count += 1

        # Gate check
        gate_result = self.gate.check(
            action.astype(np.float32), phase_id=self._get_phase_id(),
            direction_id=self.direction_id, is_warm=self.executor.is_warm,
        )

        # Safety: fall back to safe action if gate fails
        executed_action = action.copy()
        if not gate_result.passed or not np.isfinite(action).all():
            executed_action = self._safe_fallback_action()

        # Execute
        self._env.step(executed_action)
        obs_vec = self._get_obs()

        # Compute reward
        reward, components = self._compute_reward(obs_vec, action, gate_result)
        for k, v in components.items():
            self._episode_reward_components[k] = self._episode_reward_components.get(k, 0.0) + v

        # Build features for next step
        features = self._build_features(obs_vec, phase_id=self._get_phase_id())

        # Termination
        task_success = self._check_success(obs_vec)
        truncated = self._step_count >= self.max_steps
        terminated = task_success

        info = self._get_info(task_success)
        info["gate_pass"] = gate_result.passed
        info["gate_reason"] = gate_result.reason
        info["reward_components"] = dict(components)

        self._prev_obs = obs_vec
        self._prev_action = action
        return features, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._env is None:
            return np.zeros(_OBS_DIM, dtype=np.float32)
        try:
            return build_robosuite_obs_vector(
                self._env.get_eef_pose(),
                self._env.get_gripper_state(),
                self._env.get_all_object_poses(),
                cubes=["cube1", "cube2"],
            )
        except Exception:
            return np.zeros(_OBS_DIM, dtype=np.float32)

    def _get_phase_id(self) -> int:
        """Simple phase inference from step count and object states."""
        obs = self._prev_obs
        if obs is None:
            return 0
        gripper_open = obs[3]
        has_object = obs[4] > 0.5
        top_z = obs[7] if self.top_object == "cube1" else obs[10]
        bottom_z = obs[10] if self.top_object == "cube1" else obs[7]
        table_z = 0.83

        # Heuristic phase detection
        if not has_object and gripper_open > 0.03:
            if top_z < table_z + 0.06:
                return 0  # reach_pick
            if self._ever_lift:
                return 3  # reach_place (after losing object)
            return 0
        if not has_object and gripper_open < 0.03:
            return 1  # grasp (attempting)
        if has_object and top_z < table_z + 0.05:
            return 1  # grasp (just picked up)
        if has_object and top_z < table_z + 0.10:
            self._ever_lift = True
            return 2  # lift
        if has_object and top_z >= table_z + 0.10:
            self._ever_lift = True
            return 3  # reach_place
        if not has_object and top_z > table_z + 0.06:
            return 4  # release (object stacked)
        return 5  # retreat

    def _build_features(self, obs_vec: np.ndarray, phase_id: int) -> np.ndarray:
        """Build normalized 110-dim features using the executor's pipeline."""
        result = self.executor.predict(
            obs_vec, phase_id=phase_id,
            top_object=self.top_object, bottom_object=self.bottom_object,
            return_features=True,
        )
        if "features" in result:
            return result["features"].astype(np.float32)
        # Fallback
        return np.zeros((_HISTORY_LEN, _FEATURE_DIM), dtype=np.float32)

    def _compute_reward(
        self, obs_vec: np.ndarray, action: np.ndarray, gate_result,
    ) -> tuple[float, dict[str, float]]:
        components: dict[str, float] = {}
        total = 0.0

        # Success
        if self._check_success(obs_vec):
            components["success"] = self._rw["success"]
            total += components["success"]
            return total, components

        # Holding reward
        has_object = obs_vec[4] > 0.5 if len(obs_vec) > 4 else False
        if has_object:
            components["holding"] = self._rw["holding"]
            total += components["holding"]

        # Lift reward: height of top object above table
        top_z = obs_vec[7] if self.top_object == "cube1" else obs_vec[10]
        table_z = 0.83
        lift_h = max(0.0, top_z - table_z - 0.04)  # above cube height
        if has_object and lift_h > 0:
            components["lift_height"] = self._rw["lift_height"] * min(lift_h, 0.05) / 0.05
            total += components["lift_height"]

        # XY alignment: distance from top to bottom
        top_x, top_y = obs_vec[5:7] if self.top_object == "cube1" else obs_vec[8:10]
        bottom_x, bottom_y = obs_vec[8:10] if self.top_object == "cube1" else obs_vec[5:7]
        xy_dist = np.sqrt((top_x - bottom_x) ** 2 + (top_y - bottom_y) ** 2)
        if has_object:
            components["xy_align"] = self._rw["xy_align"] * max(0.0, 1.0 - xy_dist / 0.05)
            total += components["xy_align"]

        # Action smoothness
        if self._prev_action is not None:
            smooth_penalty = self._rw["action_smooth"] * np.sum((action - self._prev_action) ** 2)
            components["action_smooth"] = -float(smooth_penalty)
            total -= smooth_penalty

        # Gate/safety
        if not gate_result.passed:
            components["unsafe"] = self._rw["unsafe"]
            total += components["unsafe"]

        return total, components

    def _check_success(self, obs_vec: np.ndarray) -> bool:
        """Check if top cube is stacked on bottom cube."""
        if len(obs_vec) < 11:
            return False
        top_idx = 5 if self.top_object == "cube1" else 8
        bottom_idx = 8 if self.top_object == "cube1" else 5
        top_z = obs_vec[top_idx + 2]
        bottom_z = obs_vec[bottom_idx + 2]
        xy_dist = np.sqrt(
            (obs_vec[top_idx] - obs_vec[bottom_idx]) ** 2 +
            (obs_vec[top_idx + 1] - obs_vec[bottom_idx + 1]) ** 2
        )
        return top_z > bottom_z + 0.02 and xy_dist < 0.05

    def _safe_fallback_action(self) -> np.ndarray:
        """Return a safe neutral action when gate fails."""
        return np.zeros(7, dtype=np.float64)

    def _get_info(self, task_success: bool) -> dict[str, Any]:
        obs = self._prev_obs
        return {
            "success": task_success,
            "step": self._step_count,
            "direction_id": self.direction_id,
            "ever_lift": self._ever_lift,
            "ever_transport": self._ever_transport,
            "phase_id": self._get_phase_id(),
            "has_object": bool(obs[4] > 0.5) if obs is not None and len(obs) > 4 else False,
            "top_z": float(obs[7]) if obs is not None and len(obs) > 7 else 0.0,
            "bottom_z": float(obs[10]) if obs is not None and len(obs) > 10 else 0.0,
            "reward_components": dict(self._episode_reward_components),
            "timeout": self._step_count >= self.max_steps,
            "unsafe_leak": False,
            "runtime_error": None,
        }

    def close(self):
        if self._env is not None:
            try:
                self._env.env.close()
            except Exception:
                pass

    @property
    def feature_dim(self) -> int:
        return _FEATURE_DIM

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def history_len(self) -> int:
        return _HISTORY_LEN
