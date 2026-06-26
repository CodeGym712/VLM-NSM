"""ACT-lite skill executor for closed-loop smoke evaluation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.learning.act_skill_policy import ACTSkillPolicy
from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector
from vlm_nsm.learning.skill_dataset import DEFAULT_OBJECT_TO_ID, DEFAULT_SKILL_TO_ID
from vlm_nsm.skills.skill_types import SkillCommand, SkillExecutionResult


class ACTSkillExecutor:
    """Execute one skill using a trained ACT-lite action chunk policy."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        max_steps: int = 1000,
        goal_check_interval: int = 10,
        action_clip: float = 1.0,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps
        self.goal_check_interval = goal_check_interval
        self.action_clip = action_clip

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        self.model = ACTSkillPolicy(**config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.normalization_stats = checkpoint.get("normalization_stats") or {}
        self.model_config = config

    def execute(
        self,
        skill_command: SkillCommand,
        env: Any,
        context: dict[str, Any] | None = None,
    ) -> SkillExecutionResult:
        if skill_command.skill_name != "pick_place":
            raise NotImplementedError(skill_command.skill_name)
        if not skill_command.target_object_id:
            raise ValueError("pick_place requires target_object_id")

        context = dict(context or {})
        cubes = context.get("cubes") or _infer_cubes(skill_command)
        max_steps = int(context.get("max_steps", self.max_steps))
        history = self._initial_history(env, cubes)
        goal_predicates = context.get("goal_predicates") or [
            ["on", skill_command.object_id, skill_command.target_object_id]
        ]

        steps = 0
        chunks = 0
        last_info: dict[str, Any] = {}
        failure_type: str | None = None

        while steps < max_steps:
            action_chunk = self._predict_action_chunk(history, skill_command)
            chunks += 1
            for action in action_chunk:
                env.step(np.clip(action, -self.action_clip, self.action_clip).astype(np.float32))
                steps += 1
                obs = self._obs(env, cubes)
                history.append(obs)

                if steps % self.goal_check_interval == 0 or steps >= max_steps:
                    satisfied, predicates = _physical_goal_satisfied(env, cubes, goal_predicates)
                    last_info = {"physical_predicates": sorted([list(p) for p in predicates])}
                    if satisfied:
                        return SkillExecutionResult(
                            success=True,
                            skill_command=skill_command,
                            steps=steps,
                            failure_type=None,
                            trajectory_path=context.get("trajectory_path"),
                            metrics={
                                "chunks": chunks,
                                "executor": type(self).__name__,
                                **last_info,
                            },
                        )
                if steps >= max_steps:
                    break

        failure_type = "horizon"
        return SkillExecutionResult(
            success=False,
            skill_command=skill_command,
            steps=steps,
            failure_type=failure_type,
            trajectory_path=context.get("trajectory_path"),
            metrics={
                "chunks": chunks,
                "executor": type(self).__name__,
                **last_info,
            },
        )

    def _initial_history(self, env: Any, cubes: list[str]) -> deque[np.ndarray]:
        obs = self._obs(env, cubes)
        history_len = int(self.model_config["obs_history_len"])
        return deque([obs.copy() for _ in range(history_len)], maxlen=history_len)

    def _obs(self, env: Any, cubes: list[str]) -> np.ndarray:
        return build_robosuite_obs_vector(
            env.get_eef_pose(),
            env.get_gripper_state(),
            env.get_all_object_poses(),
            cubes=cubes,
        ).astype(np.float32)

    def _predict_action_chunk(self, history: deque[np.ndarray], command: SkillCommand) -> np.ndarray:
        obs_history = np.stack(list(history), axis=0).astype(np.float32)
        obs_history = self._normalize(obs_history, "obs")
        goal_condition = np.array(
            [
                DEFAULT_SKILL_TO_ID.get(str(command.skill_name), 0),
                DEFAULT_OBJECT_TO_ID.get(command.object_id, 0),
                DEFAULT_OBJECT_TO_ID.get(command.target_object_id or "", 0),
            ],
            dtype=np.int64,
        )
        with torch.no_grad():
            pred = self.model(
                torch.from_numpy(obs_history).unsqueeze(0).to(self.device),
                torch.from_numpy(goal_condition).unsqueeze(0).to(self.device),
            ).squeeze(0).cpu().numpy()
        return self._denormalize(pred, "action")

    def _normalize(self, value: np.ndarray, prefix: str) -> np.ndarray:
        mean = np.asarray(self.normalization_stats[f"{prefix}_mean"], dtype=np.float32)
        std = np.asarray(self.normalization_stats[f"{prefix}_std"], dtype=np.float32)
        return (value - mean) / std

    def _denormalize(self, value: np.ndarray, prefix: str) -> np.ndarray:
        mean = np.asarray(self.normalization_stats[f"{prefix}_mean"], dtype=np.float32)
        std = np.asarray(self.normalization_stats[f"{prefix}_std"], dtype=np.float32)
        return value * std + mean


def _infer_cubes(command: SkillCommand) -> list[str]:
    cubes = [command.object_id]
    if command.target_object_id and command.target_object_id != "table":
        cubes.append(command.target_object_id)
    return list(dict.fromkeys(cubes))


def _physical_goal_satisfied(env: Any, cubes: list[str], goal_predicates: list[list[str]]) -> tuple[bool, set[tuple[str, ...]]]:
    object_poses = env.get_all_object_poses()
    targets = {}
    for cube in cubes:
        targets[cube] = object_poses[cube]
    if "table" in object_poses:
        targets["table"] = object_poses["table"]
    thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
    predicates = physical_state_to_predicates(object_poses, targets, thresholds)
    return goal_satisfied(predicates, goal_predicates, mode="symbolic"), predicates
