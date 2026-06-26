"""EmbodiedSkillExecutor v2 — end-to-end embodied execution pipeline.

Integrates:
  ActionGrounder → SkillStepDecomposer → SymbolConditionedObservationFilter
  → ControllerBackend → TerminationFailureMonitor
  → PhysicalPredicateExtractor → PredicateMonitor

Supports optional DemonstrationRecorder for data collection.
MockEnvWrapper compatible — no real Robosuite dependency.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from vlm_nsm.core.data_types import Action, Predicate, PredicateTuple, StepSignal
from vlm_nsm.core.action_grounder import ActionGrounder
from vlm_nsm.core.metrics import reality_gap
from vlm_nsm.core.physical_predicate_extractor import (
    extract_on_predicates,
    extract_clear_predicates,
    physical_state_to_predicates,
)
from vlm_nsm.core.predicate_monitor import goal_satisfied, compare_symbolic_physical
from vlm_nsm.embodied.skill_step_decomposer import SkillStepDecomposer
from vlm_nsm.embodied.symbol_conditioned_observation_filter import (
    SymbolConditionedObservationFilter,
)
from vlm_nsm.embodied.termination_failure_monitor import TerminationFailureMonitor
from vlm_nsm.embodied.controllers.base_controller import BaseController


def _make_obs_dict(cubes: list[str], object_poses: dict, gripper_state: dict) -> dict[str, Any]:
    """Build observation dict from MockEnvWrapper state."""
    obs = {}
    for cube in cubes:
        pose = object_poses.get(cube, {"x": 0.0, "y": 0.0, "z": 0.0})
        obs[f"{cube}_pos"] = [pose["x"], pose["y"], pose["z"]]
        obs[f"{cube}_quat"] = [0.0, 0.0, 0.0, 1.0]
    obs["table_pos"] = [0.0, 0.0, 0.0]
    obs["table_quat"] = [0.0, 0.0, 0.0, 1.0]
    gpos = gripper_state.get("pos", {"x": 0.0, "y": 0.0, "z": 0.3})
    obs["gripper_pos"] = [gpos["x"], gpos["y"], gpos["z"]]
    obs["gripper_quat"] = [0.0, 0.0, 0.0, 1.0]
    obs["gripper_has_object"] = gripper_state.get("has_object", False)
    obs["grasped_object"] = gripper_state.get("grasped_object", "")
    return obs


def _flatten_obs_dict(obs_dict: dict[str, Any], cubes: list[str]) -> np.ndarray:
    """Flatten observation dict into a fixed-dim numeric array (full obs)."""
    features: list[float] = []
    for cube in cubes:
        for key in (f"{cube}_pos", f"{cube}_quat"):
            val = obs_dict.get(key, [0.0] * (3 if "pos" in key else 4))
            features.extend(float(v) for v in val)
    for key in ("table_pos", "table_quat", "gripper_pos", "gripper_quat"):
        val = obs_dict.get(key, [0.0] * (3 if "pos" in key else 4))
        features.extend(float(v) for v in val)
    features.append(float(obs_dict.get("gripper_has_object", False)))
    return np.array(features, dtype=np.float32)


_STEP_NAMES = ["reach_pick", "grasp", "lift", "reach_place", "release", "retreat"]


class EmbodiedSkillExecutor:
    """End-to-end embodied executor with decomposed skill steps.

    Args:
        env_wrapper: MockEnvWrapper-compatible environment.
        cubes: List of cube names.
        controller: BaseController instance.
        action_grounder: Optional ActionGrounder (auto-created if None).
        decomposer: Optional SkillStepDecomposer (auto-created if None).
        obs_filter: Optional SymbolConditionedObservationFilter.
        term_monitor: Optional TerminationFailureMonitor.
        recorder: Optional DemonstrationRecorder.
        block_height: Block height for pose calculations.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        pose_noise_std: Std of Gaussian noise applied to poses.
        grasp_fail_prob: Probability of grasp failure (0.0 to 1.0).
        place_noise_std: Std of Gaussian noise applied to place positions.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        env_wrapper: Any,
        cubes: list[str] | None = None,
        controller: BaseController | None = None,
        action_grounder: ActionGrounder | None = None,
        decomposer: SkillStepDecomposer | None = None,
        obs_filter: SymbolConditionedObservationFilter | None = None,
        term_monitor: TerminationFailureMonitor | None = None,
        recorder: Any | None = None,
        block_height: float = 0.04,
        obs_dim: int = 16,
        action_dim: int = 4,
        pose_noise_std: float = 0.0,
        grasp_fail_prob: float = 0.0,
        place_noise_std: float = 0.0,
        seed: int = 0,
    ):
        self.env = env_wrapper
        self.cubes = cubes or ["cube1", "cube2", "cube3"]
        self.controller = controller
        self.action_grounder = action_grounder or ActionGrounder(env_wrapper)
        self.decomposer = decomposer or SkillStepDecomposer()
        self.obs_filter = obs_filter or SymbolConditionedObservationFilter(obs_dim=obs_dim)
        self.term_monitor = term_monitor or TerminationFailureMonitor()
        self.recorder = recorder
        self.block_height = block_height
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Noise config
        self._pose_noise_std = pose_noise_std
        self._grasp_fail_prob = grasp_fail_prob
        self._place_noise_std = place_noise_std
        self._rng = np.random.RandomState(seed)

        # Internal gripper state for mock simulation
        self._gripper_state = {
            "pos": {"x": 0.0, "y": 0.0, "z": 0.3},
            "has_object": False,
            "grasped_object": "",
            "near_object": "",
        }

    def noise_config(self) -> dict[str, Any]:
        return {
            "pose_noise_std": self._pose_noise_std,
            "grasp_fail_prob": self._grasp_fail_prob,
            "place_noise_std": self._place_noise_std,
        }

    def reset(self, init_state: Any = None) -> dict[str, Any]:
        """Reset environment and internal state."""
        self.env.reset()
        self._gripper_state = {
            "pos": {"x": 0.0, "y": 0.0, "z": 0.3},
            "has_object": False,
            "grasped_object": "",
            "near_object": "",
        }
        self._rng = np.random.RandomState(self._rng.randint(2**31))
        if self.controller:
            self.controller.reset()
        self.term_monitor.reset()
        return self._get_obs()

    def _get_obs(self) -> dict[str, Any]:
        """Build observation dict from current env state."""
        return _make_obs_dict(self.cubes, self.env.object_poses, self._gripper_state)

    def _get_full_obs_array(self) -> np.ndarray:
        """Build full (unfiltered) observation array from current env state."""
        obs_dict = self._get_obs()
        return _flatten_obs_dict(obs_dict, self.cubes)

    def _simulate_step(self, step_name: str, grounded: dict | None):
        """Simulate a skill step in mock env by updating internal state.

        Applies noise and failure injection based on config.

        Args:
            step_name: Current skill step name.
            grounded: Grounded action info from ActionGrounder (optional).
        """
        gs = self._gripper_state
        target_obj = grounded["object"] if grounded else self.cubes[0]
        target_place = grounded["target"] if grounded else self.cubes[-1]
        target_pose = self.env.get_target_pose(target_place)
        obj_pose = self.env.get_object_pose(target_obj)

        if step_name == "reach_pick":
            nx = obj_pose["x"]
            ny = obj_pose["y"]
            if self._pose_noise_std > 0:
                nx += self._rng.randn() * self._pose_noise_std
                ny += self._rng.randn() * self._pose_noise_std
            gs["pos"] = {"x": nx, "y": ny, "z": obj_pose["z"] + 0.08}
            gs["near_object"] = target_obj

        elif step_name == "grasp":
            grasp_ok = self._rng.rand() >= self._grasp_fail_prob
            if grasp_ok:
                gs["has_object"] = True
                gs["grasped_object"] = gs["near_object"]
            else:
                gs["has_object"] = False
                gs["grasped_object"] = gs["near_object"]

        elif step_name == "lift":
            obj_name = gs["grasped_object"]
            if obj_name and obj_name in self.env.object_poses:
                if gs["has_object"]:
                    pose = self.env.object_poses[obj_name]
                    pose["z"] = pose["z"] + 0.08
                    gs["pos"] = {"x": pose["x"], "y": pose["y"], "z": pose["z"] + 0.04}
            if not gs["has_object"]:
                gs["pos"]["z"] = gs["pos"].get("z", 0.3) + 0.04

        elif step_name == "reach_place":
            gs["pos"] = {
                "x": target_pose["x"],
                "y": target_pose["y"],
                "z": target_pose["z"] + 0.08,
            }
            gs["near_object"] = target_place

        elif step_name == "release":
            obj_name = gs["grasped_object"]
            if obj_name and obj_name in self.env.object_poses:
                z = self.block_height / 2.0 if target_place == "table" else target_pose["z"] + self.block_height
                px = target_pose["x"]
                py = target_pose["y"]
                if self._place_noise_std > 0:
                    px += self._rng.randn() * self._place_noise_std
                    py += self._rng.randn() * self._place_noise_std
                self.env.object_poses[obj_name] = {
                    "x": px,
                    "y": py,
                    "z": z,
                }
                self.env.supports[obj_name] = target_place
            gs["has_object"] = False
            gs["grasped_object"] = ""

        elif step_name == "retreat":
            gs["pos"]["z"] = gs["pos"].get("z", 0.3) + 0.05

    def execute_plan(
        self,
        actions: list[Action],
        goal_predicates: list[Predicate | PredicateTuple | list[str]] | None = None,
    ) -> dict[str, Any]:
        """Execute a symbolic plan through the embodied pipeline.

        Args:
            actions: List of symbolic Actions from the planner.
            goal_predicates: Goal predicates to check at the end.

        Returns:
            Dict with Experiment C metrics:
            - execution_success, goal_success, step_success_rate,
              skill_success_rate, episode_success, failure_stage_distribution,
              predicate_match_rate, num_steps, failure_type, etc.
        """
        self.reset(None)
        total_steps = 0
        successful_steps = 0
        successful_skills = 0
        total_skills = 0
        failure_stage_counts: dict[str, int] = {}
        episode_success = True
        final_failure_type: str | None = None

        for action_idx, action in enumerate(actions):
            # Ground the action
            if action.name == "move":
                grounded = self.action_grounder.ground_move(action)
            else:
                grounded = {"object": action.args[0] if action.args else "",
                            "source": action.args[1] if len(action.args) > 1 else "",
                            "target": action.args[2] if len(action.args) > 2 else "",
                            "object_pose": {"x": 0, "y": 0, "z": 0},
                            "target_pose": {"x": 0, "y": 0, "z": 0}}

            # Decompose into skill steps
            decomposed = self.decomposer.decompose(action)
            total_skills += 1
            skill_ok = True

            for step_idx, step_name in enumerate(decomposed.steps):
                # Get pre-step observation for filtering
                pre_obs = self._get_obs()

                # Build full obs array before simulation
                full_obs_arr = self._get_full_obs_array()

                # Filter observation
                filtered_obs = self.obs_filter.filter(
                    action.name, action.args, pre_obs
                )

                # Get controller action
                if self.controller:
                    ctrl_action = self.controller.act(step_name, filtered_obs)
                else:
                    ctrl_action = np.zeros(self.action_dim, dtype=np.float32)

                # Simulate step in mock env
                self._simulate_step(step_name, grounded)

                # Get post-step observation for termination check
                post_obs = self._get_obs()

                # Check termination
                signal = self.term_monitor.check(step_name, post_obs, step_idx)

                total_steps += 1

                # Record if recorder is active
                if self.recorder:
                    self.recorder.record_step(
                        obs=filtered_obs,
                        action=ctrl_action,
                        done=signal.done,
                        skill_id=step_idx,
                        step_id=step_idx,
                        success=signal.success,
                        filtered_obs=filtered_obs,
                        full_obs=full_obs_arr,
                    )

                if signal.failed:
                    failure_type = signal.failure_type or f"{step_name}_failed"
                    failure_stage_counts[failure_type] = failure_stage_counts.get(failure_type, 0) + 1
                    episode_success = False
                    final_failure_type = failure_type
                    skill_ok = False
                    break  # Stop this action
                else:
                    successful_steps += 1

            if skill_ok:
                successful_skills += 1

            if self.recorder and not skill_ok:
                break  # Stop episode on failure

        # Predicate extraction from final state
        object_poses = self.env.get_all_object_poses()
        targets = {}
        for cube in self.cubes:
            targets[cube] = object_poses[cube]
        targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        thresholds = {"xy_threshold": 0.02, "z_min": -0.001, "z_max": self.block_height * 1.5}

        physical_predicates = physical_state_to_predicates(object_poses, targets, thresholds)

        # Goal check
        goal_ok = False
        predicate_match_rate = 0.0
        comparison = None
        if goal_predicates:
            goal_ok = goal_satisfied(physical_predicates, goal_predicates, mode="symbolic")
            symbolic_set = {_to_pdt(p) for p in goal_predicates}
            physical_set = physical_predicates
            match_count = sum(1 for g in symbolic_set if g in physical_set)
            predicate_match_rate = match_count / max(len(symbolic_set), 1)
            comparison = compare_symbolic_physical(goal_predicates, physical_predicates)

        # Track B: symbolic_success is always 1 (planner produced a valid plan)
        symbolic_success = 1
        physical_success = 1 if (episode_success and goal_ok) else 0

        total_steps = max(total_steps, 1)
        total_skills = max(total_skills, 1)

        result = {
            "execution_success": episode_success,
            "goal_success": bool(goal_ok),
            "step_success_rate": successful_steps / total_steps,
            "skill_success_rate": successful_skills / total_skills,
            "episode_success": episode_success and bool(goal_ok),
            "failure_stage_distribution": failure_stage_counts,
            "predicate_match_rate": predicate_match_rate,
            "goal_satisfied": bool(goal_ok),
            "num_steps": total_steps,
            "failure_type": final_failure_type,
            "symbolic_success": symbolic_success,
            "physical_success": physical_success,
            "symbolic_success_count": symbolic_success,
            "physical_success_count": physical_success,
            "reality_gap": reality_gap(symbolic_success, physical_success),
            "physical_predicates": sorted(str(p) for p in physical_predicates),
        }

        if comparison:
            result["mismatch_rate"] = comparison["mismatch_rate"]
            result["missing_in_physical"] = comparison["missing_in_physical"]
            result["extra_in_physical"] = comparison["extra_in_physical"]

        return result


def _to_pdt(pred):
    if isinstance(pred, Predicate):
        return pred.as_tuple()
    return tuple(pred)
