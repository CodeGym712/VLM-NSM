"""Run robosuite embodied stacking evaluation.

Executes the full embodied pipeline with RobosuiteEnvWrapper
and proportional control, reporting Experiment C metrics.

Uses robosuite 1.5.2 normalized action space ([-1, 1] for OSC_POSE).

Usage:
    python -m vlm_nsm.experiments.run_embodied_stacking_robosuite \\
        --num_episodes 5 --render

    # Offscreen (no rendering):
    python -m vlm_nsm.experiments.run_embodied_stacking_robosuite \\
        --num_episodes 10

Stage 7A-10 additions:
  - Phase-level trace recorder (step count per phase)
  - Horizon failure subtypes (horizon_before_grasp, horizon_holding_object, etc.)
  - Goal-aware termination: skip_retreat_if_goal_satisfied, terminate_after_safe_release
  - Config propagation to grasp/place state machines
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vlm_nsm.core.data_types import Predicate
from vlm_nsm.core.metrics import reality_gap
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import goal_satisfied
from vlm_nsm.core.utils import PROJECT_ROOT, ensure_dir

_STEP_LIMIT = 500
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0
_REACH_THRESHOLD = 0.02
_LIFT_HEIGHT = 0.05
_GRASP_CLOSE_STEPS = 50
_RELEASE_OPEN_STEPS = 50
_RELEASE_STEPS_MIN = 10
_RELEASE_STEPS_MAX = 50
_POST_RELEASE_STABLE_STEPS = 5

# OSC_POSE max position delta per step (for action normalization)
_OSC_POSE_MAX_DELTA = 0.05

# Predicate thresholds for robosuite (calibrated via calibrate_predicate_thresholds.py)
_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}

# Phase-to-skill-id mapping for DemonstrationRecorder
_PHASE_TO_SKILL_ID = {
    "reach_pick": 0, "grasp": 1, "lift": 2,
    "reach_place": 3, "release": 4, "retreat": 5,
}

# Module-level phase tracker for HDF5 recording fallback
_CURRENT_PHASE = {"name": "", "id": 0}

# Minimum remaining steps needed to attempt a skill phase
_MIN_BUDGET_REACH = 10
_MIN_BUDGET_GRASP = 80
_MIN_BUDGET_LIFT = 10
_MIN_BUDGET_REACH_PLACE = 10
_MIN_BUDGET_RELEASE = _RELEASE_OPEN_STEPS + 5
_MIN_BUDGET_RETREAT = 10


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _norm_action(dx: float, dy: float, dz: float, gripper: float) -> np.ndarray:
    """Convert physical deltas to 7-dim normalized action [-1, 1]."""
    return np.array([
        np.clip(dx / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dy / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dz / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        0.0, 0.0, 0.0,
        gripper,
    ], dtype=np.float32)


def _pid_reach_loop(env, pid_ctrl, target: dict, step_name: str,
                    gripper_cmd: float, step_limit: int,
                    reach_threshold: float,
                    id_ensemble=None, phase_id: int = 0,
                    vel_history: list | None = None) -> tuple[bool, int]:
    """Execute a reach-like phase using PID controller.

    Args:
        env: Environment wrapper (with get_eef_pose and step).
        pid_ctrl: OperationalSpacePID instance.
        target: Target position dict {"x","y","z"}.
        step_name: Phase name for PID gain scheduling.
        gripper_cmd: Gripper command (-1.0 open, 1.0 close).
        step_limit: Maximum steps before giving up.
        reach_threshold: Distance threshold for convergence.
        id_ensemble: Optional IDModelEnsemble for velocity feedforward.
        phase_id: Integer phase ID for ID model input.
        vel_history: Optional mutable list for velocity history (3 frames).

    Returns:
        (success, steps_taken). If success=False, the caller should
        handle failure reporting.
    """
    pid_ctrl.set_phase(step_name)
    pid_ctrl.set_target(pos=target, vel=None,
                        close_gripper=(gripper_cmd > 0))
    steps = 0
    target_pos = np.array([target["x"], target["y"], target["z"]], dtype=np.float32)
    _prev_eef = None
    _vel_buf = [0.0] * 9
    for _ in range(step_limit):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            return False, steps

        eef_arr = np.array([eef["x"], eef["y"], eef["z"]], dtype=np.float32)

        # ID model feedforward: predict target velocity, update PID set_target
        learned_vel = None
        if id_ensemble is not None and id_ensemble.models:
            try:
                x = np.zeros(28, dtype=np.float32)
                x[0:3] = eef_arr
                x[3] = 1.0 if gripper_cmd > 0 else -1.0
                x[4:13] = _vel_buf
                x[13:25] = [0.0] * 12  # action_history (zero for now)
                x[25:28] = target_pos
                pred = id_ensemble.predict(x, confidence_threshold=0.3)
                if pred.confidence >= 0.3:
                    learned_vel = pred.velocity
            except Exception:
                pass

        if learned_vel is not None and np.all(np.isfinite(learned_vel)):
            pid_ctrl._target_vel = learned_vel  # directly update feedforward

        pid_ctrl.set_current(eef)
        action = pid_ctrl.act(step_name, None)
        action[6] = gripper_cmd

        if _prev_eef is not None:
            vel = (eef_arr - _prev_eef) / 0.02
            _vel_buf = (_vel_buf[3:] + list(vel))[-9:]
        _prev_eef = eef_arr.copy()

        try:
            env.step(action)
        except ValueError:
            return False, steps
        steps += 1
        if _dist(eef, target) < reach_threshold:
            return True, steps
    return False, steps


def _build_obs_for_monitor(
    env, cubes: list[str], grasped_object: str = ""
) -> dict:
    """Build observation dict compatible with TerminationFailureMonitor."""
    poses = env.get_all_object_poses()
    gripper = env.get_gripper_state()

    obs = {}
    for cube in cubes:
        pose = poses.get(cube, {"x": 0.0, "y": 0.0, "z": 0.0})
        obs[f"{cube}_pos"] = [pose["x"], pose["y"], pose["z"]]
        obs[f"{cube}_quat"] = [0.0, 0.0, 0.0, 1.0]

    obs["table_pos"] = [0.0, 0.0, 0.0]
    obs["table_quat"] = [0.0, 0.0, 0.0, 1.0]

    gpos = gripper["pos"]
    obs["gripper_pos"] = [gpos["x"], gpos["y"], gpos["z"]]
    obs["gripper_quat"] = [0.0, 0.0, 0.0, 1.0]
    obs["gripper_has_object"] = gripper.get("has_object", False)
    obs["grasped_object"] = grasped_object

    return obs


def build_stacking_goal(target_stack: list[str]) -> list[Predicate]:
    return [Predicate("on", [obj, support]) for support, obj in zip(target_stack, target_stack[1:])]


def _remaining_budget(total_steps: int, horizon: int) -> int:
    return max(0, horizon - total_steps)


def _check_goal_satisfied(env, cubes: list[str], goal_predicates: list[Predicate] | None) -> bool:
    """Check if goal predicates are satisfied in current env state."""
    if not goal_predicates:
        return False
    object_poses = env.get_all_object_poses()
    targets = dict(object_poses)
    targets["table"] = env.get_object_pose("table")
    physical_predicates = physical_state_to_predicates(object_poses, targets, _PREDICATE_THRESHOLDS)
    return bool(goal_satisfied(physical_predicates, goal_predicates, mode="symbolic"))


class _RobosuiteRecordingWrapper:
    """Thin wrapper around RobosuiteEnvWrapper that records every ``step()``.

    Intercepts ``env.step(action)`` to capture observation and action
    at each timestep. All other attribute access is delegated to the
    underlying environment via ``__getattr__``, making this transparent
    to callers (including grasp/place state machines).
    """

    def __init__(self, env, recorder, cubes, policy_override=None, override_phases=None,
                 bc_blend_alpha: float = 1.0,
                 shadow_executor=None, shadow_gate=None,  # Stage 20-V
                 shadow_direction_id: int = -1, shadow_top_object: str = "cube1",
                 shadow_bottom_object: str = "cube2",
                 execute_learned_when_safe: bool = False,  # Stage 20-W
                 record_full_features: bool = False):  # Stage 20-Y
        self._env = env
        self._recorder = recorder
        self._cubes = cubes
        self._phase = ""
        self._phase_step = 0
        self._policy = policy_override
        self._override_ids = override_phases or set()
        self._blend_alpha = bc_blend_alpha
        self.bc_steps = 0
        self.scripted_steps = 0
        # Stage 20-V: shadow executor for per-step learned prediction
        self._shadow_executor = shadow_executor
        self._shadow_gate = shadow_gate
        self._shadow_direction_id = shadow_direction_id
        self._shadow_top_object = shadow_top_object
        self._shadow_bottom_object = shadow_bottom_object
        # Stage 20-W: gated execution mode
        self._execute_learned = execute_learned_when_safe
        # Stage 20-Y: full feature recording
        self._record_full_features = record_full_features
        self.shadow_diagnostics: list[dict] = []
        self.learned_predictions = 0
        self.learned_executed_steps = 0
        self.fallback_steps = 0
        # Stage 20-Y: per-step feature buffers
        self._feature_records: list[dict[str, np.ndarray | Any]] = []

    def set_phase(self, phase: str):
        """Set the current skill phase name for step recording."""
        self._phase = phase
        self._phase_step = 0

    def step(self, action):
        from vlm_nsm.learning.robosuite_schema import build_robosuite_obs_vector
        # Build observation from current state
        obs = build_robosuite_obs_vector(
            self._env.get_eef_pose(),
            self._env.get_gripper_state(),
            self._env.get_all_object_poses(),
            cubes=self._cubes,
        )
        scripted_action_arr = np.asarray(action, dtype=np.float32)
        action_arr = scripted_action_arr.copy()

        # Stage 7B-2R: apply policy override BEFORE recording
        _pid = _PHASE_TO_SKILL_ID.get(self._phase, -1)
        if self._policy is not None and _pid in self._override_ids:
            bc_action = self._policy(obs, _pid)
            # Stage 7B-3: blend BC with scripted action if alpha < 1.0
            if self._blend_alpha < 1.0:
                action_arr = (1.0 - self._blend_alpha) * action_arr + self._blend_alpha * bc_action
            else:
                action_arr = bc_action
            self.bc_steps += 1
        else:
            self.scripted_steps += 1

        # Stage 20-W: gated learned execution (before env.step)
        executed_source = "scripted"
        learned_action = None
        gate_passed = False
        gate_reason = "no_learned_executor"
        learned_finite = True
        is_warm = False

        if self._shadow_executor is not None:
            learned_result = self._shadow_executor.predict(
                obs, phase_id=_pid,
                top_object=self._shadow_top_object,
                bottom_object=self._shadow_bottom_object,
                return_features=self._record_full_features,  # Stage 20-Y
            )
            learned_action = learned_result["action"]
            learned_finite = learned_result["finite"]
            is_warm = self._shadow_executor.is_warm
            self.learned_predictions += 1

            gate_result = None
            if self._shadow_gate is not None:
                gate_result = self._shadow_gate.check(
                    learned_action, phase_id=_pid,
                    direction_id=self._shadow_direction_id,
                    is_warm=is_warm,
                )
                gate_passed = gate_result.passed
                gate_reason = gate_result.reason

            # Stage 20-W: execute learned action if safe
            if self._execute_learned and gate_passed and learned_finite and is_warm:
                action_arr = learned_action.copy()
                executed_source = "learned"
                self.learned_executed_steps += 1
            elif self._execute_learned:
                executed_source = "fallback"
                self.fallback_steps += 1

        # Execute the step (scripted or learned)
        # Phase 1.5: expose current phase to inner env for VLA override callbacks
        if hasattr(self._env, '_vla_phase'):
            self._env._vla_phase = self._phase
        result = self._env.step(action_arr)

        # Record if recorder is available
        if self._recorder is not None:
            self._recorder.record_step(
                obs=obs, filtered_obs=obs,
                action=action_arr, done=False,
                skill_id=_pid,
                step_id=self._phase_step, success=True,
                phase_id=_pid,
            )

        # Stage 20-V/W: shadow/gated diagnostics
        if self._shadow_executor is not None and learned_action is not None:
            diag = {
                "step": self._phase_step,
                "phase_name": self._phase,
                "phase_id": _pid,
                "scripted_action": scripted_action_arr.tolist(),
                "learned_action": learned_action.tolist(),
                "executed_action": action_arr.tolist(),
                "executed_source": executed_source,
                "action_error": float(np.linalg.norm(learned_action - scripted_action_arr)),
                "dim6_error": float(abs(learned_action[6] - scripted_action_arr[6])),
                "learned_action_norm": learned_result["action_norm"],
                "action_finite": learned_finite,
                "history_warm": is_warm,
                "gate_pass": gate_passed,
                "gate_reason": gate_reason,
                "fallback_reason": gate_reason if executed_source == "fallback" else "",
            }
            if "holding_logit" in learned_result:
                diag["holding_logit"] = learned_result["holding_logit"]
            if len(obs) > 4:
                diag["obs4_holding_signal"] = float(obs[4])
            # Stage 20-Y: record full features for DAgger
            if self._record_full_features and "features" in learned_result:
                diag["features"] = learned_result["features"]  # (H, 110) normalized
                diag["obs_padded"] = learned_result["obs_padded"]  # (H, 11)
                diag["phase_padded"] = learned_result["phase_padded"]  # (H,)
                diag["history_warm"] = learned_result["history_warm"]
                diag["n_avail"] = learned_result["n_avail"]
            self.shadow_diagnostics.append(diag)

        self._phase_step += 1
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


def _run_robosuite_episode(
    env, actions: list, cubes: list[str],
    goal_predicates: list[Predicate] | None = None,
    horizon: int = 1000,
    early_stop_on_goal: bool = True,
    use_place_state_machine: bool = True,
    horizon_aware: bool = True,
    place_retry: bool = True,
    kp: float = 0.05,
    max_delta: float = 0.02,
    reach_threshold: float | None = None,
    lift_height: float | None = None,
    release_open_steps: int | None = None,
    # Stage 7A-10: goal-aware termination
    skip_retreat_if_goal_satisfied: bool = False,
    terminate_after_safe_release: bool = False,
    # Stage 7A-10: config propagation for state machines
    grasp_close_steps: int | None = None,
    grasp_max_retries: int | None = None,
    place_max_retries: int | None = None,
    place_mode: str = "simple",
    # Stage 7A-11: event-driven grasp + lift early exit
    holding_check_enabled: bool = False,
    close_steps_min: int | None = None,
    close_steps_max: int | None = None,
    close_stabilize_steps: int | None = None,
    lift_early_exit_enabled: bool = False,
    # Stage 7A-11: release phase enhanced
    release_steps_min: int | None = None,
    release_steps_max: int | None = None,
    post_release_stable_steps: int | None = None,
    # Stage 7B-1: optional demonstration recorder
    recorder=None,
    # Stage 7B-2R: policy override for hybrid BC eval (applied before recording)
    policy_override=None,
    override_phases: set[int] | None = None,
    # Stage 7A-12: config-specified close mode (starting point for horizon-aware)
    close_mode: str = "fixed",
    close_steps_fast: int | None = None,
    fallback_extend_close_if_not_holding: bool = False,
    # Stage 7B-3: BC blend alpha (1.0 = pure BC, 0.0 = pure scripted)
    bc_blend_alpha: float = 1.0,
    # Stage 19-D5: runtime-only scripted robustness profile offsets.
    grasp_z_offset: float = 0.0,
    place_z_offset: float = 0.0,
    release_z_offset: float = 0.0,
    stabilization_steps: int = 0,
    # Stage 20-V: per-step shadow executor
    shadow_executor=None,
    shadow_gate=None,
    shadow_direction_id: int = -1,
    shadow_top_object: str = "cube1",
    shadow_bottom_object: str = "cube2",
    # Stage 20-W: gated learned execution
    execute_learned_when_safe: bool = False,
    # Stage 20-Y: full feature recording for DAgger
    record_full_features: bool = False,
    # Phase 1A: preplace fast mode (aggressive P-control ~ IK)
    preplace_fast_mode: bool = False,
    # Phase 2.0: skip env reset (multi-step persistent scene)
    skip_reset: bool = False,
    # Phase 1B: high-place path (lift → XY align → slow descend)
    high_place_enabled: bool = False,
    # Phase 3.0: inverse dynamics PID controller
    use_pid_controller: bool = True,
    # Phase 3.1: ID model ensemble for velocity feedforward
    id_ensemble_paths: list | None = None,
) -> dict:
    """Run a single robosuite episode following a symbolic plan.

    Args:
        env: Robosuite environment wrapper.
        actions: List of symbolic actions from planner.
        cubes: List of canonical cube names.
        goal_predicates: Goal predicates to check.
        horizon: Maximum steps before episode termination.
        early_stop_on_goal: If True, stop executing when goal is satisfied.
        use_place_state_machine: If True, use place_state_machine for place steps.
        horizon_aware: If True, check remaining budget before each phase.
        place_retry: If True, allow retry in place_state_machine.
        kp: Proportional gain for position control.
        max_delta: Max position delta per step.
        reach_threshold: Distance threshold for reach convergence.
        lift_height: Height to lift object above grasp position.
        release_open_steps: Number of steps for gripper release.
        skip_retreat_if_goal_satisfied: Skip retreat phase if goal already satisfied.
        terminate_after_safe_release: Terminate after release, skip retreat.
        grasp_close_steps: Override for grasp close steps.
        grasp_max_retries: Override for grasp max retries.
        place_max_retries: Override for place max retries.
        place_mode: Place state machine mode (simple/fast/safe).

    Returns:
        Dict with execution metrics including strict/semantic success.
    """
    from vlm_nsm.embodied.skill_step_decomposer import SkillStepDecomposer
    from vlm_nsm.embodied.termination_failure_monitor import TerminationFailureMonitor

    decomposer = SkillStepDecomposer()
    term_monitor = TerminationFailureMonitor()
    term_monitor.reset()

    # Phase 3.0: PID controller initialization
    pid_ctrl = None
    id_ensemble = None
    if use_pid_controller:
        from vlm_nsm.embodied.controllers.operational_space_pid import (
            OperationalSpacePID,
            GainSchedule,
        )
        pid_ctrl = OperationalSpacePID(gain_schedule=GainSchedule())

        # Phase 3.1: ID model ensemble for velocity feedforward
        if id_ensemble_paths:
            from vlm_nsm.embodied.controllers.learned_id_model import IDModelEnsemble
            id_ensemble = IDModelEnsemble(model_paths=id_ensemble_paths)

    # Phase 2.0: skip_reset for multi-step execution on same scene
    if not skip_reset:
        env.reset()

    # Stage 7B-1: wrap env for recording if recorder is provided
    # Stage 7B-2R: also wrap when policy_override is active (with or without recorder)
    # Stage 20-V: also wrap when shadow_executor is provided (for per-step shadow)
    if recorder is not None or policy_override is not None or shadow_executor is not None:
        env = _RobosuiteRecordingWrapper(
            env, recorder, cubes,
            policy_override=policy_override,
            override_phases=override_phases,
            bc_blend_alpha=bc_blend_alpha,
            shadow_executor=shadow_executor,
            shadow_gate=shadow_gate,
            shadow_direction_id=shadow_direction_id,
            shadow_top_object=shadow_top_object,
            shadow_bottom_object=shadow_bottom_object,
            execute_learned_when_safe=execute_learned_when_safe,
            record_full_features=record_full_features,
        )
        _rec_set_phase = env.set_phase  # from _RobosuiteRecordingWrapper
    else:
        # No recording wrapper — set _phase attr directly on env for collection scripts
        def _rec_set_phase(phase_name: str):
            env._phase = phase_name
            _CURRENT_PHASE["name"] = phase_name
            _CURRENT_PHASE["id"] = _PHASE_TO_SKILL_ID.get(phase_name, 0)
        if not hasattr(env, '_phase'):
            env._phase = ""

    # Resolve optional params to module defaults
    _rt = _REACH_THRESHOLD if reach_threshold is None else reach_threshold
    _lh = _LIFT_HEIGHT if lift_height is None else lift_height
    _ros = _RELEASE_OPEN_STEPS if release_open_steps is None else release_open_steps

    total_steps = 0
    successful_steps = 0
    successful_skills = 0
    total_skills = 0
    failure_stage_counts: dict[str, int] = {}
    episode_success = True
    final_failure_type: str | None = None
    grasped_object = ""
    goal_was_achieved = False
    goal_achieved_at_step: int | None = None
    early_stopped = False
    place_done = False  # track if place_state_machine handled release+retreat

    # Stage 7A-10: phase-level trace recorder (step count per phase)
    phase_trace: dict[str, int] = {"reach_pick": 0, "grasp": 0, "lift": 0,
                                    "reach_place": 0, "release": 0, "retreat": 0}
    _last_phase_before_horizon: str | None = None  # for horizon failure subtype

    # Stage 7A-11: termination stats
    _skip_retreat_triggered = False
    _terminate_release_triggered = False

    for action_idx, action in enumerate(actions):
        target_obj = action.args[0] if action.args else cubes[0]
        target_place = action.args[2] if len(action.args) > 2 else cubes[-1]

        # Early stop: check if goal already satisfied before executing this skill
        if early_stop_on_goal and goal_was_achieved:
            early_stopped = True
            break

        decomposed = decomposer.decompose(action)
        total_skills += 1
        skill_ok = True

        for step_idx, step_name in enumerate(decomposed.steps):
            # Skip release/retreat if place_state_machine handled them
            if place_done and step_name in ("release", "retreat"):
                continue

            # Check remaining budget before each step phase
            remaining = _remaining_budget(total_steps, horizon)
            if horizon_aware:
                min_budget_map = {
                    "reach_pick": _MIN_BUDGET_REACH,
                    "grasp": _MIN_BUDGET_GRASP,
                    "lift": _MIN_BUDGET_LIFT,
                    "reach_place": _MIN_BUDGET_REACH_PLACE,
                    "release": _MIN_BUDGET_RELEASE,
                    "retreat": _MIN_BUDGET_RETREAT,
                }
                min_needed = min_budget_map.get(step_name, 10)
                if remaining < min_needed:
                    # Stage 7A-10: horizon failure with subtype context
                    _last_phase_before_horizon = step_name
                    if step_name == "grasp":
                        failure_type = "horizon_before_grasp"
                    elif step_name == "reach_place":
                        failure_type = "horizon_preplace"
                    elif step_name == "release":
                        failure_type = "horizon_holding_object"
                    elif goal_was_achieved:
                        failure_type = "horizon_after_goal"
                    else:
                        failure_type = f"horizon_{step_name}"
                    failure_stage_counts[failure_type] = failure_stage_counts.get(failure_type, 0) + 1
                    episode_success = False
                    final_failure_type = failure_type
                    skill_ok = False
                    break

            if step_name == "reach_pick":
                _rec_set_phase("reach_pick")
                obj_pose = env.get_object_pose(target_obj)
                target = {"x": obj_pose["x"], "y": obj_pose["y"], "z": obj_pose["z"] + 0.10 + grasp_z_offset}
                grasped_object = target_obj

                if use_pid_controller and pid_ctrl is not None:
                    pid_ok, pid_steps = _pid_reach_loop(
                        env, pid_ctrl, target, "reach_pick",
                        _GRIPPER_OPEN, min(_STEP_LIMIT, remaining), _rt,
                        id_ensemble=id_ensemble, phase_id=0,
                    )
                    total_steps += pid_steps
                    phase_trace["reach_pick"] += pid_steps
                    if pid_ok:
                        successful_steps += 1
                    else:
                        skill_ok = False; episode_success = False
                        final_failure_type = f"{step_name}_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break
                else:
                    for _ in range(min(_STEP_LIMIT, remaining)):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        dx = np.clip(kp * (target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        total_steps += 1
                        phase_trace["reach_pick"] += 1
                        if _dist(eef, target) < _rt:
                            successful_steps += 1
                            break

            elif step_name == "grasp":
                _rec_set_phase("grasp")
                from vlm_nsm.embodied.grasp_state_machine import execute_grasp as _exec_grasp, GraspConfig

                # Stage 7A-12: horizon-aware grasp config with config-specified base close_mode
                _remaining = _remaining_budget(total_steps, horizon)
                _ha_grasp_close_steps = (grasp_close_steps if grasp_close_steps is not None
                                          else _GRASP_CLOSE_STEPS)
                _ha_retries = (grasp_max_retries if grasp_max_retries is not None else 3)
                _ha_skip_approach = True  # default for stacking
                _ha_close_mode = close_mode  # start from config-specified mode

                if horizon_aware:
                    if _remaining < 80:
                        # Too little remaining — will be caught by budget check
                        pass
                    elif _remaining < 120:
                        # Very tight: aggressive close, single retry max
                        _ha_close_mode = "aggressive"
                        _ha_retries = min(_ha_retries, 1)
                        _ha_grasp_close_steps = min(_ha_grasp_close_steps, 10)
                    elif _remaining < 180:
                        # Tight: adaptive close, reduced retries
                        if _ha_close_mode == "fixed":
                            _ha_close_mode = "adaptive"
                        _ha_retries = min(_ha_retries, 2)
                    elif _remaining < 250:
                        # Moderate: adaptive close
                        if _ha_close_mode == "fixed":
                            _ha_close_mode = "adaptive"

                _grasp_cfg = GraspConfig(kp=kp, max_delta=max_delta,
                                          descend_offset=max(0.0, 0.005 + grasp_z_offset),
                                          close_steps=_ha_grasp_close_steps,
                                          max_retries=_ha_retries,
                                          holding_check_enabled=holding_check_enabled,
                                          lift_early_exit_enabled=lift_early_exit_enabled,
                                          close_steps_min=close_steps_min if close_steps_min is not None else 15,
                                          close_steps_max=close_steps_max if close_steps_max is not None else _ha_grasp_close_steps,
                                          close_stabilize_steps=close_stabilize_steps if close_stabilize_steps is not None else 5,
                                          close_mode=_ha_close_mode,
                                          close_steps_fast=close_steps_fast if close_steps_fast is not None else 15,
                                          fallback_extend_close_if_not_holding=fallback_extend_close_if_not_holding)
                _grasp_result = _exec_grasp(env, target_obj, config=_grasp_cfg,
                                             skip_approach=True, skip_lift=True)
                _grasp_steps = _grasp_result.get("total_step_count", 0)
                total_steps += _grasp_steps
                phase_trace["grasp"] += _grasp_steps
                if _grasp_result["success"]:
                    successful_steps += 1
                else:
                    skill_ok = False
                    episode_success = False
                    final_failure_type = _grasp_result.get("failure_type") or "grasp_failed"
                    failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                    break

            elif step_name == "lift":
                _rec_set_phase("lift")
                obj_pose = env.get_object_pose(target_obj)
                lift_target = {
                    "x": obj_pose["x"], "y": obj_pose["y"],
                    "z": obj_pose["z"] + _LIFT_HEIGHT + 0.10,
                }

                if use_pid_controller and pid_ctrl is not None:
                    pid_ok, pid_steps = _pid_reach_loop(
                        env, pid_ctrl, lift_target, "lift",
                        _GRIPPER_CLOSE, min(_STEP_LIMIT, remaining), _rt,
                        id_ensemble=id_ensemble, phase_id=2,
                    )
                    total_steps += pid_steps
                    phase_trace["lift"] += pid_steps
                    if pid_ok:
                        successful_steps += 1
                    else:
                        skill_ok = False; episode_success = False
                        final_failure_type = f"{step_name}_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break
                else:
                    for _ in range(min(_STEP_LIMIT, remaining)):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        dx = np.clip(kp * (lift_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (lift_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (lift_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        total_steps += 1
                        phase_trace["lift"] += 1
                        if _dist(eef, lift_target) < _rt:
                            successful_steps += 1
                            break

            elif step_name == "reach_place" and use_place_state_machine:
                _rec_set_phase("reach_place")

                # Phase 2.0: holding check — verify object is still in gripper
                _holding_ok = False  # Fix H5: default to NOT holding for safety
                _holding_check_attempts = 0
                while _holding_check_attempts < 2:
                    try:
                        _obj_pose = env.get_object_pose(target_obj)
                        _eef_pose = env.get_eef_pose()
                        _obj_z = _obj_pose.get("z", 0.0)
                        _obj_xy = np.array([_obj_pose.get("x", 0), _obj_pose.get("y", 0)])
                        _eef_xy = np.array([_eef_pose.get("x", 0), _eef_pose.get("y", 0)])
                        _table_z = 0.80
                        _z_lifted = _obj_z > _table_z + 0.03
                        _near_eef = float(np.linalg.norm(_obj_xy - _eef_xy)) < 0.05
                        _holding_ok = _z_lifted and _near_eef
                        break  # query succeeded
                    except (KeyError, AttributeError, TypeError):
                        _holding_check_attempts += 1
                        if _holding_check_attempts < 2:
                            import time as _time
                            _time.sleep(0.05)  # brief pause before retry

                if not _holding_ok and _remaining_budget(total_steps, horizon) > 150:
                    # Object dropped during transport — attempt regrasp
                    from vlm_nsm.embodied.grasp_state_machine import execute_grasp as _exec_grasp, GraspConfig
                    _grasp_cfg = GraspConfig(kp=kp, max_delta=max_delta, reach_threshold=_rt,
                                             close_steps=(grasp_close_steps or 70),
                                             max_retries=2,  # Phase 2.0: allow retry
                                             close_mode=close_mode or "adaptive")
                    _grasp_result = _exec_grasp(env, target_obj, config=_grasp_cfg,
                                                skip_approach=False, skip_lift=False)
                    total_steps += _grasp_result.get("total_step_count", 0)
                    if _grasp_result.get("success", False):
                        # Regrasp succeeded — continue to place
                        _holding_ok = True
                    else:
                        skill_ok = False; episode_success = False
                        final_failure_type = "transport_drop_regrasp_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break

                # Use place_state_machine for composite place (includes release+retreat)
                from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
                _place_max_retries = place_max_retries if place_max_retries is not None else (3 if place_retry else 1)
                _place_kwargs = {}
                if place_z_offset:
                    _place_kwargs["preplace_height"] = 0.15 + place_z_offset
                if release_z_offset:
                    _place_kwargs["descend_offset"] = max(0.0, 0.02 + release_z_offset)
                if stabilization_steps:
                    _place_kwargs["hold_steps"] = 10 + max(0, stabilization_steps)
                if preplace_fast_mode:
                    _place_kwargs["preplace_fast_mode"] = True
                if high_place_enabled:
                    _place_kwargs["high_place_enabled"] = True
                    _place_kwargs["high_place_z_margin"] = 0.06
                _place_cfg = PlaceConfig(kp=kp, max_delta=max_delta,
                                          max_retries=_place_max_retries,
                                          reach_threshold=_rt, step_limit=min(200, max(remaining, 50)),
                                          mode=place_mode,
                                          **_place_kwargs)
                _place_result = execute_place(env, target_obj, target_place,
                                              config=_place_cfg, skip_preplace=False,
                                              phase_callback=_rec_set_phase)
                _place_steps = _place_result.get("step_count", 0)
                total_steps += _place_steps
                phase_trace["reach_place"] += _place_steps
                if _place_result.get("success", False):
                    successful_steps += 1
                    grasped_object = ""
                else:
                    # Phase 2.0 recovery: attempt regrasp on place failure
                    _ft = _place_result.get("failure_type") or "place_failed"
                    _can_recover = _remaining_budget(total_steps, horizon) > 250
                    if _can_recover:
                        # Try regrasp + replan: re-pick the object, then re-place
                        from vlm_nsm.embodied.grasp_state_machine import execute_grasp as _exec_grasp, GraspConfig
                        _regrasp_cfg = GraspConfig(kp=kp, max_delta=max_delta, reach_threshold=_rt,
                                                   close_steps=(grasp_close_steps or 70),
                                                   max_retries=2, close_mode=close_mode or "adaptive")
                        _regrasp_result = _exec_grasp(env, target_obj, config=_regrasp_cfg,
                                                      skip_approach=False, skip_lift=False)
                        total_steps += _regrasp_result.get("total_step_count", 0)
                        if _regrasp_result.get("success", False):
                            # Re-place after successful regrasp
                            _replace_result = execute_place(env, target_obj, target_place,
                                                           config=_place_cfg, skip_preplace=False,
                                                           phase_callback=_rec_set_phase)
                            total_steps += _replace_result.get("step_count", 0)
                            if _replace_result.get("success", False):
                                successful_steps += 1
                                grasped_object = ""
                                place_done = True
                                successful_steps += 2
                                continue  # skip failure handling below
                    # Recovery failed or not possible
                    skill_ok = False; episode_success = False
                    final_failure_type = _ft
                    failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                    break
                place_done = True
                successful_steps += 2
                # Count release+retreat as successful too since place_state_machine handled them
                successful_steps += 2

            elif step_name == "reach_place" and not use_place_state_machine:
                tgt_pose = env.get_target_pose(target_place)
                place_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"] + 0.10 + place_z_offset}

                if use_pid_controller and pid_ctrl is not None:
                    pid_ok, pid_steps = _pid_reach_loop(
                        env, pid_ctrl, place_target, "reach_place",
                        _GRIPPER_CLOSE, min(2 * _STEP_LIMIT, remaining), _rt,
                        id_ensemble=id_ensemble, phase_id=3,
                    )
                    total_steps += pid_steps
                    phase_trace["reach_place"] += pid_steps
                    if pid_ok:
                        successful_steps += 1
                    else:
                        skill_ok = False; episode_success = False
                        final_failure_type = f"{step_name}_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break
                else:
                    for _ in range(min(2 * _STEP_LIMIT, remaining)):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        dx = np.clip(kp * (place_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (place_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (place_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_CLOSE))
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        total_steps += 1
                        phase_trace["reach_place"] += 1
                        if _dist(eef, place_target) < _rt:
                            successful_steps += 1
                            break

            elif step_name == "release" and not place_done:
                _rec_set_phase("release")
                _r_min = release_steps_min if release_steps_min is not None else _RELEASE_STEPS_MIN
                _r_max = release_steps_max if release_steps_max is not None else min(_ros, remaining)
                _r_stable = post_release_stable_steps if post_release_stable_steps is not None else _POST_RELEASE_STABLE_STEPS
                _r_stable += max(0, stabilization_steps)
                # Min steps: always run at least this many
                for _ in range(min(_r_min, _r_max)):
                    try:
                        env.step(_norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                    except ValueError:
                        skill_ok = False; episode_success = False
                        final_failure_type = f"{step_name}_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break
                    total_steps += 1
                    phase_trace["release"] += 1
                if skill_ok:
                    # Max steps (early exit if gripper is fully open)
                    for _ in range(_r_max - _r_min):
                        if not skill_ok:
                            break
                        try:
                            env.step(_norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        total_steps += 1
                        phase_trace["release"] += 1
                        # Early exit if gripper fully open
                        _grip = env.get_gripper_state()
                        if _grip.get("open", -1.0) > 0.9 and _r_stable > 0:
                            # Run stabilize steps
                            for _ in range(_r_stable):
                                try:
                                    env.step(_norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                                except ValueError:
                                    skill_ok = False; episode_success = False
                                    final_failure_type = f"{step_name}_failed"
                                    failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                                    break
                                total_steps += 1
                                phase_trace["release"] += 1
                            break
                    successful_steps += 1
                grasped_object = ""
                # Stage 7A-10: terminate_after_safe_release
                if terminate_after_safe_release and goal_was_achieved:
                    _terminate_release_triggered = True
                    early_stopped = True
                    break

            elif step_name == "retreat" and not place_done:
                _rec_set_phase("retreat")
                # Stage 7A-10: skip retreat if goal already satisfied
                if skip_retreat_if_goal_satisfied and goal_was_achieved:
                    _skip_retreat_triggered = True
                    successful_steps += 1  # count as success
                    early_stopped = True
                    break
                try:
                    eef = env.get_eef_pose()
                except ValueError:
                    skill_ok = False; episode_success = False
                    final_failure_type = f"{step_name}_failed"
                    failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                    break
                retreat_target = {"x": eef["x"], "y": eef["y"], "z": eef["z"] + 0.15}

                if use_pid_controller and pid_ctrl is not None:
                    pid_ok, pid_steps = _pid_reach_loop(
                        env, pid_ctrl, retreat_target, "retreat",
                        _GRIPPER_OPEN, min(_STEP_LIMIT, remaining), _rt,
                        id_ensemble=id_ensemble, phase_id=5,
                    )
                    total_steps += pid_steps
                    phase_trace["retreat"] += pid_steps
                    if pid_ok:
                        successful_steps += 1
                    else:
                        skill_ok = False; episode_success = False
                        final_failure_type = f"{step_name}_failed"
                        failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                        break
                else:
                    for _ in range(min(_STEP_LIMIT, remaining)):
                        try:
                            eef = env.get_eef_pose()
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        dx = np.clip(kp * (retreat_target["x"] - eef["x"]), -max_delta, max_delta)
                        dy = np.clip(kp * (retreat_target["y"] - eef["y"]), -max_delta, max_delta)
                        dz = np.clip(kp * (retreat_target["z"] - eef["z"]), -max_delta, max_delta)
                        try:
                            env.step(_norm_action(dx, dy, dz, _GRIPPER_OPEN))
                        except ValueError:
                            skill_ok = False; episode_success = False
                            final_failure_type = f"{step_name}_failed"
                            failure_stage_counts[final_failure_type] = failure_stage_counts.get(final_failure_type, 0) + 1
                            break
                        total_steps += 1
                        phase_trace["retreat"] += 1
                        if _dist(eef, retreat_target) < _rt:
                            successful_steps += 1
                            break

            # Build observation for termination monitor and check
            obs = _build_obs_for_monitor(env, cubes, grasped_object)
            signal = term_monitor.check(step_name, obs, step_idx)

            if signal.failed:
                failure_type = signal.failure_type or f"{step_name}_failed"
                failure_stage_counts[failure_type] = failure_stage_counts.get(failure_type, 0) + 1
                episode_success = False
                final_failure_type = failure_type
                skill_ok = False
                break

            # Check goal after each step (for early stop)
            if early_stop_on_goal and not goal_was_achieved:
                if _check_goal_satisfied(env, cubes, goal_predicates):
                    goal_was_achieved = True
                    goal_achieved_at_step = total_steps

        # Check goal after each skill
        if early_stop_on_goal and not goal_was_achieved:
            if _check_goal_satisfied(env, cubes, goal_predicates):
                goal_was_achieved = True
                goal_achieved_at_step = total_steps

        # Early stop after skill if goal achieved (skip remaining skills)
        if early_stop_on_goal and goal_was_achieved:
            early_stopped = True
            break

        if not skill_ok:
            break

    # Final goal check
    object_poses = env.get_all_object_poses()
    targets = dict(object_poses)
    targets["table"] = env.get_object_pose("table")

    physical_predicates = physical_state_to_predicates(object_poses, targets, _PREDICATE_THRESHOLDS)

    goal_ok = False
    predicate_match_rate = 0.0
    if goal_predicates:
        goal_ok = goal_satisfied(physical_predicates, goal_predicates, mode="symbolic")
        symbolic_set = {_to_pdt(p) for p in goal_predicates}
        physical_set = physical_predicates
        match_count = sum(1 for g in symbolic_set if g in physical_set)
        predicate_match_rate = match_count / max(len(symbolic_set), 1)

    # Also check if goal was achieved earlier (for early_stop scenarios where
    # the final state might differ due to retreat movements)
    if not goal_ok and goal_was_achieved:
        # Goal was achieved mid-episode but final state moved away — still count semantic success
        pass  # goal_ok stays False for final state, but semantic_goal_success uses goal_was_achieved

    # Classify timeout type
    hit_horizon = total_steps >= horizon - 5  # close enough to horizon
    timeout_type = None
    if hit_horizon and not early_stopped:
        if goal_was_achieved or goal_ok:
            timeout_type = "timeout_after_goal"
        else:
            timeout_type = "timeout_before_goal"

    # Determine failure type for timeout cases
    if timeout_type and final_failure_type is None:
        final_failure_type = timeout_type
        if timeout_type not in failure_stage_counts:
            failure_stage_counts[timeout_type] = failure_stage_counts.get(timeout_type, 0) + 1

    # Track B: symbolic_success is always 1 (planner produced a valid plan)
    symbolic_success = 1
    # strict_physical_success: execution clean AND goal satisfied AND no timeout
    strict_physical_success = 1 if (episode_success and goal_ok and timeout_type is None) else 0
    # semantic_goal_success: goal was achieved regardless of execution hiccups
    semantic_goal_success = 1 if (goal_ok or goal_was_achieved) else 0

    total_steps = max(total_steps, 1)
    total_skills = max(total_skills, 1)

    # Stage 7A-11: termination stats
    termination_stats = {
        "terminate_on_goal_count": 1 if (early_stopped and goal_was_achieved) else 0,
        "terminate_after_safe_release_count": 1 if _terminate_release_triggered else 0,
        "skip_retreat_if_goal_satisfied_count": 1 if _skip_retreat_triggered else 0,
        "normal_completion_count": 1 if (not hit_horizon and not early_stopped) else 0,
        "horizon_termination_count": 1 if hit_horizon else 0,
    }

    # Stage 7B-1: end episode on recorder if active
    if recorder is not None:
        recorder.end_episode()
    # Stage 7B-2R: if env was wrapped for policy-only (no recorder), end_episode is N/A

    return {
        "execution_success": episode_success,
        "goal_success": bool(goal_ok),
        "semantic_goal_success": semantic_goal_success,
        "strict_physical_success": strict_physical_success,
        "step_success_rate": successful_steps / total_steps,
        "skill_success_rate": successful_skills / total_skills,
        "episode_success": episode_success and bool(goal_ok),
        "failure_stage_distribution": failure_stage_counts,
        "predicate_match_rate": predicate_match_rate,
        "goal_satisfied": bool(goal_ok),
        "num_steps": total_steps,
        "failure_type": final_failure_type,
        "failure_phase": _last_phase_before_horizon or step_name,
        "timeout_type": timeout_type,
        "goal_achieved_at_step": goal_achieved_at_step,
        "early_stopped": early_stopped,
        "symbolic_success": symbolic_success,
        "physical_success": strict_physical_success,
        "reality_gap": reality_gap(symbolic_success, strict_physical_success),
        "semantic_reality_gap": reality_gap(symbolic_success, semantic_goal_success),
        "physical_predicates": sorted(str(p) for p in physical_predicates),
        "strict_physical_success": strict_physical_success,
        "phase_trace": phase_trace,
        "termination_stats": termination_stats,
        # Stage 7B-2R: policy override step counts (0 if no override wrapper)
        "bc_steps": env.bc_steps if hasattr(env, "bc_steps") else 0,
        "scripted_steps": env.scripted_steps if hasattr(env, "scripted_steps") else 0,
        # Stage 20-V: shadow diagnostics from learned executor
        "shadow_diagnostics": env.shadow_diagnostics if hasattr(env, "shadow_diagnostics") else [],
        "learned_predictions": env.learned_predictions if hasattr(env, "learned_predictions") else 0,
        # Stage 20-W: gated execution stats
        "learned_executed_steps": env.learned_executed_steps if hasattr(env, "learned_executed_steps") else 0,
        "fallback_steps": env.fallback_steps if hasattr(env, "fallback_steps") else 0,
    }


def _to_pdt(pred):
    if hasattr(pred, "as_tuple"):
        return pred.as_tuple()
    return tuple(pred)


def run_robosuite_eval(
    n_blocks: int = 2,
    num_episodes: int = 5,
    render: bool = False,
    early_stop_on_goal: bool = True,
    use_place_state_machine: bool = True,
    horizon_aware: bool = True,
    place_retry: bool = True,
    kp: float = 0.05,
    max_delta: float = 0.02,
    # Stage 7A-10: goal-aware termination
    skip_retreat_if_goal_satisfied: bool = False,
    terminate_after_safe_release: bool = False,
    # Stage 7A-10: config propagation for state machines
    grasp_close_steps: int | None = None,
    grasp_max_retries: int | None = None,
    place_max_retries: int | None = None,
    place_mode: str = "simple",
    # Stage 7A-11: event-driven grasp + lift early exit
    holding_check_enabled: bool = False,
    close_steps_min: int | None = None,
    close_steps_max: int | None = None,
    close_stabilize_steps: int | None = None,
    lift_early_exit_enabled: bool = False,
    # Stage 7A-11: release phase enhanced
    release_steps_min: int | None = None,
    release_steps_max: int | None = None,
    post_release_stable_steps: int | None = None,
    # Stage 7A-12: config-specified close mode
    close_mode: str = "fixed",
    close_steps_fast: int | None = None,
    fallback_extend_close_if_not_holding: bool = False,
    # Phase 3.0: PID controller
    use_pid_controller: bool = True,
    # Phase 3.1: ID ensemble for velocity feedforward
    id_ensemble_paths: list | None = None,
) -> dict:
    """Run robosuite embodied stacking evaluation.

    Args:
        n_blocks: Number of blocks (2 for Stack env in robosuite 1.5.2).
        num_episodes: Number of episodes.
        render: Whether to render.
        early_stop_on_goal: If True, stop executing when goal is satisfied.
        use_place_state_machine: If True, use place_state_machine for place steps.
        horizon_aware: If True, check remaining budget before each phase.
        place_retry: If True, allow retry in place_state_machine.

    Returns:
        Dict with Experiment C metrics including dual reality gaps.
    """
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # 3-block: monkey-patch Stack to include cubeC (must happen before env creation)
    if n_blocks >= 3:
        import vlm_nsm.executors.stack_three_cube_env as _three_cube  # noqa: F401

    cubes = [f"cube{i}" for i in range(1, n_blocks + 1)]
    target_stack = cubes[::-1]
    goal_predicates = build_stacking_goal(target_stack)

    # Planner
    planner = PlannerAdapter("bfs")
    plan_result = planner.solve(cubes=cubes, target_stack=target_stack)
    if not plan_result.success:
        raise RuntimeError(f"Planner failed: {plan_result.error}")

    # Robosuite env
    render_mode = "human" if render else "none"
    env = RobosuiteEnvWrapper(
        env_name="Stack",
        robots="Panda",
        cube_names=cubes,
        render_mode=render_mode,
    )

    horizon = env.env.horizon if hasattr(env.env, "horizon") else 1000
    print(f"[eval] env horizon={horizon}", flush=True)
    flags = []
    if early_stop_on_goal: flags.append("early_stop")
    if use_place_state_machine: flags.append("place_sm")
    if horizon_aware: flags.append("horizon_aware")
    if place_retry: flags.append("place_retry")
    print(f"[eval] flags: {flags}", flush=True)
    extra_flags = []
    if skip_retreat_if_goal_satisfied: extra_flags.append("skip_retreat")
    if terminate_after_safe_release: extra_flags.append("term_after_release")
    print(f"[eval] kp={kp} max_delta={max_delta} place_mode={place_mode} extra={extra_flags}", flush=True)

    # Aggregate metrics across episodes
    step_successes: list[float] = []
    skill_successes: list[float] = []
    episode_successes: list[bool] = []
    goal_successes: list[bool] = []
    semantic_goal_successes: list[bool] = []
    strict_physical_successes: list[bool] = []
    failure_stages: dict[str, int] = {}
    all_predicate_rates: list[float] = []
    symbolic_success_count = 0
    strict_physical_success_count = 0
    semantic_goal_success_count = 0
    all_step_counts: list[int] = []
    timeout_types: list[str] = []
    # Stage 7A-11: termination stats aggregation
    term_stats_agg: dict[str, int] = {
        "terminate_on_goal_count": 0,
        "terminate_after_safe_release_count": 0,
        "skip_retreat_if_goal_satisfied_count": 0,
        "normal_completion_count": 0,
        "horizon_termination_count": 0,
    }
    # Stage 7A-11: per-episode data for phase budget analysis
    per_episode_data: list[dict] = []

    for _ in range(num_episodes):
        result = _run_robosuite_episode(
            env, plan_result.actions, cubes, goal_predicates,
            horizon=horizon, early_stop_on_goal=early_stop_on_goal,
            use_place_state_machine=use_place_state_machine,
            horizon_aware=horizon_aware,
            place_retry=place_retry,
            kp=kp, max_delta=max_delta,
            skip_retreat_if_goal_satisfied=skip_retreat_if_goal_satisfied,
            terminate_after_safe_release=terminate_after_safe_release,
            grasp_close_steps=grasp_close_steps,
            grasp_max_retries=grasp_max_retries,
            place_max_retries=place_max_retries,
            place_mode=place_mode,
            holding_check_enabled=holding_check_enabled,
            close_steps_min=close_steps_min,
            close_steps_max=close_steps_max,
            close_stabilize_steps=close_stabilize_steps,
            lift_early_exit_enabled=lift_early_exit_enabled,
            release_steps_min=release_steps_min,
            release_steps_max=release_steps_max,
            post_release_stable_steps=post_release_stable_steps,
            close_mode=close_mode,
            close_steps_fast=close_steps_fast,
            fallback_extend_close_if_not_holding=fallback_extend_close_if_not_holding,
            use_pid_controller=use_pid_controller,
            id_ensemble_paths=id_ensemble_paths,
        )

        step_successes.append(result.get("step_success_rate", 0.0))
        skill_successes.append(result.get("skill_success_rate", 0.0))
        episode_successes.append(result.get("episode_success", False))
        goal_successes.append(result.get("goal_success", False))
        semantic_goal_successes.append(result.get("semantic_goal_success", False))
        strict_physical_successes.append(result.get("strict_physical_success", False))
        all_step_counts.append(result.get("num_steps", 0))
        all_predicate_rates.append(result.get("predicate_match_rate", 0.0))

        ft = result.get("failure_type")
        if ft:
            failure_stages[ft] = failure_stages.get(ft, 0) + 1
        tt = result.get("timeout_type")
        if tt:
            timeout_types.append(tt)

        # Stage 7A-11: aggregate termination stats
        _ts = result.get("termination_stats", {})
        for _k in term_stats_agg:
            term_stats_agg[_k] += _ts.get(_k, 0)

        # Stage 7A-11: collect per-episode data for phase budget export
        per_episode_data.append({
            "phase_trace": result.get("phase_trace", {}),
            "termination_stats": _ts,
            "num_steps": result.get("num_steps", 0),
            "strict_physical_success": result.get("strict_physical_success", False),
            "semantic_goal_success": result.get("semantic_goal_success", False),
            "failure_type": result.get("failure_type"),
        })

        # symbolic_success always counts (planner produced a valid plan for every episode)
        symbolic_success_count += 1
        if result.get("strict_physical_success"):
            strict_physical_success_count += 1
        if result.get("semantic_goal_success"):
            semantic_goal_success_count += 1

    num_ep = max(num_episodes, 1)
    episode_success_count = sum(1 for s in episode_successes if s)
    strict_rg = reality_gap(symbolic_success_count, strict_physical_success_count)
    semantic_rg = reality_gap(symbolic_success_count, semantic_goal_success_count)

    summary = {
        "task": "stacking_robosuite",
        "n_blocks": n_blocks,
        "num_episodes": num_episodes,
        "env_name": "Stack",
        "robot": "Panda",
        "controller": "proportional_osc_pose",
        "predicate_thresholds": dict(_PREDICATE_THRESHOLDS),
        "early_stop_on_goal": early_stop_on_goal,
        "use_place_state_machine": use_place_state_machine,
        "horizon_aware": horizon_aware,
        "place_retry": place_retry,
        "step_success_rate": sum(step_successes) / num_ep if step_successes else 0.0,
        "skill_success_rate": sum(skill_successes) / num_ep if skill_successes else 0.0,
        "episode_success_rate": episode_success_count / num_ep,
        "goal_success_rate": sum(1 for s in goal_successes if s) / num_ep if goal_successes else 0.0,
        "semantic_goal_success_rate": sum(1 for s in semantic_goal_successes if s) / num_ep if semantic_goal_successes else 0.0,
        "strict_physical_success_rate": sum(1 for s in strict_physical_successes if s) / num_ep if strict_physical_successes else 0.0,
        "failure_stage_distribution": failure_stages,
        "predicate_match_rate": sum(all_predicate_rates) / num_ep if all_predicate_rates else 0.0,
        "symbolic_success_count": symbolic_success_count,
        "strict_physical_success_count": strict_physical_success_count,
        "semantic_goal_success_count": semantic_goal_success_count,
        "symbolic_success_rate": symbolic_success_count / num_ep,
        "strict_physical_success_rate": strict_physical_success_count / num_ep,
        "semantic_goal_success_rate": semantic_goal_success_count / num_ep,
        "reality_gap": strict_rg,
        "strict_reality_gap": strict_rg,
        "semantic_reality_gap": semantic_rg,
        "mean_steps": float(np.mean(all_step_counts)) if all_step_counts else 0.0,
        "p95_steps": float(np.percentile(all_step_counts, 95)) if all_step_counts else 0.0,
        "timeout_after_goal_count": sum(1 for t in timeout_types if t == "timeout_after_goal"),
        "timeout_before_goal_count": sum(1 for t in timeout_types if t == "timeout_before_goal"),
        # Stage 7A-11: termination stats
        "termination_stats": term_stats_agg,
        # Stage 7A-11: per-episode data for phase budget and termination analysis
        "per_episode": per_episode_data,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run robosuite embodied stacking evaluation")
    parser.add_argument("--n_blocks", type=int, default=2, choices=[2, 3])
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stop on goal achievement")
    parser.add_argument("--disable-place-state-machine", action="store_true",
                        help="Use simple place instead of place_state_machine")
    parser.add_argument("--disable-horizon-aware", action="store_true",
                        help="Skip remaining budget checks")
    parser.add_argument("--disable-place-retry", action="store_true",
                        help="Disable retry in place_state_machine")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file for execution parameters")
    parser.add_argument("--kp", type=float, default=None,
                        help="Proportional gain (overrides config/module default)")
    parser.add_argument("--max_delta", type=float, default=None,
                        help="Max position delta per step (overrides config/module default)")
    parser.add_argument("--skip-retreat-if-goal-satisfied", action="store_true",
                        help="Skip retreat phase if goal already satisfied")
    parser.add_argument("--terminate-after-safe-release", action="store_true",
                        help="Terminate after release, skip retreat")
    parser.add_argument("--grasp-close-steps", type=int, default=None,
                        help="Override grasp close steps")
    parser.add_argument("--grasp-max-retries", type=int, default=None,
                        help="Override grasp max retries")
    parser.add_argument("--place-max-retries", type=int, default=None,
                        help="Override place max retries")
    parser.add_argument("--place-mode", type=str, default="simple",
                        choices=["simple", "fast", "safe"],
                        help="Place state machine mode")
    parser.add_argument("--holding-check-enabled", action="store_true",
                        help="Enable event-driven close early exit")
    parser.add_argument("--lift-early-exit-enabled", action="store_true",
                        help="Enable lift early exit on object lift detection")
    parser.add_argument("--close-steps-min", type=int, default=None)
    parser.add_argument("--close-steps-max", type=int, default=None)
    parser.add_argument("--close-stabilize-steps", type=int, default=None)
    parser.add_argument("--release-steps-min", type=int, default=None)
    parser.add_argument("--release-steps-max", type=int, default=None)
    parser.add_argument("--post-release-stable-steps", type=int, default=None)
    parser.add_argument("--use-pid-controller", action="store_true", default=True,
                        help="Use OperationalSpacePID (default). Use --no-pid-controller for legacy P-controller.")
    parser.add_argument("--no-pid-controller", action="store_false", dest="use_pid_controller",
                        help="Use legacy P-controller instead of PID")
    parser.add_argument("--id-ensemble", nargs="+", default=None,
                        help="Paths to ID model checkpoints for velocity feedforward")
    args = parser.parse_args()

    # Load config if provided
    kp = 0.05
    max_delta = 0.02
    early_stop_on_goal = not args.no_early_stop
    use_place_state_machine = not args.disable_place_state_machine
    horizon_aware = not args.disable_horizon_aware
    place_retry = not args.disable_place_retry
    skip_retreat_if_goal_satisfied = args.skip_retreat_if_goal_satisfied
    terminate_after_safe_release = args.terminate_after_safe_release
    grasp_close_steps: int | None = args.grasp_close_steps
    grasp_max_retries: int | None = args.grasp_max_retries
    place_max_retries: int | None = args.place_max_retries
    place_mode: str = args.place_mode
    holding_check_enabled: bool = args.holding_check_enabled
    lift_early_exit_enabled: bool = args.lift_early_exit_enabled
    close_steps_min: int | None = args.close_steps_min
    close_steps_max: int | None = args.close_steps_max
    close_stabilize_steps: int | None = args.close_stabilize_steps
    release_steps_min: int | None = args.release_steps_min
    release_steps_max: int | None = args.release_steps_max
    post_release_stable_steps: int | None = args.post_release_stable_steps
    # Stage 7A-12: config-specified close mode params
    close_mode: str = "fixed"
    close_steps_fast: int | None = None
    fallback_extend_close_if_not_holding: bool = False

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = PROJECT_ROOT / args.config
        if config_path.exists():
            import json as _json
            cfg = _json.loads(config_path.read_text())
            print(f"[eval] loaded config from {config_path}")
            kp = cfg.get("kp", kp)
            max_delta = cfg.get("max_delta", max_delta)
            # Override module-level execution constants from config
            import sys
            _mod = sys.modules[__name__]
            if "reach_threshold" in cfg:
                _mod._REACH_THRESHOLD = cfg["reach_threshold"]
            if "grasp_close_steps" in cfg:
                _mod._GRASP_CLOSE_STEPS = cfg["grasp_close_steps"]
            if "lift_height" in cfg:
                _mod._LIFT_HEIGHT = cfg["lift_height"]
            if "release_open_steps" in cfg:
                _mod._RELEASE_OPEN_STEPS = cfg["release_open_steps"]
                _mod._MIN_BUDGET_RELEASE = cfg["release_open_steps"] + 5
            early_stop_on_goal = cfg.get("early_stop_on_goal", early_stop_on_goal)
            use_place_state_machine = cfg.get("use_place_state_machine", use_place_state_machine)
            horizon_aware = cfg.get("horizon_aware", horizon_aware)
            place_retry = cfg.get("place_retry", place_retry)
            # Stage 7A-10: goal-aware termination
            skip_retreat_if_goal_satisfied = cfg.get("skip_retreat_if_goal_satisfied", skip_retreat_if_goal_satisfied)
            terminate_after_safe_release = cfg.get("terminate_after_safe_release", terminate_after_safe_release)
            # Stage 7A-10: state machine config propagation
            if grasp_close_steps is None:
                grasp_close_steps = cfg.get("grasp_close_steps")
            if grasp_max_retries is None:
                grasp_max_retries = cfg.get("grasp_max_retries")
            if place_max_retries is None:
                place_max_retries = cfg.get("place_max_retries")
            place_mode = cfg.get("place_mode", place_mode)
            # Stage 7A-11: event-driven grasp + lift early exit
            holding_check_enabled = cfg.get("holding_check_enabled", holding_check_enabled)
            lift_early_exit_enabled = cfg.get("lift_early_exit_enabled", lift_early_exit_enabled)
            if close_steps_min is None:
                close_steps_min = cfg.get("close_steps_min")
            if close_steps_max is None:
                close_steps_max = cfg.get("close_steps_max")
            if close_stabilize_steps is None:
                close_stabilize_steps = cfg.get("close_stabilize_steps")
            # Stage 7A-11: release phase enhanced
            if release_steps_min is None:
                release_steps_min = cfg.get("release_steps_min")
            if release_steps_max is None:
                release_steps_max = cfg.get("release_steps_max")
            if post_release_stable_steps is None:
                post_release_stable_steps = cfg.get("post_release_stable_steps")
            # Stage 7A-12: close mode from config
            close_mode = cfg.get("close_mode", close_mode)
            if close_steps_fast is None:
                close_steps_fast = cfg.get("close_steps_fast")
            fallback_extend_close_if_not_holding = cfg.get("fallback_extend_close_if_not_holding",
                                                           fallback_extend_close_if_not_holding)

    # CLI overrides take precedence over config file
    if args.kp is not None:
        kp = args.kp
    if args.max_delta is not None:
        max_delta = args.max_delta

    print(f"[eval] kp={kp} max_delta={max_delta}"
          f" early_stop={early_stop_on_goal}"
          f" place_sm={use_place_state_machine}"
          f" horizon_aware={horizon_aware}"
          f" place_retry={place_retry}"
          f" skip_retreat={skip_retreat_if_goal_satisfied}"
          f" term_release={terminate_after_safe_release}"
          f" place_mode={place_mode}"
          f" holding_check={holding_check_enabled}"
          f" lift_early={lift_early_exit_enabled}", flush=True)

    try:
        summary = run_robosuite_eval(
            n_blocks=args.n_blocks,
            num_episodes=args.num_episodes,
            render=args.render,
            early_stop_on_goal=early_stop_on_goal,
            use_place_state_machine=use_place_state_machine,
            horizon_aware=horizon_aware,
            place_retry=place_retry,
            kp=kp, max_delta=max_delta,
            skip_retreat_if_goal_satisfied=skip_retreat_if_goal_satisfied,
            terminate_after_safe_release=terminate_after_safe_release,
            grasp_close_steps=grasp_close_steps,
            grasp_max_retries=grasp_max_retries,
            place_max_retries=place_max_retries,
            place_mode=place_mode,
            holding_check_enabled=holding_check_enabled,
            close_steps_min=close_steps_min,
            close_steps_max=close_steps_max,
            close_stabilize_steps=close_stabilize_steps,
            lift_early_exit_enabled=lift_early_exit_enabled,
            release_steps_min=release_steps_min,
            release_steps_max=release_steps_max,
            post_release_stable_steps=post_release_stable_steps,
            close_mode=close_mode,
            close_steps_fast=close_steps_fast,
            fallback_extend_close_if_not_holding=fallback_extend_close_if_not_holding,
            use_pid_controller=args.use_pid_controller,
            id_ensemble_paths=args.id_ensemble,
        )
    except ImportError as e:
        print(f"ERROR: Robosuite not available: {e}")
        summary = {"error": str(e)}

    print(json.dumps(summary, indent=2))

    results_dir = ensure_dir(PROJECT_ROOT / "generated" / "results")
    summary_path = results_dir / "embodied_stacking_robosuite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
