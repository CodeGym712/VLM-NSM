"""Closed-loop grasp state machine with retry and diagnostics.

Shared grasp logic used by debug_robosuite_primitives.py
and run_embodied_stacking_robosuite.py.

State machine phases per attempt:
  1. APPROACH   — move EEF to above the object
  2. DESCEND    — move EEF down to object height
  3. CLOSE      — close gripper, check contact via open_val
  4. LIFT       — raise EEF, check if object moved up

On failure, retry from APPROACH (up to max_retries).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_OSC_POSE_MAX_DELTA = 0.05
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0
_REACH_THRESHOLD = 0.02
_LIFT_HEIGHT = 0.05
_STEP_LIMIT = 500


def norm_action(dx: float, dy: float, dz: float, gripper: float) -> np.ndarray:
    """Convert physical deltas to 7-dim normalized action [-1, 1]."""
    return np.array([
        np.clip(dx / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dy / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        np.clip(dz / _OSC_POSE_MAX_DELTA, -1.0, 1.0),
        0.0, 0.0, 0.0,
        gripper,
    ], dtype=np.float32)


def _dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


class GraspConfig:
    """Configuration for the grasp state machine.

    Defaults calibrated via empirical testing (Stage 7A-6)::

        reach_threshold=0.015  (tighter approach than 0.02 for reliable grasp)
        descend_offset=0.005   (close to object surface)
        close_steps=70         (allow full closure with object between fingers)
    """

    def __init__(self, **kwargs: Any):
        self.kp: float = kwargs.get("kp", 0.05)
        self.max_delta: float = kwargs.get("max_delta", 0.02)
        self.approach_height: float = kwargs.get("approach_height", 0.08)
        self.descend_offset: float = kwargs.get("descend_offset", 0.005)
        self.close_steps: int = kwargs.get("close_steps", 70)
        self.open_steps: int = kwargs.get("open_steps", 50)
        self.step_limit: int = kwargs.get("step_limit", _STEP_LIMIT)
        self.max_retries: int = kwargs.get("max_retries", 3)
        self.reach_threshold: float = kwargs.get("reach_threshold", 0.015)
        self.lift_height: float = kwargs.get("lift_height", _LIFT_HEIGHT)
        self.lift_threshold: float = kwargs.get("lift_threshold", 0.005)  # min object z-delta to confirm lift
        # Stage 7A-11: event-driven close early exit
        self.close_steps_max: int = kwargs.get("close_steps_max", 70)
        self.close_steps_min: int = kwargs.get("close_steps_min", 15)
        self.close_stabilize_steps: int = kwargs.get("close_stabilize_steps", 5)
        self.holding_check_enabled: bool = kwargs.get("holding_check_enabled", False)
        # Stage 7A-11: lift early exit
        self.lift_early_exit_enabled: bool = kwargs.get("lift_early_exit_enabled", False)
        # Stage 7A-12: adaptive close mode and grasp confidence
        self.close_mode: str = kwargs.get("close_mode", "fixed")  # "fixed", "adaptive", "aggressive"
        self.close_steps_fast: int = kwargs.get("close_steps_fast", 15)  # fast-close steps for candidate B/C
        self.fallback_extend_close_if_not_holding: bool = kwargs.get("fallback_extend_close_if_not_holding", False)
        self.log_grasp_confidence: bool = kwargs.get("log_grasp_confidence", False)
        self.mark_grasp_uncertain: bool = kwargs.get("mark_grasp_uncertain", False)
        # Stage Control-v2: MICRO_ALIGN before grasp
        self.micro_align_enabled: bool = kwargs.get("micro_align_enabled", True)
        self.micro_xy_tol: float = kwargs.get("micro_xy_tol", 0.008)       # 8mm default
        self.micro_z_tol: float = kwargs.get("micro_z_tol", 0.008)
        self.micro_max_delta: float = kwargs.get("micro_max_delta", 0.004) # 4mm default
        self.micro_align_max_steps: int = kwargs.get("micro_align_max_steps", 30)
        # Object-specific overrides (applied per-object by caller)
        self.object_profile: dict[str, Any] | None = kwargs.get("object_profile")
        # Holding confidence
        self.holding_confidence_threshold: float = kwargs.get("holding_confidence_threshold", 0.7)
        self.regrasp_enabled: bool = kwargs.get("regrasp_enabled", True)


def grasp_attempt(env, object_name: str, config: GraspConfig | None = None,
                  skip_approach: bool = False, skip_lift: bool = False
                  ) -> tuple[bool, dict[str, Any]]:
    """Single grasp attempt: approach, descend, close, lift.

    Args:
        env: Robosuite environment wrapper.
        object_name: Canonical object name (e.g. "cube1").
        config: GraspConfig parameters.
        skip_approach: If True, skip the approach phase (EEF assumed already near object).
        skip_lift: If True, skip the lift phase (caller handles lifting).

    Returns:
        (success, diagnostics_dict). The dict includes ``step_count``.
        When skip_lift=True, success is based on contact detection alone.
    """
    if config is None:
        config = GraspConfig()

    diag: dict[str, Any] = {}
    step_count: int = 0
    obj_pose = env.get_object_pose(object_name)
    diag["object_z_before"] = obj_pose["z"]

    # Phase 1: APPROACH — move to above the object
    if skip_approach:
        diag["approach"] = {
            "success": True,
            "steps": 0,
            "target_distance": 0.0,
            "final_eef": env.get_eef_pose(),
        }
        approach_success = True
    else:
        approach_target = {
            "x": obj_pose["x"], "y": obj_pose["y"],
            "z": obj_pose["z"] + config.approach_height,
        }
        initial_eef = env.get_eef_pose()
        diag["initial_eef"] = initial_eef
        diag["approach_target"] = approach_target

        approach_steps = 0
        approach_success = False
        for _ in range(config.step_limit):
            eef = env.get_eef_pose()
            dx = np.clip(config.kp * (approach_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
            dy = np.clip(config.kp * (approach_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
            dz = np.clip(config.kp * (approach_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
            env.step(norm_action(dx, dy, dz, _GRIPPER_OPEN))
            step_count += 1
            approach_steps += 1
            if _dist(eef, approach_target) < config.reach_threshold:
                approach_success = True
                break

        eef_after_approach = env.get_eef_pose()
        diag["approach"] = {
            "success": approach_success,
            "steps": approach_steps,
            "final_eef": {"x": eef_after_approach["x"], "y": eef_after_approach["y"], "z": eef_after_approach["z"]},
            "target_distance": _dist(eef_after_approach, approach_target),
        }

    if not approach_success:
        diag["step_count"] = step_count
        return False, {**diag, "failure_subtype": "prereach"}

    # Phase 2: DESCEND — move down to object height
    obj_pose = env.get_object_pose(object_name)
    descend_target = {
        "x": obj_pose["x"], "y": obj_pose["y"],
        "z": obj_pose["z"] + config.descend_offset,
    }

    descend_steps = 0
    for _ in range(config.step_limit):
        eef = env.get_eef_pose()
        dx = np.clip(config.kp * (descend_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
        dy = np.clip(config.kp * (descend_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
        dz = np.clip(config.kp * (descend_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
        env.step(norm_action(dx, dy, dz, _GRIPPER_OPEN))
        step_count += 1
        descend_steps += 1
        if _dist(eef, descend_target) < config.reach_threshold:
            break

    eef_after_descend = env.get_eef_pose()
    # Compute EEF-to-object offset for diagnostics
    obj_xy = (obj_pose["x"], obj_pose["y"])
    eef_xy = (eef_after_descend["x"], eef_after_descend["y"])
    diag["descend"] = {
        "steps": descend_steps,
        "final_eef": {"x": eef_after_descend["x"], "y": eef_after_descend["y"], "z": eef_after_descend["z"]},
        "eef_to_obj_xy": math.hypot(eef_xy[0] - obj_xy[0], eef_xy[1] - obj_xy[1]),
        "eef_to_obj_z": eef_after_descend["z"] - obj_pose["z"],
    }

    # Phase 2.5: MICRO_ALIGN — fine-tune XY/Z alignment before grasp
    micro_steps = 0
    if config.micro_align_enabled:
        obj_pose = env.get_object_pose(object_name)
        micro_target = {
            "x": obj_pose["x"], "y": obj_pose["y"],
            "z": obj_pose["z"] + config.descend_offset,
        }
        for _ in range(config.micro_align_max_steps):
            eef = env.get_eef_pose()
            xy_err = math.hypot(micro_target["x"] - eef["x"],
                               micro_target["y"] - eef["y"])
            z_err = abs(micro_target["z"] - eef["z"])
            if xy_err < config.micro_xy_tol and z_err < config.micro_z_tol:
                break
            dx = np.clip(config.kp * (micro_target["x"] - eef["x"]),
                        -config.micro_max_delta, config.micro_max_delta)
            dy = np.clip(config.kp * (micro_target["y"] - eef["y"]),
                        -config.micro_max_delta, config.micro_max_delta)
            dz = np.clip(config.kp * (micro_target["z"] - eef["z"]),
                        -config.micro_max_delta, config.micro_max_delta)
            env.step(norm_action(dx, dy, dz, _GRIPPER_OPEN))
            step_count += 1
            micro_steps += 1
    diag["micro_align"] = {"steps": micro_steps, "enabled": config.micro_align_enabled}

    # Phase 3: CLOSE gripper (Stage 7A-11/12: adaptive event-driven close)
    gripper_before = env.get_gripper_state()
    _open_before = gripper_before.get("open", -1.0)
    diag["gripper_open_before"] = _open_before

    # Resolve close mode
    _close_mode = config.close_mode
    if _close_mode == "fixed" and config.holding_check_enabled:
        _close_mode = "adaptive"  # upgrade for backward compatibility
    _use_event_driven = _close_mode in ("adaptive", "aggressive")
    _cs_min = 5 if _close_mode == "aggressive" else config.close_steps_min
    _cs_stable = 2 if _close_mode == "aggressive" else config.close_stabilize_steps
    close_steps_to_run = (config.close_steps_max if _use_event_driven
                          else config.close_steps)
    _contact_detected = False
    _contact_at_step = None
    for _step_i in range(close_steps_to_run):
        env.step(norm_action(0.0, 0.0, 0.0, _GRIPPER_CLOSE))
        step_count += 1
        # Event-driven close: check for contact early
        if _use_event_driven and _step_i >= _cs_min:
            _cur_grip = env.get_gripper_state()
            _cur_open = _cur_grip.get("open", -1.0)
            _cur_has_obj = _cur_grip.get("has_object", False)
            _partial_close = (_open_before > 0.03 and _cur_open > 0.005
                              and _cur_open < _open_before * 0.8)
            if _partial_close or _cur_has_obj:
                if not _contact_detected:
                    _contact_detected = True
                    _contact_at_step = step_count
                # After stabilize steps, exit early
                if step_count - _contact_at_step >= _cs_stable:
                    break

    gripper_after = env.get_gripper_state()
    _open_after = gripper_after.get("open", -1.0)
    diag["gripper_open_after"] = _open_after
    diag["gripper_has_object_after"] = gripper_after.get("has_object", False)

    # Detect contact: gripper was open, closed significantly but not fully.
    # Fully closed (~0.0) = fingers met without object = nocontact.
    # Partially closed (~0.027) = object between fingers = contact.
    _contact = (_open_before > 0.03             # gripper was open
                and _open_after > 0.005          # not fully closed (object between fingers)
                and _open_after < _open_before * 0.8)  # closed significantly

    obj_z_after_close = env.get_object_pose(object_name)["z"]
    diag["obj_z_after_close"] = obj_z_after_close

    if not _contact:
        diag["step_count"] = step_count
        return False, {**diag, "failure_subtype": "nocontact"}

    if skip_lift:
        # Lift handled by caller; contact detection is sufficient
        diag["step_count"] = step_count
        return True, diag

    # Phase 4: LIFT — use object pose for lift target (works even with skip_approach)
    _obj_now = env.get_object_pose(object_name)
    lift_target = {
        "x": _obj_now["x"], "y": _obj_now["y"],
        "z": _obj_now["z"] + config.lift_height,
    }

    lift_steps = 0
    _obj_z_at_lift_start = _obj_now["z"]
    for _ in range(config.step_limit):
        eef = env.get_eef_pose()
        dx = np.clip(config.kp * (lift_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
        dy = np.clip(config.kp * (lift_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
        dz = np.clip(config.kp * (lift_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
        env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
        step_count += 1
        lift_steps += 1
        # Stage 7A-11: lift early exit — if object clearly lifted, break early
        if config.lift_early_exit_enabled:
            _cur_obj_z = env.get_object_pose(object_name)["z"]
            if _cur_obj_z - _obj_z_at_lift_start > config.lift_threshold:
                break
        if _dist(eef, lift_target) < config.reach_threshold:
            break

    obj_z_after_lift = env.get_object_pose(object_name)["z"]
    obj_z_delta = obj_z_after_lift - obj_z_after_close
    diag["lift"] = {
        "steps": lift_steps,
        "obj_z_after_lift": obj_z_after_lift,
        "obj_z_delta": obj_z_delta,
    }

    diag["step_count"] = step_count

    # Stage 7A-12: grasp confidence heuristic (rules-based, no ML)
    _open_ratio = (_open_after / max(_open_before, 1e-8)
                   if _open_before > 0.03 else 0.0)
    conf = 0.0
    if _contact:
        conf += 0.4  # gripper closed on something
    if _open_ratio < 0.3:
        conf += 0.3  # gripper closed significantly
    if obj_z_delta > config.lift_threshold:
        conf += 0.3  # object lifted
    diag["confidence"] = min(conf, 1.0)
    diag["grasp_open_ratio"] = _open_ratio

    if obj_z_delta > config.lift_threshold:
        diag["step_count"] = step_count
        return True, diag
    else:
        diag["step_count"] = step_count
        return False, {**diag, "failure_subtype": "slip"}


def execute_grasp(env, object_name: str, config: GraspConfig | None = None,
                  skip_approach: bool = False, skip_lift: bool = False) -> dict[str, Any]:
    """Execute closed-loop grasp with retry.

    Args:
        env: Robosuite environment wrapper.
        object_name: Canonical object name (e.g. "cube1").
        config: GraspConfig parameters.
        skip_approach: If True, skip approach (EEF assumed already near object).
        skip_lift: If True, skip lift phase (caller handles it).

    Returns:
        Dict with keys: success, total_attempts, attempts (list of per-attempt diagnostics),
        failure_type, object_z_before, object_z_after, object_z_delta, config.
    """
    if config is None:
        config = GraspConfig()

    attempts: list[dict[str, Any]] = []
    overall_success = False
    total_step_count: int = 0
    _max_total_steps = 900  # leave margin below default horizon 1000

    for attempt_num in range(config.max_retries):
        if total_step_count >= _max_total_steps:
            break
        if attempt_num > 0:
            # Re-open gripper before retry
            try:
                for _ in range(config.open_steps):
                    env.step(norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
                    total_step_count += 1
            except ValueError:
                break  # episode terminated, no more retries

        try:
            success, diag = grasp_attempt(env, object_name, config,
                                           skip_approach=skip_approach, skip_lift=skip_lift)
        except ValueError:
            # Episode terminated mid-attempt (horizon reached)
            diag = {"failure_subtype": "horizon", "step_count": 0}
            success = False
        diag["attempt"] = attempt_num + 1
        total_step_count += diag.get("step_count", 0)
        attempts.append(diag)

        if success:
            overall_success = True
            break

    final_obj_pose = env.get_object_pose(object_name)
    obj_z_before = attempts[0].get("object_z_before", 0.0)
    obj_z_after = final_obj_pose["z"]

    # Stage 7A-12: aggregate grasp confidence across attempts
    _confidences = [a.get("confidence", 0.0) for a in attempts]
    _best_confidence = max(_confidences) if _confidences else 0.0
    _is_uncertain = config.mark_grasp_uncertain and _best_confidence < 0.5

    return {
        "success": overall_success,
        "total_attempts": len(attempts),
        "max_retries": config.max_retries,
        "total_step_count": total_step_count,
        "config": {
            "kp": config.kp,
            "max_delta": config.max_delta,
            "approach_height": config.approach_height,
            "descend_offset": config.descend_offset,
            "close_steps": config.close_steps,
            "max_retries": config.max_retries,
            "lift_height": config.lift_height,
            "lift_threshold": config.lift_threshold,
            "close_steps_max": config.close_steps_max,
            "close_steps_min": config.close_steps_min,
            "close_stabilize_steps": config.close_stabilize_steps,
            "holding_check_enabled": config.holding_check_enabled,
            "lift_early_exit_enabled": config.lift_early_exit_enabled,
            "close_mode": config.close_mode,
            "close_steps_fast": config.close_steps_fast,
            "fallback_extend_close_if_not_holding": config.fallback_extend_close_if_not_holding,
            "log_grasp_confidence": config.log_grasp_confidence,
            "mark_grasp_uncertain": config.mark_grasp_uncertain,
        },
        "object_z_before": obj_z_before,
        "object_z_after": obj_z_after,
        "object_z_delta": obj_z_after - obj_z_before,
        "failure_type": None if overall_success else attempts[-1].get("failure_subtype", "grasp_failed"),
        "attempts": attempts,
        # Stage 7A-12: grasp confidence
        "grasp_confidence": _best_confidence,
        "grasp_uncertain": _is_uncertain,
        "grasp_confidence_logged": config.log_grasp_confidence,
    }
