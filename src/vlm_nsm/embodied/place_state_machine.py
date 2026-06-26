"""Closed-loop place state machine with retry and verification.

State machine phases per attempt:
  1. PREPLACE     — move EEF to above the target placement location
  2. XY_MICRO_ALIGN — fine-tune XY alignment above target
  3. DESCEND      — lower object onto the target surface
  4. RELEASE      — open gripper to release object
  5. HOLD         — brief pause with gripper open (let object settle)
  6. RETREAT      — move EEF up and away
  7. VERIFY       — check on-predicate; retry if placement missed

On failure, retry from PREPLACE (up to max_retries).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_OSC_POSE_MAX_DELTA = 0.05
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0
_REACH_THRESHOLD = 0.02


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


def _xy_dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


# Place state machine presets
_PLACE_MODE_PRESETS: dict[str, dict] = {
    "simple": {
        "preplace_height": 0.15,
        "descend_offset": 0.02,
        "xy_align_threshold": 0.01,
        "hold_steps": 10,
        "xy_micro_align_steps": 30,
    },
    "fast": {
        "preplace_height": 0.08,
        "descend_offset": 0.02,
        "xy_align_threshold": 0.015,
        "hold_steps": 5,
        "xy_micro_align_steps": 10,
    },
    "safe": {
        "preplace_height": 0.15,
        "descend_offset": 0.01,
        "xy_align_threshold": 0.01,
        "hold_steps": 20,
        "xy_micro_align_steps": 50,
    },
}


def _resolve_place_preset(mode: str = "simple", **overrides: Any) -> dict:
    """Resolve place config preset with overrides.

    Args:
        mode: One of "simple", "fast", "safe".
        **overrides: Individual parameter overrides (take precedence over preset).

    Returns:
        Dict of resolved config parameters.
    """
    base = dict(_PLACE_MODE_PRESETS.get(mode, _PLACE_MODE_PRESETS["simple"]))
    base.update(overrides)
    return base


class PlaceConfig:
    """Configuration for the place state machine.

    Defaults calibrated via empirical testing::

        preplace_height=0.15     (approach height above target)
        descend_offset=0.02     (how close to surface before release)
        xy_align_threshold=0.01 (fine XY alignment tolerance)
        release_steps=30        (steps to fully open gripper)
        hold_steps=10           (pause after release)
        retreat_height=0.15     (how far to move up after release)
        xy_micro_align_steps=30 (max steps for micro alignment)

    Mode presets (apply before kwargs):
      - "simple": balanced defaults
      - "fast": lower preplace, fewer align steps
      - "safe": tighter descend, longer hold, more align steps
    """

    def __init__(self, **kwargs: Any):
        mode = kwargs.pop("mode", "simple")
        preset = _PLACE_MODE_PRESETS.get(mode, _PLACE_MODE_PRESETS["simple"])
        resolved = dict(preset)
        resolved.update(kwargs)
        self.kp: float = resolved.get("kp", 0.05)
        self.max_delta: float = resolved.get("max_delta", 0.02)
        self.preplace_height: float = resolved.get("preplace_height", 0.15)
        self.descend_offset: float = resolved.get("descend_offset", 0.02)
        self.xy_align_threshold: float = resolved.get("xy_align_threshold", 0.01)
        self.release_steps: int = resolved.get("release_steps", 30)
        self.hold_steps: int = resolved.get("hold_steps", 10)
        self.retreat_height: float = resolved.get("retreat_height", 0.15)
        self.step_limit: int = resolved.get("step_limit", 200)
        self.max_retries: int = resolved.get("max_retries", 2)
        self.reach_threshold: float = resolved.get("reach_threshold", 0.02)
        self.xy_micro_align_steps: int = resolved.get("xy_micro_align_steps", 30)
        self.xy_align_kp: float = resolved.get("xy_align_kp", 0.03)
        # Phase 1A: enhanced diagnostics + per-task profiles
        self.micro_place_enabled: bool = resolved.get("micro_place_enabled", True)
        self.micro_place_xy_tol: float = resolved.get("micro_place_xy_tol", 0.006)
        self.micro_place_max_delta: float = resolved.get("micro_place_max_delta", 0.0025)
        self.micro_place_max_steps: int = resolved.get("micro_place_max_steps", 20)
        self.release_stabilize_steps: int = resolved.get("release_stabilize_steps", 10)
        self.release_verify_steps: int = resolved.get("release_verify_steps", 8)
        self.preplace_accept_xy_tol: float = resolved.get("preplace_accept_xy_tol", 0.015)


# Per-task place profiles
_GREEN_ON_RED_PLACE_PROFILE = {
    "mode": "safe",
    "micro_place_xy_tol": 0.005,
    "micro_place_max_delta": 0.002,
    "micro_place_max_steps": 30,
    "descend_offset": 0.015,
    "release_stabilize_steps": 15,
    "release_verify_steps": 12,
    "xy_align_threshold": 0.008,
    "preplace_accept_xy_tol": 0.012,
    # Phase 1B: holding-aware damping for green cube transport
    "kp": 0.03,         # reduced from 0.05 (slower approach)
    "max_delta": 0.010, # reduced from 0.02 (half speed to prevent slip)
}


def get_place_config_for_task(
    pick_object: str, target_object: str, **overrides
) -> "PlaceConfig":
    """Get PlaceConfig with per-task profile applied."""
    if pick_object == "cube2" and target_object == "cube1":
        # green on red: most failure-prone, use conservative profile
        profile = dict(_GREEN_ON_RED_PLACE_PROFILE)
        profile.update(overrides)
        return PlaceConfig(**profile)
    return PlaceConfig(**overrides)


def place_attempt(
    env, object_name: str, target_name: str,
    config: PlaceConfig | None = None,
    skip_preplace: bool = False,
    phase_callback=None,
) -> tuple[bool, dict[str, Any]]:
    """Single place attempt: preplace, xy_micro_align, descend, release, hold, retreat.

    Args:
        env: Robosuite environment wrapper.
        object_name: Canonical object name being held (e.g. "cube2").
        target_name: Target surface name (e.g. "cube1" or "table").
        config: PlaceConfig parameters.
        skip_preplace: If True, skip preplace (EEF assumed already near target).

    Returns:
        (success, diagnostics_dict).
    """
    if config is None:
        config = PlaceConfig()

    diag: dict[str, Any] = {}
    step_count: int = 0

    # Get target pose
    tgt_pose = env.get_target_pose(target_name)
    diag["target_pose"] = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": tgt_pose["z"]}

    # Phase 1: PREPLACE — move to above target
    # High-place mode: lift first → XY align at safe height → slow descend
    high_place = getattr(config, 'high_place_enabled', False)
    preplace_fast = getattr(config, 'preplace_fast_mode', False)
    _kp = 0.25 if preplace_fast else config.kp
    _max_delta = 0.05 if preplace_fast else config.max_delta
    _preplace_mode = "high_place" if high_place else ("fast" if preplace_fast else "p_control")
    _high_z_margin = getattr(config, 'high_place_z_margin', 0.06)

    if skip_preplace:
        diag["preplace"] = {"success": True, "steps": 0, "mode": _preplace_mode}
        preplace_success = True
    elif high_place:
        # Step A: Lift to safe height first (Z-only move, no XY)
        eef_start = env.get_eef_pose()
        safe_z = max(eef_start["z"], tgt_pose["z"] + _high_z_margin)
        lift_target = {"x": eef_start["x"], "y": eef_start["y"], "z": safe_z}
        preplace_steps = 0
        for _ in range(config.step_limit):
            try: eef = env.get_eef_pose()
            except ValueError: break
            dx = 0.0  # no XY movement during lift
            dy = 0.0
            dz = np.clip(config.kp * (lift_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
            try: env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
            except ValueError: break
            step_count += 1; preplace_steps += 1
            if abs(eef["z"] - lift_target["z"]) < config.reach_threshold:
                break

        # Step B: XY align at safe height
        xy_target = {"x": tgt_pose["x"], "y": tgt_pose["y"], "z": safe_z}
        _xy_kp = 0.03; _xy_max_d = 0.008
        xy_steps = 0
        for _ in range(config.xy_micro_align_steps):
            try: eef = env.get_eef_pose()
            except ValueError: break
            dx = np.clip(_xy_kp * (xy_target["x"] - eef["x"]), -_xy_max_d, _xy_max_d)
            dy = np.clip(_xy_kp * (xy_target["y"] - eef["y"]), -_xy_max_d, _xy_max_d)
            try: env.step(norm_action(dx, dy, 0.0, _GRIPPER_CLOSE))
            except ValueError: break
            step_count += 1; preplace_steps += 1; xy_steps += 1
            if _xy_dist(eef, tgt_pose) < config.xy_align_threshold:
                break

        # Step C: Slow descend to preplace height
        descend_target = {"x": tgt_pose["x"], "y": tgt_pose["y"],
                          "z": tgt_pose["z"] + config.preplace_height}
        _desc_kp = 0.03; _desc_max_d = 0.006
        for _ in range(config.step_limit):
            try: eef = env.get_eef_pose()
            except ValueError: break
            dx = np.clip(_desc_kp * (descend_target["x"] - eef["x"]), -_desc_max_d, _desc_max_d)
            dy = np.clip(_desc_kp * (descend_target["y"] - eef["y"]), -_desc_max_d, _desc_max_d)
            dz = np.clip(_desc_kp * (descend_target["z"] - eef["z"]), -_desc_max_d, _desc_max_d)
            try: env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
            except ValueError: break
            step_count += 1; preplace_steps += 1
            if _dist(eef, descend_target) < config.reach_threshold:
                preplace_success = True
                break
        diag["preplace_high"] = {"lift_steps": preplace_steps - xy_steps - (preplace_steps - xy_steps - max(0, preplace_steps - xy_steps)),
                                  "xy_steps": xy_steps}
    else:
        preplace_target = {
            "x": tgt_pose["x"], "y": tgt_pose["y"],
            "z": tgt_pose["z"] + config.preplace_height,
        }
        preplace_steps = 0
        preplace_success = False
        for _ in range(config.step_limit):
            try: eef = env.get_eef_pose()
            except ValueError: break
            dx = np.clip(_kp * (preplace_target["x"] - eef["x"]), -_max_delta, _max_delta)
            dy = np.clip(_kp * (preplace_target["y"] - eef["y"]), -_max_delta, _max_delta)
            dz = np.clip(_kp * (preplace_target["z"] - eef["z"]), -_max_delta, _max_delta)
            try: env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
            except ValueError: break
            step_count += 1; preplace_steps += 1
            if _dist(eef, preplace_target) < config.reach_threshold:
                preplace_success = True
                break
        eef_final = env.get_eef_pose()
        obj_pose = env.get_object_pose(object_name)
        diag["preplace"] = {
            "success": preplace_success,
            "steps": preplace_steps,
            "target_distance": _dist(eef_final, preplace_target),
            "eef_xy_error": math.hypot(eef_final["x"] - preplace_target["x"],
                                       eef_final["y"] - preplace_target["y"]),
            "eef_z_error": eef_final["z"] - preplace_target["z"],
            "held_obj_xy_error": math.hypot(obj_pose["x"] - preplace_target["x"],
                                            obj_pose["y"] - preplace_target["y"]),
            "held_obj_z": obj_pose["z"],
            "target_z": preplace_target["z"],
            "preplace_height": config.preplace_height,
            "phase_steps": preplace_steps,
            "mode": _preplace_mode,
        }

    if not preplace_success:
        eef_final = env.get_eef_pose()
        obj_pose = env.get_object_pose(object_name)
        return False, {
            **diag,
            "step_count": step_count,
            "failure_subtype": "preplace",
            "failure_detail": {
                "eef_xy_err": math.hypot(eef_final["x"] - preplace_target["x"],
                                         eef_final["y"] - preplace_target["y"]),
                "eef_z_err": eef_final["z"] - preplace_target["z"],
                "held_obj_z": obj_pose["z"],
                "held_obj_xy_err": math.hypot(obj_pose["x"] - preplace_target["x"],
                                              obj_pose["y"] - preplace_target["y"]),
                "preplace_height": config.preplace_height,
                "phase_steps": preplace_steps,
                "reach_threshold": config.reach_threshold,
            },
        }

    # Phase 2: XY_MICRO_ALIGN — fine-tune XY alignment
    eef_before = env.get_eef_pose()
    xy_dist = _xy_dist(eef_before, tgt_pose)
    align_steps = 0
    if xy_dist > config.xy_align_threshold:
        align_target = {
            "x": tgt_pose["x"], "y": tgt_pose["y"],
            "z": eef_before["z"],  # keep current Z
        }
        for _ in range(config.xy_micro_align_steps):
            try:
                eef = env.get_eef_pose()
            except ValueError:
                break
            dx = np.clip(config.xy_align_kp * (align_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
            dy = np.clip(config.xy_align_kp * (align_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
            dz = 0.0
            try:
                env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
            except ValueError:
                break
            step_count += 1
            align_steps += 1
            if _xy_dist(eef, tgt_pose) < config.xy_align_threshold:
                break

    eef_after_align = env.get_eef_pose()
    diag["xy_micro_align"] = {
        "steps": align_steps,
        "xy_dist_before": xy_dist,
        "xy_dist_after": _xy_dist(eef_after_align, tgt_pose),
    }

    # Phase 3: DESCEND — lower object onto surface
    descend_target = {
        "x": tgt_pose["x"], "y": tgt_pose["y"],
        "z": tgt_pose["z"] + config.descend_offset,
    }
    descend_steps = 0
    for _ in range(config.step_limit):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            break
        dx = np.clip(config.kp * (descend_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
        dy = np.clip(config.kp * (descend_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
        dz = np.clip(config.kp * (descend_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
        try:
            env.step(norm_action(dx, dy, dz, _GRIPPER_CLOSE))
        except ValueError:
            break
        step_count += 1
        descend_steps += 1
        if _dist(eef, descend_target) < config.reach_threshold:
            break

    eef_after_descend = env.get_eef_pose()
    diag["descend"] = {
        "steps": descend_steps,
        "eef_z_after": eef_after_descend["z"],
        "target_z_offset": eef_after_descend["z"] - tgt_pose["z"],
    }

    # Phase 4: RELEASE — open gripper
    if phase_callback:
        phase_callback("release")
    # Direct phase recording: set _phase attr on env for collection scripts
    if hasattr(env, '_phase'):
        env._phase = "release"
    obj_z_before_release = env.get_object_pose(object_name)["z"]
    for _ in range(config.release_steps):
        try:
            env.step(norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
        except ValueError:
            break
        step_count += 1

    obj_z_after_release = env.get_object_pose(object_name)["z"]
    diag["release"] = {
        "steps": config.release_steps,
        "obj_z_before": obj_z_before_release,
        "obj_z_after": obj_z_after_release,
        "obj_z_delta": obj_z_after_release - obj_z_before_release,
    }

    # Phase 5: HOLD — brief settle
    for _ in range(config.hold_steps):
        try:
            env.step(norm_action(0.0, 0.0, 0.0, _GRIPPER_OPEN))
        except ValueError:
            break
        step_count += 1
    diag["hold"] = {"steps": config.hold_steps}

    # Phase 6: RETREAT — move up
    if phase_callback:
        phase_callback("retreat")
    if hasattr(env, '_phase'):
        env._phase = "retreat"
    eef_before_retreat = env.get_eef_pose()
    retreat_target = {
        "x": eef_before_retreat["x"], "y": eef_before_retreat["y"],
        "z": eef_before_retreat["z"] + config.retreat_height,
    }
    retreat_steps = 0
    for _ in range(config.step_limit):
        try:
            eef = env.get_eef_pose()
        except ValueError:
            break
        dx = np.clip(config.kp * (retreat_target["x"] - eef["x"]), -config.max_delta, config.max_delta)
        dy = np.clip(config.kp * (retreat_target["y"] - eef["y"]), -config.max_delta, config.max_delta)
        dz = np.clip(config.kp * (retreat_target["z"] - eef["z"]), -config.max_delta, config.max_delta)
        try:
            env.step(norm_action(dx, dy, dz, _GRIPPER_OPEN))
        except ValueError:
            break
        step_count += 1
        retreat_steps += 1
        if _dist(eef, retreat_target) < config.reach_threshold:
            break
    diag["retreat"] = {"steps": retreat_steps}

    diag["step_count"] = step_count
    return True, diag


def execute_place(
    env, object_name: str, target_name: str,
    config: PlaceConfig | None = None,
    skip_preplace: bool = False,
    phase_callback=None,
) -> dict[str, Any]:
    """Execute closed-loop place with retry and on-predicate verification.

    Args:
        env: Robosuite environment wrapper.
        object_name: Canonical object name being held.
        target_name: Target surface name.
        config: PlaceConfig parameters.
        skip_preplace: If True, skip preplace phase.
        phase_callback: Optional callable(phase_name) called at each sub-phase
            transition (release, retreat) for HDF5 recording.

    Returns:
        Dict with keys: success, total_attempts, failure_type, on_predicate_ok, step_count.
    """
    if config is None:
        config = PlaceConfig()

    attempts: list[dict[str, Any]] = []
    overall_success = False
    total_step_count: int = 0
    on_predicate_ok = False
    final_failure_subtype: str | None = None

    for attempt_num in range(config.max_retries):
        if attempt_num > 0:
            # Re-approach before retry
            skip_preplace = False

        try:
            success, diag = place_attempt(env, object_name, target_name, config,
                                           skip_preplace=skip_preplace,
                                           phase_callback=phase_callback)
        except ValueError:
            diag = {"failure_subtype": "horizon", "step_count": 0}
            success = False

        diag["attempt"] = attempt_num + 1
        total_step_count += diag.get("step_count", 0)
        attempts.append(diag)

        if not success:
            final_failure_subtype = diag.get("failure_subtype", "place_failed")
            continue

        # Phase 7: VERIFY — check on-predicate
        try:
            all_poses = env.get_all_object_poses()
            obj_z = all_poses.get(object_name, {}).get("z", 0.0)
            tgt_pose = env.get_target_pose(target_name)
            tgt_z = tgt_pose["z"]
            obj_xy = (all_poses.get(object_name, {}).get("x", 0.0),
                      all_poses.get(object_name, {}).get("y", 0.0))
            tgt_xy = (tgt_pose["x"], tgt_pose["y"])
            xy_err = math.hypot(obj_xy[0] - tgt_xy[0], obj_xy[1] - tgt_xy[1])

            # Check: object is on or near the target surface
            from vlm_nsm.core.physical_predicate_extractor import extract_on_predicates

            thresholds = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}
            on_preds = extract_on_predicates(all_poses, dict(all_poses), thresholds)
            on_predicate_ok = any(
                len(p) == 3 and p[1] == object_name and p[2] == target_name
                for p in on_preds
            )

            diag["verify"] = {
                "obj_z": obj_z,
                "target_z": tgt_z,
                "z_diff": obj_z - tgt_z,
                "xy_error": xy_err,
                "on_predicate": on_predicate_ok,
            }

            if on_predicate_ok:
                overall_success = True
                break
            else:
                final_failure_subtype = "verify_failed"
                # Fix H4: Break immediately so caller can regrasp before retrying.
                # The caller (run_embodied_stacking_robosuite.py) handles the
                # regrasp → re-place recovery loop when success=False.
                break
        except ValueError:
            diag["verify"] = {"error": "env_error", "on_predicate": False}
            final_failure_subtype = "verify_error"
            continue

    if not overall_success and final_failure_subtype is None:
        final_failure_subtype = "place_failed"

    return {
        "success": overall_success,
        "on_predicate_ok": on_predicate_ok,
        "total_attempts": len(attempts),
        "step_count": total_step_count,
        "failure_type": None if overall_success else final_failure_subtype,
        "attempts": attempts,
    }
