"""Stage 19-R1: Minimal real robosuite rollout smoke test.

Runs symbolic goals through the execution pipeline:
goal_json -> gate -> alignment -> target_stack -> planner -> skill_executor -> robosuite.

No language parser, Qwen, RGB detector, or stereo. Controller/planner validation only.

CLI:
  MUJOCO_GL=egl python3 -m vlm_nsm.experiments.stage19_run_symbolic_rollouts \
    --rollout_suite generated/eval_inputs/stage_19/symbolic_goal_rollout_smoke_12.json \
    --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \
    --checkpoint generated/results/stage_7B-4/dagger_reach_bc/best.pt \
    --output_summary generated/results/stage_19/stage19_r1_rollout_smoke_summary.json \
    --output_episodes generated/results/stage_19/stage19_r1_rollout_smoke_episodes.jsonl \
    --abort_on_unsafe_goal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from vlm_nsm.core.alignment import AlignmentLayer
from vlm_nsm.core.data_types import GoalParseResult
from vlm_nsm.core.planner_adapter import PlannerAdapter
from vlm_nsm.core.utils import ensure_dir
from vlm_nsm.planning.stacking_utils import target_stack_from_predicates
from vlm_nsm.skills.skill_plan_from_symbolic import skill_plan_from_symbolic
from vlm_nsm.skills.skill_types import SkillCommand
from vlm_nsm.learning.robosuite_schema import ACTION_DIM, OBS_DIM
from vlm_nsm.data.stage20_phase_labeling import phase_names, relabel_composite_place_phases

VALID_PREDICATES = ["on", "clear", "free-gripper", "grasped", "smaller"]
TYPE_TABLE = {"cube1": "block", "cube2": "block", "table": "table"}
PREDICATE_SIGNATURE = {
    "on": [["block", "block"], ["block", "table"]],
    "clear": [["block"]],
    "free-gripper": [[]],
    "grasped": [["block"]],
    "smaller": [["block", "block"], ["block", "table"]],
}

_PHASE_TO_SKILL_ID = {
    "reach_pick": 0, "grasp": 1, "lift": 2,
    "reach_place": 3, "release": 4, "retreat": 5,
}

FAILURE_LABELS = [
    "none", "rejected_by_gate", "unsafe_goal_leak",
    "goal_resolution_failure", "planner_failure", "env_reset_failure",
    "grasp_failure", "lift_failure", "place_failure",
    "stack_unstable", "wrong_object_grasped", "wrong_target_object",
    "timeout", "controller_runtime_error", "goal_checker_mismatch", "unknown",
]

SCRIPTED_PROFILE_CHOICES = [
    "default",
    "cube2_grasp_lower",
    "cube2_place_higher",
    "cube2_robust",
]

PROFILE_TARGET_OBJECTS = {"cube2", "green_cube"}

PROFILE_OFFSETS = {
    "default": {
        "grasp_z_offset": 0.0,
        "place_z_offset": 0.0,
        "release_z_offset": 0.0,
        "stabilization_steps": 0,
    },
    "cube2_grasp_lower": {
        "grasp_z_offset": -0.003,
        "place_z_offset": 0.0,
        "release_z_offset": 0.0,
        "stabilization_steps": 0,
    },
    "cube2_place_higher": {
        "grasp_z_offset": 0.0,
        "place_z_offset": 0.010,
        "release_z_offset": 0.005,
        "stabilization_steps": 10,
    },
    "cube2_robust": {
        "grasp_z_offset": -0.003,
        "place_z_offset": 0.010,
        "release_z_offset": 0.005,
        "stabilization_steps": 10,
    },
}


# ---------------------------------------------------------------------------
# Static goal parser — bypasses NL parsing, returns pre-set goal_json
# ---------------------------------------------------------------------------

class StaticGoalParser:
    """Parser that returns a pre-set goal_json, bypassing NL parsing."""

    def __init__(self, goal_json: dict):
        self._goal_json = goal_json

    def parse(self, **kwargs) -> GoalParseResult:
        return GoalParseResult(success=True, goal_json=self._goal_json, raw_output="static")


# ---------------------------------------------------------------------------
# Gate validation (same logic as dry-run executor)
# ---------------------------------------------------------------------------

def check_gate(goal_json: dict, available_objects: list[str]) -> dict:
    """Validate a goal_json through the execution gate.

    Returns {"passed": bool, "reason": str}.
    """
    status = goal_json.get("status", "ok")
    if status == "ungroundable":
        return {"passed": False, "reason": f"parser status is 'ungroundable': {goal_json.get('reason', 'no reason')}"}

    alignment = AlignmentLayer()
    align_result = alignment.align_and_validate(
        goal_json=goal_json,
        valid_objects=available_objects + ["table"],
        valid_predicates=VALID_PREDICATES,
        type_table={k: v for k, v in TYPE_TABLE.items() if k in available_objects or k == "table"},
        predicate_signature=PREDICATE_SIGNATURE,
        task_type=goal_json.get("task_type", "stacking"),
        n_blocks=len(available_objects),
    )
    if not align_result.success:
        return {"passed": False, "reason": f"alignment failed: {align_result.error}"}

    on_predicates = [
        (p.name, p.args[0], p.args[1])
        for p in align_result.goal_predicates
        if p.name == "on" and len(p.args) == 2
    ]
    try:
        target_stack = target_stack_from_predicates(on_predicates, available_objects)
    except ValueError as e:
        return {"passed": False, "reason": f"target_stack derivation failed: {e}"}

    return {"passed": True, "reason": "ok", "target_stack": target_stack, "aligned_predicates": align_result.goal_predicates}


# ---------------------------------------------------------------------------
# Build goal predicates from target stack
# ---------------------------------------------------------------------------

def build_stacking_goal(target_stack: list[str]) -> list[list[str]]:
    """Convert bottom-to-top target stack into physical goal predicates."""
    goals: list[list[str]] = []
    for i in range(len(target_stack) - 1):
        top = target_stack[i + 1]
        bottom = target_stack[i]
        goals.append(["on", top, bottom])
    return goals


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _trajectory_length(traj_data: dict[str, np.ndarray]) -> int:
    for key in ("action", "obs", "filtered_obs", "phase_id", "step_id"):
        if key in traj_data:
            return int(len(traj_data[key]))
    return 0


def add_stage20_trajectory_fields(
    traj_data: dict[str, np.ndarray],
    *,
    episode_index: int,
    episode_entry: dict,
    result: dict,
    scripted_profile: str,
) -> dict[str, np.ndarray]:
    """Add Stage 20 metadata fields to a per-step trajectory dict."""
    output = dict(traj_data)
    steps = _trajectory_length(output)
    top_object = result.get("resolved_top") or episode_entry.get("top_object") or ""
    bottom_object = result.get("resolved_bottom") or episode_entry.get("bottom_object") or ""
    direction = episode_entry.get("direction") or (
        f"{top_object}_to_{bottom_object}" if top_object and bottom_object else "unknown"
    )
    phase_ids = np.asarray(output.get("phase_id", np.full((steps,), -1, dtype=np.int32)), dtype=np.int32)
    output["phase_id_raw"] = phase_ids.copy()
    output["phase_name_raw"] = phase_names(phase_ids)
    if "action" in output and steps:
        try:
            relabelled_phase_ids, relabel_stats = relabel_composite_place_phases(phase_ids, output["action"])
            phase_ids = relabelled_phase_ids.astype(np.int32)
        except ValueError:
            relabel_stats = None
    else:
        relabel_stats = None
    output["phase_id"] = phase_ids
    output["phase_name"] = phase_names(phase_ids)
    output["phase_label_patch_applied"] = np.full(
        (steps,),
        bool(relabel_stats.applied if relabel_stats is not None else False),
        dtype=bool,
    )
    output["phase_label_changed"] = (phase_ids != np.asarray(output["phase_id_raw"], dtype=np.int32))
    output["phase_label_patch_version"] = np.array(["stage20_k_v1"] * steps)
    output["direction"] = np.array([direction] * steps)
    output["goal_id"] = np.array([episode_entry.get("goal_id", "")] * steps)
    output["top_object"] = np.array([top_object] * steps)
    output["bottom_object"] = np.array([bottom_object] * steps)
    output["teacher"] = np.array(["scripted"] * steps)
    output["teacher_profile"] = np.array([scripted_profile] * steps)
    output["scripted_profile"] = np.array([scripted_profile] * steps)
    output["scripted_grasp_z_offset"] = np.full((steps,), float(result.get("grasp_z_offset", 0.0)), dtype=np.float32)
    output["scripted_place_z_offset"] = np.full((steps,), float(result.get("place_z_offset", 0.0)), dtype=np.float32)
    output["scripted_release_z_offset"] = np.full((steps,), float(result.get("release_z_offset", 0.0)), dtype=np.float32)
    output["scripted_stabilization_steps"] = np.full(
        (steps,),
        int(result.get("stabilization_steps", 0) or 0),
        dtype=np.int32,
    )
    output["stage20_episode_index"] = np.full((steps,), episode_index, dtype=np.int32)
    output["stage20_success"] = np.full((steps,), bool(result.get("task_success", False)), dtype=bool)
    return output


def save_stage20_trajectory(
    output_dir: Path,
    episode_entry: dict,
    result: dict,
    recorder,
    episode_index: int,
    scripted_profile: str,
) -> str | None:
    """Save one recorded trajectory and return relative path."""
    traj_data = recorder.pop_last_episode_data() if recorder is not None else None
    if traj_data is None:
        return None
    rel_path = Path("trajectories") / f"{episode_entry['episode_id']}.npz"
    abs_path = output_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    labelled = add_stage20_trajectory_fields(
        traj_data,
        episode_index=episode_index,
        episode_entry=episode_entry,
        result=result,
        scripted_profile=scripted_profile,
    )
    np.savez_compressed(abs_path, **labelled)
    return str(rel_path)


def build_stage20_manifest_record(
    episode_entry: dict,
    result: dict,
    trajectory_path: str | None,
    scripted_profile: str,
) -> dict:
    top_object = result.get("resolved_top") or episode_entry.get("top_object")
    bottom_object = result.get("resolved_bottom") or episode_entry.get("bottom_object")
    direction = episode_entry.get("direction") or (
        f"{top_object}_to_{bottom_object}" if top_object and bottom_object else "unknown"
    )
    return _json_safe({
        "episode_id": episode_entry.get("episode_id"),
        "skill_name": "pick_place",
        "object_id": top_object,
        "target_object_id": bottom_object,
        "relation": "on",
        "source": "table",
        "direction": direction,
        "goal_id": episode_entry.get("goal_id"),
        "top_object": top_object,
        "bottom_object": bottom_object,
        "top_role": "object_to_pick",
        "bottom_role": "support_object",
        "teacher": "scripted",
        "teacher_profile": scripted_profile,
        "scripted_profile": scripted_profile,
        "scripted_offsets": {
            "grasp_z_offset": result.get("grasp_z_offset", 0.0),
            "place_z_offset": result.get("place_z_offset", 0.0),
            "release_z_offset": result.get("release_z_offset", 0.0),
            "stabilization_steps": result.get("stabilization_steps", 0),
        },
        "success": bool(result.get("task_success", False)),
        "num_steps": int(result.get("steps", 0) or 0),
        "failure_type": None if result.get("failure_reason") == "none" else result.get("failure_reason"),
        "failure_phase": result.get("skill_failure_phase"),
        "trajectory_path": trajectory_path,
        "seed": episode_entry.get("seed", 0),
        "initial_scene": {
            "object_poses": {
                "cube1": result.get("initial_cube1_pose"),
                "cube2": result.get("initial_cube2_pose"),
            },
        },
        "final_scene": {
            "object_poses": {
                "cube1": result.get("final_cube1_pose"),
                "cube2": result.get("final_cube2_pose"),
            },
        },
    })


def resolve_scripted_profile(profile: str, top_object: str | None) -> dict:
    """Resolve runtime-only scripted profile parameters for one episode.

    Non-default profiles only affect cube2/green_cube as the top object.
    Defaults are all zero so the canonical execution config remains unchanged.
    """
    if profile not in PROFILE_OFFSETS:
        raise ValueError(f"Unknown scripted_profile: {profile}")

    applies = profile != "default" and top_object in PROFILE_TARGET_OBJECTS
    offsets = dict(PROFILE_OFFSETS[profile] if applies else PROFILE_OFFSETS["default"])
    return {
        "scripted_profile": profile,
        "grasp_z_offset": offsets["grasp_z_offset"],
        "place_z_offset": offsets["place_z_offset"],
        "release_z_offset": offsets["release_z_offset"],
        "stabilization_steps": offsets["stabilization_steps"],
        "profile_applied_to_object": top_object if applies else None,
    }


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_single_episode(
    episode_entry: dict,
    available_objects: list[str],
    env,
    planner,
    policy_override,
    override_phases: set[int],
    exec_config: dict,
    collect_diagnostics: bool = False,
    scripted_profile: str = "default",
    recorder=None,
) -> dict:
    """Run a single episode and return the per-episode result dict."""
    episode_id = episode_entry["episode_id"]
    goal_id = episode_entry["goal_id"]
    parser_output = episode_entry["parser_output"]
    expected_execution = episode_entry.get("expected_execution", "execute")
    seed = episode_entry.get("seed", 0)

    result: dict = {
        "episode_id": episode_id,
        "goal_id": goal_id,
        "seed": seed,
        "parser_status": parser_output.get("status", "ok"),
        "gate_decision": "reject",
        "resolved_top": None,
        "resolved_bottom": None,
        "target_stack_valid": False,
        "env_reset_ok": False,
        "rollout_started": False,
        "task_success": False,
        "stack_success": False,
        "grasp_success": None,
        "place_success": None,
        "final_top_pose": None,
        "final_bottom_pose": None,
        "final_pose_error": None,
        "steps": 0,
        "failure_reason": "none",
        "exception_type": None,
        "exception_message": None,
        "unsafe_execution": False,
        "scripted_profile": scripted_profile,
        "grasp_z_offset": 0.0,
        "place_z_offset": 0.0,
        "release_z_offset": 0.0,
        "stabilization_steps": 0,
        "profile_applied_to_object": None,
    }

    # Diagnostics fields
    if collect_diagnostics:
        result.update({
            "placement_id": episode_entry.get("placement_id"),
            "top_body_name": None,
            "bottom_body_name": None,
            "cube1_body_name": None,
            "cube2_body_name": None,
            "cube1_color": "red",
            "cube2_color": "green",
            "initial_top_pose": None,
            "initial_bottom_pose": None,
            "initial_cube1_pose": None,
            "initial_cube2_pose": None,
            "final_cube1_pose": None,
            "final_cube2_pose": None,
            "target_place_pose": None,
            "planned_grasp_pose": None,
            "planned_place_pose": None,
            "actual_grasped_object": None,
            "min_gripper_distance_to_top": None,
            "min_gripper_distance_to_bottom": None,
            "top_displacement": None,
            "bottom_displacement": None,
            "initial_top_bottom_distance_xy": None,
            "initial_relative_xy_top_minus_bottom": None,
            "top_region": None,
            "bottom_region": None,
            "phase_transition_trace": None,
            "skill_failure_phase": None,
            "execution_mode": "hybrid" if policy_override is not None else "scripted_only",
        })

    # Gate check
    gate = check_gate(parser_output, available_objects)
    if not gate["passed"]:
        result["gate_decision"] = "reject"
        result["failure_reason"] = "rejected_by_gate"
        if expected_execution == "execute":
            result["exception_message"] = f"Gate rejected valid goal: {gate['reason']}"
        return result

    result["gate_decision"] = "execute"

    # Unsafe goal detection
    if expected_execution == "reject":
        result["unsafe_execution"] = True
        result["failure_reason"] = "unsafe_goal_leak"
        result["exception_message"] = "UNSAFE: reject-expected goal passed gate"
        return result

    # Resolve top/bottom
    target_stack = gate["target_stack"]
    resolved_top = target_stack[-1] if target_stack else None
    resolved_bottom = target_stack[0] if len(target_stack) >= 2 else None
    result["resolved_top"] = resolved_top
    result["resolved_bottom"] = resolved_bottom
    result["target_stack_valid"] = True
    profile_config = resolve_scripted_profile(scripted_profile, resolved_top)
    result.update(profile_config)

    # Collect diagnostics: body name mapping
    if collect_diagnostics:
        try:
            result["cube1_body_name"] = env.get_object_body_name("cube1") if hasattr(env, "get_object_body_name") else None
            result["cube2_body_name"] = env.get_object_body_name("cube2") if hasattr(env, "get_object_body_name") else None
        except Exception:
            pass
        if resolved_top:
            try:
                result["top_body_name"] = env.get_object_body_name(resolved_top) if hasattr(env, "get_object_body_name") else None
            except Exception:
                pass
        if resolved_bottom:
            try:
                result["bottom_body_name"] = env.get_object_body_name(resolved_bottom) if hasattr(env, "get_object_body_name") else None
            except Exception:
                pass

    # Collect initial poses (before reset, env just created)
    if collect_diagnostics:
        try:
            result["initial_cube1_pose"] = env.get_object_pose("cube1")
        except Exception:
            pass
        try:
            result["initial_cube2_pose"] = env.get_object_pose("cube2")
        except Exception:
            pass
        if resolved_top:
            try:
                result["initial_top_pose"] = env.get_object_pose(resolved_top)
            except Exception:
                pass
        if resolved_bottom:
            try:
                result["initial_bottom_pose"] = env.get_object_pose(resolved_bottom)
            except Exception:
                pass

    # Env reset
    try:
        env.reset()
        result["env_reset_ok"] = True
    except Exception as e:
        result["failure_reason"] = "env_reset_failure"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        return result

    # Collect post-reset poses
    if collect_diagnostics:
        try:
            result["initial_cube1_pose"] = env.get_object_pose("cube1")
        except Exception:
            pass
        try:
            result["initial_cube2_pose"] = env.get_object_pose("cube2")
        except Exception:
            pass
        if resolved_top:
            try:
                result["initial_top_pose"] = env.get_object_pose(resolved_top)
            except Exception:
                pass
        if resolved_bottom:
            try:
                result["initial_bottom_pose"] = env.get_object_pose(resolved_bottom)
            except Exception:
                pass

    # Build goal predicates
    goal_predicates = build_stacking_goal(target_stack)

    # Plan
    try:
        plan_result = planner.solve(cubes=available_objects, target_stack=target_stack)
        if not plan_result.success:
            result["failure_reason"] = "planner_failure"
            result["exception_message"] = plan_result.error
            return result
    except Exception as e:
        result["failure_reason"] = "planner_failure"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        return result

    # Skill plan
    try:
        skill_plan = skill_plan_from_symbolic(plan_result.actions)
    except Exception as e:
        result["failure_reason"] = "planner_failure"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = f"skill_plan: {e}"
        return result

    # Execute
    result["rollout_started"] = True
    try:
        from vlm_nsm.execution.scripted_skill_executor import ScriptedSkillExecutor

        # Filter exec_config to only include valid kwargs for _run_robosuite_episode
        valid_episode_keys = {
            "kp", "max_delta", "use_place_state_machine", "horizon_aware",
            "place_retry", "grasp_close_steps", "grasp_max_retries",
            "place_max_retries", "place_mode", "skip_retreat_if_goal_satisfied",
            "holding_check_enabled", "lift_early_exit_enabled",
            "close_steps_min", "close_steps_max", "close_stabilize_steps",
            "release_steps_min", "release_steps_max", "post_release_stable_steps",
            "close_mode", "close_steps_fast", "fallback_extend_close_if_not_holding",
            "reach_threshold", "lift_height", "release_open_steps",
            "terminate_after_safe_release",
            "grasp_z_offset", "place_z_offset", "release_z_offset",
            "stabilization_steps",
        }
        filtered_config = {k: v for k, v in exec_config.items() if k in valid_episode_keys}
        filtered_config.update({
            "grasp_z_offset": profile_config["grasp_z_offset"],
            "place_z_offset": profile_config["place_z_offset"],
            "release_z_offset": profile_config["release_z_offset"],
            "stabilization_steps": profile_config["stabilization_steps"],
        })

        executor = ScriptedSkillExecutor(controller_config=filtered_config)
        all_skill_results = []
        overall_success = True

        for cmd in skill_plan.commands:
            skill_result = executor.execute(
                skill_command=cmd,
                env=env,
                context={
                    "cubes": available_objects,
                    "goal_predicates": goal_predicates,
                    "policy_override": policy_override,
                    "override_phases": override_phases,
                    "recorder": recorder,
                },
            )
            all_skill_results.append(skill_result)
            result["steps"] += skill_result.steps
            if not skill_result.success:
                overall_success = False
                raw = skill_result.metrics.get("raw_result", {})
                ft = raw.get("failure_type") or "unknown"
                result["failure_reason"] = _map_failure_type(ft)
                if collect_diagnostics:
                    result["skill_failure_phase"] = raw.get("failure_phase") or raw.get("last_phase")
                    result["phase_transition_trace"] = raw.get("phase_trace")
                break

        result["task_success"] = overall_success
        result["stack_success"] = overall_success

        if all_skill_results:
            last_raw = all_skill_results[-1].metrics.get("raw_result", {})
            result["grasp_success"] = _check_grasp_success(last_raw)
            result["place_success"] = _check_place_success(last_raw)
            if collect_diagnostics and not result["phase_transition_trace"]:
                result["phase_transition_trace"] = last_raw.get("phase_trace")
            if collect_diagnostics:
                result["actual_grasped_object"] = last_raw.get("grasped_object")

    except Exception as e:
        result["failure_reason"] = "controller_runtime_error"
        result["exception_type"] = type(e).__name__
        result["exception_message"] = str(e)
        result["task_success"] = False
        result["stack_success"] = False

    # Read final object poses
    try:
        top_pose = env.get_object_pose(resolved_top) if resolved_top else None
        bottom_pose = env.get_object_pose(resolved_bottom) if resolved_bottom else None
        result["final_top_pose"] = top_pose
        result["final_bottom_pose"] = bottom_pose
        if top_pose and bottom_pose:
            dx = top_pose.get("x", 0) - bottom_pose.get("x", 0)
            dy = top_pose.get("y", 0) - bottom_pose.get("y", 0)
            dz = top_pose.get("z", 0) - bottom_pose.get("z", 0)
            result["final_pose_error"] = float(np.sqrt(dx*dx + dy*dy + dz*dz))

        # Compute displacements and placement diagnostics
        if collect_diagnostics:
            init_top = result.get("initial_top_pose")
            init_bottom = result.get("initial_bottom_pose")
            if init_top and top_pose:
                result["top_displacement"] = float(np.sqrt(
                    (top_pose.get("x", 0) - init_top.get("x", 0))**2 +
                    (top_pose.get("y", 0) - init_top.get("y", 0))**2 +
                    (top_pose.get("z", 0) - init_top.get("z", 0))**2
                ))
            if init_bottom and bottom_pose:
                result["bottom_displacement"] = float(np.sqrt(
                    (bottom_pose.get("x", 0) - init_bottom.get("x", 0))**2 +
                    (bottom_pose.get("y", 0) - init_bottom.get("y", 0))**2 +
                    (bottom_pose.get("z", 0) - init_bottom.get("z", 0))**2
                ))
            # Final poses for all cubes
            try:
                result["final_cube1_pose"] = env.get_object_pose("cube1")
            except Exception:
                pass
            try:
                result["final_cube2_pose"] = env.get_object_pose("cube2")
            except Exception:
                pass
            # Initial placement geometry
            if init_top and init_bottom:
                result["initial_top_bottom_distance_xy"] = float(np.sqrt(
                    (init_top.get("x", 0) - init_bottom.get("x", 0))**2 +
                    (init_top.get("y", 0) - init_bottom.get("y", 0))**2
                ))
                result["initial_relative_xy_top_minus_bottom"] = {
                    "dx": float(init_top.get("x", 0) - init_bottom.get("x", 0)),
                    "dy": float(init_top.get("y", 0) - init_bottom.get("y", 0)),
                }
            # Table region discretization
            if init_top:
                result["top_region"] = _compute_table_region(init_top)
            if init_bottom:
                result["bottom_region"] = _compute_table_region(init_bottom)
    except Exception:
        pass

    if result["task_success"]:
        result["failure_reason"] = "none"

    return result


def _map_failure_type(ft: str | None) -> str:
    if not ft:
        return "unknown"
    ft_lower = ft.lower()
    # Grasp sub-type recognition: grasp state machine returns
    # "slip", "prereach", "nocontact" as failure_subtype values
    _grasp_subtypes = {"slip", "prereach", "nocontact", "lost", "grasp_failed"}
    if ft_lower in _grasp_subtypes or "grasp" in ft_lower:
        return "grasp_failure"
    if "lift" in ft_lower:
        return "lift_failure"
    if "place" in ft_lower:
        return "place_failure"
    if "timeout" in ft_lower:
        return "timeout"
    if "horizon" in ft_lower:
        return "timeout"
    if "stack" in ft_lower and ("unstable" in ft_lower or "collapse" in ft_lower):
        return "stack_unstable"
    # Catch-all for unrecognized failure types: prefix with phase if known
    return "unknown"


def _check_grasp_success(raw: dict) -> bool | None:
    phase_trace = raw.get("phase_trace", {})
    if "grasp" in phase_trace:
        return True
    return None


def _check_place_success(raw: dict) -> bool | None:
    phase_trace = raw.get("phase_trace", {})
    if "release" in phase_trace:
        return True
    return None


def _compute_table_region(pose: dict, threshold: float = 0.08) -> str:
    """Discretize a table-top XY position into a named region.

    Table is 0.8×0.8m centered at origin. Regions:
      center, front, back, left, right, front-left, front-right, back-left, back-right
    """
    x = pose.get("x", 0)
    y = pose.get("y", 0)
    parts: list[str] = []
    if y < -threshold:
        parts.append("front")
    elif y > threshold:
        parts.append("back")
    if x < -threshold:
        parts.append("left")
    elif x > threshold:
        parts.append("right")
    if not parts:
        return "center"
    return "-".join(parts)


def _parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 19 Rollout Smoke Test")
    parser.add_argument("--rollout_suite", required=True,
                        help="Path to rollout suite JSON")
    parser.add_argument("--execution_config", required=True,
                        help="Path to execution config JSON")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to BC checkpoint (.pt). Not required if --scripted_only.")
    parser.add_argument("--output_summary", required=True,
                        help="Path for summary JSON output")
    parser.add_argument("--output_episodes", required=True,
                        help="Path for per-episode JSONL output")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps per episode")
    parser.add_argument("--render", default="false",
                        help="Render mode (none, human, offscreen)")
    parser.add_argument("--abort_on_unsafe_goal", nargs="?", const=True, default=False, type=_parse_bool,
                        help="Abort immediately if an unsafe goal passes the gate")
    parser.add_argument("--available_objects", nargs="+", default=["cube1", "cube2"],
                        help="Available object IDs")
    parser.add_argument("--scripted_only", nargs="?", const=True, default=False, type=_parse_bool,
                        help="Run purely scripted (no BC policy). Skip checkpoint loading.")
    parser.add_argument("--bc_fallback_to_scripted", nargs="?", const=True, default=False, type=_parse_bool,
                        help="When BC reach_pick times out, retry episode with scripted-only reach_pick.")
    parser.add_argument("--save_diagnostics", nargs="?", const=True, default=False, type=_parse_bool,
                        help="Save detailed per-episode diagnostics")
    parser.add_argument("--diagnostics_output",
                        default="generated/results/stage_19/stage19_d1_rollout_diagnostics.jsonl",
                        help="Path for diagnostics JSONL output")
    parser.add_argument("--scripted_profile", default="default",
                        choices=SCRIPTED_PROFILE_CHOICES,
                        help="Runtime-only scripted control profile. Default preserves current behavior.")
    parser.add_argument("--save_trajectories", nargs="?", const=True, default=False, type=_parse_bool,
                        help="Save Stage 20-compatible per-step trajectories")
    parser.add_argument("--trajectory_output_dir", default=None,
                        help="Output dataset directory for saved trajectories")
    args = parser.parse_args()

    # Parse render
    render_mode = args.render if args.render in ("none", "human", "offscreen") else "none"

    # Load suite
    suite_path = Path(args.rollout_suite)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    episodes = suite.get("episodes", [])
    available_objects = args.available_objects

    print(f"Rollout suite: {len(episodes)} episodes from {suite_path}")

    # Load execution config
    config_path = Path(args.execution_config)
    exec_config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Execution config: {config_path}")
    print(f"Scripted profile: {args.scripted_profile}")

    trajectory_output_dir = Path(args.trajectory_output_dir) if args.trajectory_output_dir else None
    trajectory_manifest_records: list[dict] = []
    trajectories_saved = 0
    if args.save_trajectories:
        if trajectory_output_dir is None:
            print("ERROR: --trajectory_output_dir required with --save_trajectories")
            sys.exit(1)
        (trajectory_output_dir / "trajectories").mkdir(parents=True, exist_ok=True)

    # Load BC checkpoint (read-only) — skip if scripted_only
    bc_policy = None
    override_phases: set[int] = set()
    if args.scripted_only:
        print("Scripted-only mode: no BC policy override")
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint required unless --scripted_only")
            sys.exit(1)
        print(f"Loading checkpoint: {args.checkpoint}")
        from vlm_nsm.learning.eval_bc import load_checkpoint
        device = "cpu"
        model, aux = load_checkpoint(args.checkpoint, device)
        print(f"  obs_dim={aux.get('obs_dim')}, action_dim={aux.get('action_dim')}")

        # Load action norm
        norm_path = Path("generated/results/stage_7B-1/action_norm_stats.json")
        if norm_path.exists():
            stats = json.loads(norm_path.read_text(encoding="utf-8"))
            action_mean = np.array(stats["action_mean"], dtype=np.float32)
            action_std = np.array(stats["action_std"], dtype=np.float32)
            print(f"Loaded action norm from {norm_path}")
        else:
            action_mean = None
            action_std = None
            print("No action norm stats found — using raw BC output")

        # Build policy_override function
        from vlm_nsm.learning.eval_bc import _get_bc_action

        model_obs_dim = aux.get("obs_dim", OBS_DIM)
        if model_obs_dim > OBS_DIM:
            condition_on_phase = True
            phase_dim = model_obs_dim - OBS_DIM
        else:
            condition_on_phase = aux.get("condition_on_phase", False)
            phase_dim = 6
        if condition_on_phase:
            print(f"  condition_on_phase=True (obs_dim={model_obs_dim}, phase_dim={phase_dim}, inferred from shape)")

        def bc_policy(obs, phase_id):
            obs_in = obs
            if condition_on_phase:
                one_hot = np.zeros(phase_dim, dtype=np.float32)
                one_hot[phase_id] = 1.0
                obs_in = np.concatenate([obs, one_hot])
            return _get_bc_action(
                model, obs_in, device, action_mean, action_std,
                fixed_zero_dims=aux.get("fixed_zero_dims"),
            )

        override_phases = {_PHASE_TO_SKILL_ID["reach_pick"]}
        print(f"BC override phases: reach_pick (phase_id=0)")

    # Planner
    planner = PlannerAdapter("bfs")

    # Import here to avoid hard dependency at module load (for unit tests)
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    # Run episodes
    summary = {
        "suite_description": suite.get("description", ""),
        "execution_mode": "scripted_only" if args.scripted_only else "hybrid",
        "total_episodes": len(episodes),
        "executed_episodes": 0,
        "rejected_episodes": 0,
        "unsafe_goal_leak_count": 0,
        "env_reset_failures": 0,
        "controller_runtime_errors": 0,
        "task_success_count": 0,
        "task_success_rate": 0.0,
        "stack_success_count": 0,
        "stack_success_rate": 0.0,
        "per_goal_success": {},
        "per_failure_reason_counts": {},
        "fallback_count": 0,
        "fallback_success_count": 0,
        "scripted_profile": args.scripted_profile,
    }

    episode_lines: list[str] = []
    abort = False

    for i, ep in enumerate(episodes):
        episode_id = ep["episode_id"]
        goal_id = ep["goal_id"]
        print(f"\n[{i+1}/{len(episodes)}] {episode_id} (goal={goal_id})")

        # Create fresh env per episode
        env = RobosuiteEnvWrapper(
            env_name="Stack",
            robots="Panda",
            cube_names=available_objects,
            render_mode=render_mode,
        )
        recorder = None
        if args.save_trajectories:
            from vlm_nsm.embodied.demonstration_recorder import DemonstrationRecorder
            recorder = DemonstrationRecorder(
                output_dir=trajectory_output_dir / "trajectories",
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
            )

        t_start = time.time()
        result = run_single_episode(
            ep, available_objects, env, planner,
            bc_policy, override_phases, exec_config,
            collect_diagnostics=args.save_diagnostics,
            scripted_profile=args.scripted_profile,
            recorder=recorder,
        )
        elapsed = time.time() - t_start

        if args.save_trajectories and recorder is not None:
            traj_path = save_stage20_trajectory(
                trajectory_output_dir,
                ep,
                result,
                recorder,
                i,
                args.scripted_profile,
            )
            if traj_path:
                trajectories_saved += 1
            result["trajectory_path"] = traj_path
            trajectory_manifest_records.append(
                build_stage20_manifest_record(ep, result, traj_path, args.scripted_profile)
            )

        try:
            env.env.close()
        except Exception:
            pass

        # D4-3: BC-to-scripted fallback on timeout
        fallback_used = False
        if (args.bc_fallback_to_scripted
                and bc_policy is not None
                and result["failure_reason"] == "timeout"
                and result["rollout_started"]):
            fallback_used = True
            print(f"  ⏳ BC timed out — falling back to scripted reach_pick...")
            env2 = RobosuiteEnvWrapper(
                env_name="Stack",
                robots="Panda",
                cube_names=available_objects,
                render_mode=render_mode,
            )
            t2_start = time.time()
            fallback_result = run_single_episode(
                ep, available_objects, env2, planner,
                None, set(), exec_config,
                collect_diagnostics=args.save_diagnostics,
                scripted_profile=args.scripted_profile,
            )
            t2_elapsed = time.time() - t2_start
            elapsed += t2_elapsed
            try:
                env2.env.close()
            except Exception:
                pass
            # Merge fallback result
            result["fallback_used"] = True
            result["fallback_result"] = {
                "task_success": fallback_result["task_success"],
                "failure_reason": fallback_result["failure_reason"],
                "steps": fallback_result["steps"],
                "grasp_success": fallback_result.get("grasp_success"),
                "place_success": fallback_result.get("place_success"),
            }
            if fallback_result["task_success"]:
                result["task_success"] = True
                result["stack_success"] = True
                result["failure_reason"] = "none"
                result["steps"] = fallback_result["steps"]
                result["grasp_success"] = fallback_result.get("grasp_success")
                result["place_success"] = fallback_result.get("place_success")
                result["final_top_pose"] = fallback_result.get("final_top_pose")
                result["final_bottom_pose"] = fallback_result.get("final_bottom_pose")
                result["final_pose_error"] = fallback_result.get("final_pose_error")
                # Update diagnostics with fallback data
                if args.save_diagnostics:
                    for diag_key in ("phase_transition_trace", "skill_failure_phase",
                                     "top_body_name", "bottom_body_name",
                                     "initial_top_pose", "initial_bottom_pose",
                                     "final_cube1_pose", "final_cube2_pose",
                                     "top_displacement", "bottom_displacement"):
                        if diag_key in fallback_result:
                            result[diag_key] = fallback_result[diag_key]
                print(f"  ✅ Fallback succeeded: scripted reach_pick in {fallback_result['steps']} steps")
            else:
                print(f"  ❌ Fallback also failed: {fallback_result['failure_reason']}")
        else:
            result["fallback_used"] = False

        # Update summary
        if result["gate_decision"] == "reject":
            summary["rejected_episodes"] += 1
        else:
            summary["executed_episodes"] += 1

        if result["unsafe_execution"]:
            summary["unsafe_goal_leak_count"] += 1
            print(f"  ⚠️  UNSAFE GOAL LEAK: {episode_id}")
            if args.abort_on_unsafe_goal:
                print("  ABORTING due to unsafe goal leak.")
                abort = True

        if result["failure_reason"] == "env_reset_failure":
            summary["env_reset_failures"] += 1
        if result["failure_reason"] == "controller_runtime_error":
            summary["controller_runtime_errors"] += 1
        if result["task_success"]:
            summary["task_success_count"] += 1
            summary["stack_success_count"] += 1

        fr = result["failure_reason"]
        summary["per_failure_reason_counts"][fr] = summary["per_failure_reason_counts"].get(fr, 0) + 1

        # Track fallback
        if result.get("fallback_used"):
            summary["fallback_count"] += 1
            if result["task_success"]:
                summary["fallback_success_count"] += 1

        # Per-goal tracking
        if goal_id not in summary["per_goal_success"]:
            summary["per_goal_success"][goal_id] = {"total": 0, "success": 0}
        summary["per_goal_success"][goal_id]["total"] += 1
        if result["task_success"]:
            summary["per_goal_success"][goal_id]["success"] += 1

        status = "✅" if result["task_success"] else "❌"
        print(f"  {status} gate={result['gate_decision']}, "
              f"success={result['task_success']}, "
              f"steps={result['steps']}, "
              f"failure={result['failure_reason']}, "
              f"time={elapsed:.1f}s")

        # JSONL
        episode_lines.append(json.dumps(result, ensure_ascii=False))

        if abort:
            break

    # Finalize summary
    executed = summary["executed_episodes"]
    summary["task_success_rate"] = round(summary["task_success_count"] / executed, 4) if executed > 0 else 0.0
    summary["stack_success_rate"] = round(summary["stack_success_count"] / executed, 4) if executed > 0 else 0.0

    # Write outputs
    summary_path = Path(args.output_summary)
    ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")

    episodes_path = Path(args.output_episodes)
    ensure_dir(episodes_path.parent)
    episodes_path.write_text("\n".join(episode_lines) + "\n", encoding="utf-8")
    print(f"Episodes JSONL written to {episodes_path}")

    # Write diagnostics if requested
    if args.save_diagnostics:
        diagnostics_path = Path(args.diagnostics_output)
        ensure_dir(diagnostics_path.parent)
        diagnostics_path.write_text("\n".join(episode_lines) + "\n", encoding="utf-8")
        print(f"Diagnostics JSONL written to {diagnostics_path}")

    if args.save_trajectories and trajectory_output_dir is not None:
        manifest_path = trajectory_output_dir / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in trajectory_manifest_records) + "\n",
            encoding="utf-8",
        )
        dataset_summary = {
            "dataset_name": "teacher_demos_cube2_robust_v0_2",
            "teacher": "scripted",
            "teacher_profile": args.scripted_profile,
            "num_episodes": len(trajectory_manifest_records),
            "success_count": sum(1 for r in trajectory_manifest_records if r.get("success")),
            "trajectories_saved": trajectories_saved,
            "manifest_path": "manifest.jsonl",
            "trajectory_dir": "trajectories",
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "scripted_profile": args.scripted_profile,
            "source_rollout_suite": str(suite_path),
            "summary_path": str(summary_path),
        }
        dataset_summary["success_rate"] = (
            dataset_summary["success_count"] / max(dataset_summary["num_episodes"], 1)
        )
        (trajectory_output_dir / "summary.json").write_text(
            json.dumps(dataset_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (trajectory_output_dir / "README.md").write_text(
            "# Stage 20 Teacher Demos cube2_robust v0_2\n\n"
            "Generated by `stage19_run_symbolic_rollouts` with optional trajectory saving.\n\n"
            "Teacher: scripted-only + cube2_robust.\n",
            encoding="utf-8",
        )
        print(f"Trajectory manifest written to {manifest_path}")
        print(f"Trajectories saved: {trajectories_saved}")

    # Print summary
    print(f"\n=== Stage 19 Rollout Results ===")
    print(f"Total episodes:     {summary['total_episodes']}")
    if summary.get("fallback_count", 0) > 0:
        print(f"BC fallbacks used:  {summary['fallback_count']} (success: {summary['fallback_success_count']})")
    print(f"Executed:           {summary['executed_episodes']}")
    print(f"Rejected by gate:   {summary['rejected_episodes']}")
    print(f"Unsafe goal leaks:  {summary['unsafe_goal_leak_count']}")
    print(f"Env reset failures: {summary['env_reset_failures']}")
    print(f"Runtime errors:     {summary['controller_runtime_errors']}")
    print(f"Task success:       {summary['task_success_count']}/{executed} ({summary['task_success_rate']:.1%})")
    print(f"Stack success:      {summary['stack_success_count']}/{executed} ({summary['stack_success_rate']:.1%})")

    print(f"\nPer-goal success:")
    for gid, gs in summary["per_goal_success"].items():
        rate = gs["success"] / gs["total"] if gs["total"] > 0 else 0
        print(f"  {gid}: {gs['success']}/{gs['total']} ({rate:.1%})")

    print(f"\nFailure taxonomy:")
    for fr, count in sorted(summary["per_failure_reason_counts"].items()):
        print(f"  {fr}: {count}")

    # Determine pass/fail
    if summary["unsafe_goal_leak_count"] > 0:
        print("\n🔴 FAIL: Unsafe goal leak detected")
    elif executed < 10:
        print(f"\n🟡 PARTIAL_PASS: Only {executed} episodes executed (expected >= 10)")
    elif summary["task_success_rate"] < 0.60:
        print(f"\n🔴 FAIL: Task success rate {summary['task_success_rate']:.1%} < 60%")
    elif summary["task_success_rate"] < 0.80:
        print(f"\n🟡 PARTIAL_PASS: Task success rate {summary['task_success_rate']:.1%} (60-80%)")
    else:
        print(f"\n🟢 PASS: {summary['task_success_rate']:.1%} success rate, 0 unsafe leaks")


if __name__ == "__main__":
    main()
