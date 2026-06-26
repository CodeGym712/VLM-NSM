"""Grasp skill — close gripper on object using grasp_state_machine."""

from __future__ import annotations

from vlm_nsm.embodied.skills.base_skill import (
    BaseSkill, SkillParams, SkillResult,
)
from vlm_nsm.embodied.grasp_state_machine import execute_grasp, GraspConfig


class GraspSkill(BaseSkill):
    """Grasp an object using closed-loop grasp state machine.

    Delegates to grasp_state_machine.execute_grasp for adaptive grasping
    with retry, micro-align, and contact detection.
    """

    name = "grasp"

    def get_target(self, env, params: SkillParams) -> dict:
        obj = env.get_object_pose(params.target_object)
        return {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.005}

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        grip = env.get_gripper_state()
        return grip.get("has_object", False)

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        """Use grasp_state_machine instead of PID loop for grasping."""
        cfg = GraspConfig(
            kp=0.05, max_delta=0.01, descend_offset=0.005,
            close_steps=120, max_retries=3,
            holding_check_enabled=True,
            micro_align_enabled=True,
            micro_xy_tol=0.005, micro_max_delta=0.006,
            micro_align_max_steps=60,
        )
        result = execute_grasp(env, params.target_object, config=cfg,
                               skip_approach=True, skip_lift=True)
        if result["success"]:
            return SkillResult(True, result.get("total_step_count", 0))
        return SkillResult(
            False, result.get("total_step_count", 0),
            result.get("failure_type", "grasp_failed"),
        )


class ReleasePlaceSkill(BaseSkill):
    """Release object using place_state_machine.

    Delegates to place_state_machine.execute_place which handles
    preplace, descend, release, hold, retreat, and verify.
    """

    name = "release_place"

    def get_target(self, env, params: SkillParams) -> dict:
        tgt = env.get_target_pose(params.target_location)
        return {"x": tgt["x"], "y": tgt["y"], "z": tgt["z"] + 0.15}

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        from vlm_nsm.core.physical_predicate_extractor import extract_on_predicates
        all_poses = env.get_all_object_poses()
        on_preds = extract_on_predicates(
            all_poses, dict(all_poses),
            {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05},
        )
        return any(
            len(p) == 3 and p[1] == params.target_object
            and p[2] == params.target_location
            for p in on_preds
        )

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        """Use place_state_machine for composite place."""
        from vlm_nsm.embodied.place_state_machine import execute_place, PlaceConfig
        cfg = PlaceConfig(kp=0.05, max_delta=0.02, max_retries=2,
                          mode="simple")
        result = execute_place(
            env, params.target_object, params.target_location,
            config=cfg, skip_preplace=False,
        )
        if result["success"]:
            return SkillResult(True, result.get("step_count", 0))
        return SkillResult(
            False, result.get("step_count", 0),
            result.get("failure_type", "place_failed"),
        )
