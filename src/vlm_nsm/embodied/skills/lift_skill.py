"""Lift skill — raise EEF after grasping an object."""

from __future__ import annotations

from vlm_nsm.embodied.skills.base_skill import (
    BaseSkill, SkillParams, SkillResult, _dist,
)


class LiftSkill(BaseSkill):
    """Lift a grasped object vertically.

    Target: object_xy + (object_z + lift_height).
    Completion: distance to target < reach_threshold OR object lifted.
    """

    name = "lift"

    def __init__(self, lift_height: float = 0.05,
                 lift_threshold: float = 0.005):
        self.lift_height = lift_height
        self.lift_threshold = lift_threshold

    def get_target(self, env, params: SkillParams) -> dict:
        obj = env.get_object_pose(params.target_object)
        return {
            "x": obj["x"], "y": obj["y"],
            "z": obj["z"] + self.lift_height + 0.10,
        }

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        if _dist(eef, target) < params.reach_threshold:
            return True
        return False

    def execute_with_pid(self, env, pid_ctrl, params: SkillParams) -> SkillResult:
        # Record initial object Z to verify lift actually happened
        try:
            obj = env.get_object_pose(params.target_object)
            self._obj_z_start = obj.get("z", 0.0)
        except (ValueError, AttributeError):
            self._obj_z_start = 0.0
        result = super().execute_with_pid(env, pid_ctrl, params)
        # Verify object was actually lifted (not just EEF moved up)
        if result.success:
            try:
                obj = env.get_object_pose(params.target_object)
                obj_z_delta = obj.get("z", 0.0) - self._obj_z_start
                if obj_z_delta < 0.02:  # object didn't lift significantly (at least 20mm expected)
                    return SkillResult(False, result.steps, "slip")
            except (ValueError, AttributeError):
                pass
        return result
