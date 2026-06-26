"""Reach skill — move EEF to a target position above an object."""

from __future__ import annotations

from vlm_nsm.embodied.skills.base_skill import (
    BaseSkill, SkillParams, SkillResult, _dist,
)


class ReachSkill(BaseSkill):
    """Move EEF to a position above an object (approach).

    Target: object_xy + (object_z + approach_height).
    Completion: distance to target < 0.01 (tighter than default 0.02 for grasp reliability).
    """

    name = "reach"

    def get_target(self, env, params: SkillParams) -> dict:
        obj = env.get_object_pose(params.target_object)
        return {
            "x": obj["x"], "y": obj["y"],
            "z": obj["z"] + params.approach_height,
        }

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return _dist(eef, target) < 0.01  # 10mm, not 20mm


class ReachPlaceSkill(BaseSkill):
    """Move EEF to a position above a target location (transport).

    Target: target_xy + (target_z + approach_height).
    """

    name = "reach_place"

    def get_target(self, env, params: SkillParams) -> dict:
        # Handle peg coordinates (Hanoi): "peg:0.12,-0.08"
        if isinstance(params.target_location, str) and params.target_location.startswith("peg:"):
            parts = params.target_location[4:].split(",")
            px, py = float(parts[0]), float(parts[1])
            return {"x": px, "y": py, "z": 0.82 + params.approach_height}

        tgt = env.get_target_pose(params.target_location)
        return {
            "x": tgt["x"], "y": tgt["y"],
            "z": tgt["z"] + params.approach_height,
        }

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return _dist(eef, target) < params.reach_threshold


class RetreatSkill(BaseSkill):
    """Move EEF up and away from the workspace."""

    name = "retreat"

    def get_target(self, env, params: SkillParams) -> dict:
        eef = env.get_eef_pose()
        return {"x": eef["x"], "y": eef["y"], "z": eef["z"] + 0.15}

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return _dist(eef, target) < params.reach_threshold
