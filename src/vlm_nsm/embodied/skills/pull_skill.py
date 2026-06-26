"""PullSkill — grasp handle and pull along Y axis.

Used for drawer-opening tasks. After grasping the handle, the EEF moves
backward along Y, pulling the handle with it.
"""

from vlm_nsm.embodied.skills.base_skill import (
    BaseSkill, SkillParams, SkillResult, _dist,
)


class PullSkill(BaseSkill):
    """Grasp handle, then pull it backward along Y axis.

    Target: handle_xy + (handle_y + pull_distance, handle_z).
    Completion: EEF Y position reaches target Y within reach_threshold.
    """

    name = "pull"

    def get_target(self, env, params: SkillParams) -> dict:
        handle = env.get_object_pose(params.target_object)
        return {
            "x": handle["x"],
            "y": handle["y"] + params.approach_height,  # pull_distance stored in approach_height
            "z": handle["z"],
        }

    def check_completion(self, env, eef: dict, target: dict,
                         params: SkillParams) -> bool:
        return _dist(eef, target) < params.reach_threshold
