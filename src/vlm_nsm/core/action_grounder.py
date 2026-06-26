from __future__ import annotations

from vlm_nsm.core.data_types import Action


class ActionGrounder:
    def __init__(self, env_wrapper, block_height: float = 0.04, grasp_offset: float = 0.08):
        self.env_wrapper = env_wrapper
        self.block_height = block_height
        self.grasp_offset = grasp_offset

    def ground_move(self, action: Action) -> dict:
        if action.name != "move" or len(action.args) != 3:
            raise ValueError("ground_move expects Action('move', [object, source, target])")
        obj, source, target = action.args
        object_pose = self.env_wrapper.get_object_pose(obj)
        target_pose = self.env_wrapper.get_target_pose(target)
        return {
            "object": obj,
            "source": source,
            "target": target,
            "object_pose": object_pose,
            "target_pose": target_pose,
            "grasp_pose": self.compute_grasp_pose(object_pose),
            "place_pose": self.compute_place_pose(target, target_pose),
        }

    def compute_grasp_pose(self, object_pose: dict) -> dict:
        return {**object_pose, "z": object_pose["z"] + self.grasp_offset}

    def compute_place_pose(self, target: str, target_pose: dict) -> dict:
        z = self.block_height / 2.0 if target == "table" else target_pose["z"] + self.block_height
        return {**target_pose, "z": z}
