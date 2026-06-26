from __future__ import annotations

from copy import deepcopy

from vlm_nsm.executors.env_wrappers import BaseEnvWrapper


class MockEnvWrapper(BaseEnvWrapper):
    def __init__(self, cubes: list[str] | None = None, block_height: float = 0.04):
        self.cubes = cubes or ["cube1", "cube2", "cube3"]
        self.block_height = block_height
        self.table_pose = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.object_poses: dict[str, dict[str, float]] = {}
        self.supports: dict[str, str] = {}
        self.reset()

    def reset(self):
        self.object_poses = {}
        self.supports = {}
        for idx, cube in enumerate(self.cubes):
            self.object_poses[cube] = {
                "x": 0.08 * idx,
                "y": 0.0,
                "z": self.block_height / 2.0,
            }
            self.supports[cube] = "table"
        return self.get_all_object_poses()

    def get_object_pose(self, name):
        if name not in self.object_poses:
            raise KeyError(f"unknown object: {name}")
        return deepcopy(self.object_poses[name])

    def get_target_pose(self, name):
        if name == "table":
            return dict(self.table_pose)
        return self.get_object_pose(name)

    def get_all_object_poses(self):
        return deepcopy(self.object_poses)

    def get_objects_on(self, target_name):
        return sorted(obj for obj, support in self.supports.items() if support == target_name)

    def step(self, action):
        return {"success": True, "action": action}

    def simulate_pick_and_place(self, object_name: str, target_name: str):
        if object_name not in self.object_poses:
            return {"success": False, "error": f"unknown object: {object_name}"}
        if target_name != "table" and target_name not in self.object_poses:
            return {"success": False, "error": f"unknown target: {target_name}"}
        if self.get_objects_on(object_name):
            return {"success": False, "error": f"object is not clear: {object_name}"}
        if target_name != "table" and self.get_objects_on(target_name):
            return {"success": False, "error": f"target is not clear: {target_name}"}

        target_pose = self.get_target_pose(target_name)
        z = self.block_height / 2.0 if target_name == "table" else target_pose["z"] + self.block_height
        self.object_poses[object_name] = {"x": target_pose["x"], "y": target_pose["y"], "z": z}
        self.supports[object_name] = target_name
        return {"success": True}
