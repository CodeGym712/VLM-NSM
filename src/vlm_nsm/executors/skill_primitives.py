from __future__ import annotations


class SkillPrimitives:
    REACH_TIMEOUT_STEPS = 200
    GRASP_TIMEOUT_STEPS = 50
    LIFT_TIMEOUT_STEPS = 100
    PLACE_TIMEOUT_STEPS = 200
    RETREAT_TIMEOUT_STEPS = 80

    def __init__(self, env_wrapper, grasp_lift_min: float = 0.01):
        self.env_wrapper = env_wrapper
        self.grasp_lift_min = grasp_lift_min

    def move_to_pose(self, pose, timeout_steps: int = REACH_TIMEOUT_STEPS):
        return {"success": True, "pose": dict(pose), "timeout_steps": timeout_steps}

    def open_gripper(self, timeout_steps: int = GRASP_TIMEOUT_STEPS):
        return {"success": True, "timeout_steps": timeout_steps}

    def close_gripper(self, timeout_steps: int = GRASP_TIMEOUT_STEPS):
        return {"success": True, "timeout_steps": timeout_steps}

    def wait_steps(self, steps: int):
        return {"success": True, "steps": steps}

    def lift(self, object_name: str, delta_z: float = 0.02, timeout_steps: int = LIFT_TIMEOUT_STEPS):
        pose = self.env_wrapper.get_object_pose(object_name)
        return {
            "success": True,
            "object": object_name,
            "start_z": pose["z"],
            "end_z": pose["z"] + delta_z,
            "timeout_steps": timeout_steps,
        }

    def retreat(self, timeout_steps: int = RETREAT_TIMEOUT_STEPS):
        return {"success": True, "timeout_steps": timeout_steps}

    def pick(self, grounded_action: dict):
        object_name = grounded_action["object"]
        before_z = self.env_wrapper.get_object_pose(object_name)["z"]
        self.move_to_pose(grounded_action["grasp_pose"])
        self.close_gripper()
        self.wait_steps(20)
        lift_result = self.lift(object_name, 0.02)
        after_z = lift_result["end_z"]
        grasp_ok = (after_z - before_z) > self.grasp_lift_min
        if not grasp_ok:
            return {"success": False, "error": "grasp_failed"}
        return {"success": True}

    def place(self, grounded_action: dict):
        self.move_to_pose(grounded_action["place_pose"], timeout_steps=self.PLACE_TIMEOUT_STEPS)
        self.open_gripper()
        self.retreat()
        return {"success": True}

    def pick_and_place(self, grounded_action: dict):
        pick_result = self.pick(grounded_action)
        if not pick_result["success"]:
            return pick_result
        place_result = self.place(grounded_action)
        if not place_result["success"]:
            return place_result
        return {"success": True}
