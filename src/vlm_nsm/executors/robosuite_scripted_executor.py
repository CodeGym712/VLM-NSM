from __future__ import annotations

from vlm_nsm.core.action_grounder import ActionGrounder
from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates
from vlm_nsm.core.predicate_monitor import compare_symbolic_physical, goal_satisfied
from vlm_nsm.executors.base_executor import BaseExecutor
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper
from vlm_nsm.executors.skill_primitives import SkillPrimitives


class RobosuiteScriptedExecutor(BaseExecutor):
    def __init__(self, env_wrapper, task_name: str = "stacking", thresholds: dict | None = None, block_height: float = 0.04):
        if task_name != "stacking":
            raise NotImplementedError("RobosuiteScriptedExecutor only supports task_name='stacking' in this stage.")
        self.env_wrapper = env_wrapper
        self.task_name = task_name
        self.thresholds = thresholds or {"xy_threshold": 0.03, "z_min": 0.02, "z_max": 0.06}
        self.grounder = ActionGrounder(env_wrapper, block_height=block_height)
        self.skills = SkillPrimitives(env_wrapper, grasp_lift_min=0.01)
        self.goal_predicates = []
        self.physical_state = set()
        self.last_comparison = {"mismatch_rate": 0.0, "missing_in_physical": [], "extra_in_physical": []}

    def reset(self, init_state=None):
        self.env_wrapper.reset()
        self.physical_state = self._extract_physical_state()
        return self.physical_state

    def execute(self, action):
        grounded = self.grounder.ground_move(action)
        if isinstance(self.env_wrapper, MockEnvWrapper):
            result = self.env_wrapper.simulate_pick_and_place(grounded["object"], grounded["target"])
            if not result["success"]:
                raise ValueError(result["error"])
        else:
            result = self.skills.pick_and_place(grounded)
            if not result["success"]:
                raise ValueError(result["error"])
        self.physical_state = self._extract_physical_state()
        self.last_comparison = compare_symbolic_physical(self.goal_predicates, self.physical_state)

    def execute_plan(self, actions):
        for idx, action in enumerate(actions):
            try:
                self.execute(action)
            except Exception as exc:
                return {
                    "success": False,
                    "failed_step": idx,
                    "error": str(exc),
                    "physical_state": set(self.physical_state),
                    "comparison": self.last_comparison,
                }
        return {
            "success": True,
            "failed_step": None,
            "error": None,
            "physical_state": set(self.physical_state),
            "comparison": self.last_comparison,
        }

    def get_state(self):
        return set(self.physical_state)

    def check_success(self, goal_predicates):
        return goal_satisfied(self.physical_state, goal_predicates)

    def set_goal(self, goal_predicates):
        self.goal_predicates = list(goal_predicates)

    def _extract_physical_state(self):
        object_poses = self.env_wrapper.get_all_object_poses()
        targets = {"table": self.env_wrapper.get_target_pose("table")} | object_poses
        return physical_state_to_predicates(object_poses, targets, self.thresholds)
