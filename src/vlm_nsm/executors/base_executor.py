from __future__ import annotations


class BaseExecutor:
    def reset(self, init_state):
        raise NotImplementedError

    def execute(self, action):
        raise NotImplementedError

    def execute_plan(self, actions):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def check_success(self, goal_predicates):
        raise NotImplementedError
