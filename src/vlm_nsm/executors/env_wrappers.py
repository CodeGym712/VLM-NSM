from __future__ import annotations


class BaseEnvWrapper:
    def reset(self):
        raise NotImplementedError

    def get_object_pose(self, name):
        raise NotImplementedError

    def get_target_pose(self, name):
        raise NotImplementedError

    def get_all_object_poses(self):
        raise NotImplementedError

    def get_objects_on(self, target_name):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
