from vlm_nsm.executors.base_executor import BaseExecutor


class RobosuiteIKExecutor(BaseExecutor):
    def reset(self, init_state):
        raise NotImplementedError("Robosuite IK executor is not part of stage 1")
