# DEPRECATED (2026-06-10): VLA execution in experiments/eval_s1_phase_override.py. See note/controller.md.
from vlm_nsm.executors.base_executor import BaseExecutor


class DiffusionPolicyExecutor(BaseExecutor):
    def reset(self, init_state):
        raise NotImplementedError("Diffusion Policy executor is not part of stage 1")
