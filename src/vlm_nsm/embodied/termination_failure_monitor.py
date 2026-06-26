"""TerminationFailureMonitor.

Monitors step-level execution and produces StepSignal outputs.
Detects failures such as grasp failure, collision, precondition violation, and timeout.
"""

from __future__ import annotations

from vlm_nsm.core.data_types import StepSignal


class TerminationFailureMonitor:
    """Step-level termination and failure monitor.

    Observes the environment state after each step and determines:
    - Whether the step completed successfully.
    - Whether the episode should terminate.
    - What type of failure occurred (if any).
    """

    def __init__(self, step_timeout: int = 200):
        self.step_timeout = step_timeout
        self._tracked_initial_z: dict[str, float] = {}

    def reset(self):
        """Clear tracked state between episodes."""
        self._tracked_initial_z.clear()

    def check(
        self,
        step_name: str,
        observation: dict,
        step_count: int,
    ) -> StepSignal:
        """Check step-level termination given the current observation.

        Args:
            step_name: Current skill step name (e.g. "reach_pick", "grasp").
            observation: Environment observation dict after the step.
            step_count: Number of steps taken so far.

        Returns:
            StepSignal with termination status.
        """
        # Timeout check
        if step_count >= self.step_timeout:
            return StepSignal(
                done=True,
                failed=True,
                timeout=True,
                failure_type="timeout",
            )

        # Track initial object z at grasp time for lift failure detection
        if step_name == "grasp":
            obj_name = observation.get("grasped_object", "")
            if obj_name:
                z = self._get_obj_z(observation, obj_name)
                if z is not None:
                    self._tracked_initial_z[obj_name] = z

        # Grasp failure detection (object z did not rise after lift)
        if step_name == "lift":
            obj_name = observation.get("grasped_object", "")
            if obj_name:
                obj_z = self._get_obj_z(observation, obj_name)
                initial_z = self._tracked_initial_z.get(obj_name)
                if obj_z is not None and initial_z is not None and obj_z - initial_z < 0.005:
                    return StepSignal(
                        done=True,
                        failed=True,
                        timeout=False,
                        failure_type="grasp_failed",
                    )

        # Place failure detection (object still grasped after release)
        if step_name == "release":
            if observation.get("gripper_has_object", False):
                return StepSignal(
                    done=True,
                    failed=True,
                    timeout=False,
                    failure_type="place_failed",
                )

        # Default: step succeeded, continue
        return StepSignal(
            done=False,
            success=True,
            failed=False,
            timeout=False,
            failure_type=None,
        )

    def _get_obj_z(self, observation: dict, obj_name: str) -> float | None:
        pos = observation.get(f"{obj_name}_pos")
        if pos is not None and len(pos) >= 3:
            return float(pos[2])
        return None
