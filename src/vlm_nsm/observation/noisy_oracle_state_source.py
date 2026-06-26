"""NoisyOracleStateSource — adds simulated perception noise to oracle poses."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from vlm_nsm.observation.observation_source import ObservationSource
from vlm_nsm.observation.oracle_state_source import OracleStateSource
from vlm_nsm.observation.scene_state import (
    ObjectState, RobotState, SceneState, ObservationLegalityReport,
)


class NoisyOracleStateSource(ObservationSource):
    """Wraps OracleStateSource and adds Gaussian noise to object positions.

    Simulates perception-like pose errors for sensitivity analysis.
    Still privileged (uses oracle as base), not deployable.
    """

    def __init__(
        self,
        cubes: list[str] | None = None,
        xy_noise_std_m: float = 0.005,
        z_noise_std_m: float = 0.005,
        bias_xyz: list[float] | None = None,
        dropout_prob: float = 0.0,
        identity_swap_prob: float = 0.0,
        seed: int = 0,
    ):
        self._oracle = OracleStateSource(cubes=cubes)
        self.xy_noise_std_m = xy_noise_std_m
        self.z_noise_std_m = z_noise_std_m
        self.bias_xyz = np.array(bias_xyz or [0.0, 0.0, 0.0], dtype=np.float32)
        self.dropout_prob = dropout_prob
        self.identity_swap_prob = identity_swap_prob
        self._rng = np.random.RandomState(seed)

    def reset(self, env: Any) -> None:
        self._oracle.reset(env)

    def get_scene_state(self, env: Any, camera_obs: Any = None) -> SceneState:
        oracle_state = self._oracle.get_scene_state(env, camera_obs)

        for name, obj in oracle_state.objects.items():
            if not obj.available or obj.position is None:
                continue

            # Dropout
            if self.dropout_prob > 0 and self._rng.rand() < self.dropout_prob:
                obj.available = False
                obj.metadata["noise_dropped"] = True
                obj.metadata["oracle_position"] = obj.position.copy()
                obj.position = None
                continue

            # Identity swap
            if self.identity_swap_prob > 0:
                other_names = [n for n in oracle_state.objects if n != name]
                if other_names and self._rng.rand() < self.identity_swap_prob:
                    other = self._rng.choice(other_names)
                    obj.position, oracle_state.objects[other].position = (
                        oracle_state.objects[other].position, obj.position)
                    obj.metadata["noise_identity_swapped_with"] = other

            # Gaussian noise
            noise_xy = self._rng.randn(2) * self.xy_noise_std_m
            noise_z = self._rng.randn() * self.z_noise_std_m
            noise = np.array([noise_xy[0], noise_xy[1], noise_z], dtype=np.float32)

            obj.metadata["oracle_position"] = obj.position.copy()
            obj.metadata["noise_xyz"] = noise.copy()
            obj.metadata["noise_xy_std_m"] = self.xy_noise_std_m
            obj.metadata["noise_z_std_m"] = self.z_noise_std_m
            obj.position = obj.position + noise + self.bias_xyz
            obj.source = "noisy_oracle"

        oracle_state.source = "noisy_oracle"
        oracle_state.metadata["noise_config"] = {
            "xy_noise_std_m": self.xy_noise_std_m,
            "z_noise_std_m": self.z_noise_std_m,
            "dropout_prob": self.dropout_prob,
            "identity_swap_prob": self.identity_swap_prob,
        }
        return oracle_state

    def get_legality_report(self) -> ObservationLegalityReport:
        return ObservationLegalityReport(
            observation_source_name=f"NoisyOracleStateSource(xy={self.xy_noise_std_m*1000:.0f}mm,z={self.z_noise_std_m*1000:.0f}mm)",
            uses_privileged_object_pose=True,
            uses_privileged_eef_pose=True,
            uses_privileged_contact=False,
            uses_perception_object_pose=False,
            real_world_deployable=False,
            notes=[
                f"Simulated perception noise: xy_std={self.xy_noise_std_m*1000:.0f}mm, z_std={self.z_noise_std_m*1000:.0f}mm",
                "Based on oracle poses; not real perception.",
                "Used for sensitivity analysis, not deployment.",
            ],
        )
