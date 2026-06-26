"""Observation layer: unified state interface for embodied control.

Provides SceneState, ObservationSource, and implementations
that decouple controller code from simulator privileged state.
"""

from vlm_nsm.observation.scene_state import (
    ObjectState,
    RobotState,
    SceneState,
    ObservationLegalityReport,
)

from vlm_nsm.observation.observation_source import ObservationSource
from vlm_nsm.observation.oracle_state_source import OracleStateSource
from vlm_nsm.observation.noisy_oracle_state_source import NoisyOracleStateSource
from vlm_nsm.observation.perception_state_source import PerceptionStateSource

__all__ = [
    "ObjectState",
    "RobotState",
    "SceneState",
    "ObservationLegalityReport",
    "ObservationSource",
    "OracleStateSource",
    "NoisyOracleStateSource",
    "PerceptionStateSource",
]
