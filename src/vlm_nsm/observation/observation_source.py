"""Abstract ObservationSource interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from vlm_nsm.observation.scene_state import SceneState, ObservationLegalityReport


class ObservationSource(ABC):
    """Abstract interface for all state observation sources.

    Controller code should consume SceneState, not call env methods directly.
    """

    @abstractmethod
    def reset(self, env: Any) -> None:
        """Reset the source for a new episode."""

    @abstractmethod
    def get_scene_state(self, env: Any, camera_obs: Any = None) -> SceneState:
        """Get current scene state through this source."""

    @abstractmethod
    def get_legality_report(self) -> ObservationLegalityReport:
        """Return an audit report describing privilege/perception usage."""
