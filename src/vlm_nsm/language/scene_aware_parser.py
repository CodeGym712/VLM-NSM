"""Scene-aware parser adapter — enriches goal parsing with scene context.

Wraps any ``GoalParser`` and injects scene information (detected objects,
spatial relations) into the parse call. This serves as the bridge between
future vision-based perception (Stage 15) and the current language-only
parsers.
"""

from __future__ import annotations

from typing import Any, Callable

from vlm_nsm.core.data_types import GoalParseResult
from vlm_nsm.perception.data_types import SceneInfo


class SceneAwareParserAdapter:
    """Adapter that wraps a base parser and enriches it with scene context.

    Args:
        base_parser: Any parser with a ``parse(instruction, **kwargs)`` method.
        scene_provider: Optional callable that returns a ``SceneInfo``.
    """

    def __init__(
        self,
        base_parser: Any,
        scene_provider: Callable[[], SceneInfo] | None = None,
    ):
        self._base_parser = base_parser
        self._scene_provider = scene_provider

    @property
    def base_parser(self) -> Any:
        return self._base_parser

    def set_scene_provider(self, provider: Callable[[], SceneInfo]) -> None:
        """Set or replace the scene provider callable."""
        self._scene_provider = provider

    def get_scene(self) -> SceneInfo | None:
        """Get current scene info from the provider."""
        if self._scene_provider is None:
            return None
        try:
            return self._scene_provider()
        except Exception:
            return None

    def parse(self, instruction: str, **kwargs: Any) -> GoalParseResult:
        """Parse with scene-aware enrichment.

        Injects ``available_objects`` from the scene if present and not
        already provided, then delegates to the base parser.
        """
        scene = self.get_scene()

        if scene is not None:
            scene_objects = scene.get_object_labels()
            if scene_objects:
                if "available_objects" not in kwargs or not kwargs.get(
                    "available_objects"
                ):
                    kwargs["available_objects"] = scene_objects
                if "objects" not in kwargs or not kwargs.get("objects"):
                    kwargs["objects"] = scene_objects

        return self._base_parser.parse(instruction=instruction, **kwargs)
