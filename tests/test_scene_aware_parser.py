"""Test SceneAwareParserAdapter — scene-aware enrichment of goal parsing."""
from vlm_nsm.core.data_types import GoalParseResult
from vlm_nsm.language.scene_aware_parser import SceneAwareParserAdapter
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo


class _MockParser:
    """Minimal parser that records last kwargs for inspection."""

    def __init__(self):
        self.last_kwargs = None

    def parse(self, instruction, **kwargs):
        self.last_kwargs = kwargs
        return GoalParseResult(
            success=True,
            goal_json={"task_type": "stacking", "goal_predicates": []},
        )


class _SceneProvider:
    def __init__(self, scene):
        self._scene = scene

    def __call__(self):
        return self._scene


def test_adapter_delegates_to_base():
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base)
    result = adapter.parse("test instruction")
    assert result.success is True
    assert result.goal_json["task_type"] == "stacking"


def test_adapter_injects_available_objects():
    base = _MockParser()
    scene = SceneInfo(
        objects=[
            DetectedObject(label="cube1", category="block"),
            DetectedObject(label="cube2", category="block"),
        ]
    )
    adapter = SceneAwareParserAdapter(base, scene_provider=_SceneProvider(scene))
    adapter.parse("stack the blocks")
    assert base.last_kwargs is not None
    assert "available_objects" in base.last_kwargs
    assert base.last_kwargs["available_objects"] == ["cube1", "cube2"]


def test_adapter_does_not_override_existing_objects():
    base = _MockParser()
    scene = SceneInfo(
        objects=[
            DetectedObject(label="cube1", category="block"),
        ]
    )
    adapter = SceneAwareParserAdapter(base, scene_provider=_SceneProvider(scene))
    adapter.parse("test", available_objects=["custom1"])
    assert base.last_kwargs["available_objects"] == ["custom1"]


def test_adapter_without_scene_provider():
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base)
    adapter.parse("test")
    assert base.last_kwargs is not None


def test_base_parser_property():
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base)
    assert adapter.base_parser is base


def test_set_scene_provider():
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base)
    scene = SceneInfo(objects=[DetectedObject(label="cube1", category="block")])
    adapter.set_scene_provider(_SceneProvider(scene))
    adapter.parse("test")
    assert base.last_kwargs["available_objects"] == ["cube1"]


def test_get_scene_returns_none_without_provider():
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base)
    assert adapter.get_scene() is None


def test_get_scene_returns_scene():
    scene = SceneInfo()
    base = _MockParser()
    adapter = SceneAwareParserAdapter(base, scene_provider=_SceneProvider(scene))
    assert adapter.get_scene() is scene


def test_provider_exception_handled():
    def _failing():
        raise RuntimeError("perception failed")

    base = _MockParser()
    adapter = SceneAwareParserAdapter(base, scene_provider=_failing)
    result = adapter.parse("test")
    assert result.success is True  # falls through gracefully
