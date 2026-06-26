"""Test scene_prompt: build_scene_description and inject_scene_into_prompt."""
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import (
    build_scene_description,
    inject_scene_into_prompt,
)


def _make_scene():
    return SceneInfo(
        objects=[
            DetectedObject(label="cube1", category="block", pose={"x": 0.1, "y": 0.2, "z": 0.0}),
            DetectedObject(label="cube2", category="block", pose={"x": 0.0, "y": 0.0, "z": 0.05}),
            DetectedObject(label="table", category="table", pose={"x": 0.0, "y": 0.0, "z": -0.01}),
        ],
        on_relations=[("cube1", "cube2")],
        free_gripper=True,
    )


def test_build_scene_description_contains_labels():
    desc = build_scene_description(_make_scene())
    assert "cube1" in desc
    assert "cube2" in desc
    assert "table" in desc


def test_build_scene_description_contains_relations():
    desc = build_scene_description(_make_scene())
    assert "cube1 is on cube2" in desc


def test_build_scene_description_contains_gripper():
    desc = build_scene_description(_make_scene())
    assert "Gripper is empty" in desc


def test_build_scene_description_with_grasped():
    scene = SceneInfo(
        objects=[DetectedObject(label="cube1", category="block")],
        grasped_object="cube1",
        free_gripper=False,
    )
    desc = build_scene_description(scene)
    assert "Grasping: cube1" in desc


def test_build_scene_description_empty():
    desc = build_scene_description(SceneInfo())
    assert "Current scene:" in desc
    assert "Gripper is empty" in desc


def test_inject_scene_into_prompt_appends():
    prompt = "Do something."
    scene = _make_scene()
    result = inject_scene_into_prompt(prompt, scene)
    assert result.startswith(prompt)
    assert "Current scene:" in result


def test_inject_scene_into_prompt_replaces_placeholder():
    prompt = "Scene: {scene_description}\nParse:"
    scene = _make_scene()
    result = inject_scene_into_prompt(prompt, scene)
    assert "{scene_description}" not in result
    assert "Current scene:" in result
    assert result.startswith("Scene:")
    assert result.endswith("\nParse:")
