"""Test perception data types: SceneInfo and DetectedObject."""
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo


def test_detected_object_defaults():
    obj = DetectedObject(label="cube1", category="block")
    assert obj.label == "cube1"
    assert obj.category == "block"
    assert obj.pose is None
    assert obj.color is None
    assert obj.confidence == 1.0
    assert obj.attributes == {}


def test_detected_object_full():
    obj = DetectedObject(
        label="cube1",
        category="block",
        pose={"x": 0.1, "y": 0.2, "z": 0.0},
        color="red",
        confidence=0.95,
    )
    assert obj.pose["x"] == 0.1
    assert obj.color == "red"


def test_scene_info_defaults():
    scene = SceneInfo()
    assert scene.objects == []
    assert scene.on_relations == []
    assert scene.free_gripper is True
    assert scene.grasped_object is None


def test_get_objects_by_category():
    scene = SceneInfo(
        objects=[
            DetectedObject(label="cube1", category="block"),
            DetectedObject(label="table", category="table"),
            DetectedObject(label="cube2", category="block"),
        ]
    )
    blocks = scene.get_objects_by_category("block")
    assert len(blocks) == 2
    assert blocks[0].label == "cube1"
    assert blocks[1].label == "cube2"


def test_get_object_labels():
    scene = SceneInfo(
        objects=[
            DetectedObject(label="cube1", category="block"),
            DetectedObject(label="table", category="table"),
        ]
    )
    assert scene.get_object_labels() == ["cube1", "table"]
