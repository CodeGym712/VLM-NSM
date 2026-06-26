"""Test scene prompt includes spatial metadata from RGB detector.

No GPU required — pure unit tests using synthetic SceneInfo.
"""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


def _make_detector_obj(
    label: str,
    color: str,
    center_norm_x: float,
    center_norm_y: float = 0.5,
    confidence: float = 0.9,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        category="block",
        color=color,
        pose=None,
        confidence=confidence,
        attributes={
            "center_norm": {"x": center_norm_x, "y": center_norm_y},
            "position_source": "unknown_rgb_detector_3d",
            "spatial": {
                "center_norm": [center_norm_x, center_norm_y],
                "image_position": "left" if center_norm_x < 0.4 else "right",
                "relations": {},
                "ambiguous_relations": [],
            },
        },
    )


class TestScenePromptSpatial:
    def test_spatial_info_in_description(self):
        """Scene description includes image center and position."""
        obj = _make_detector_obj("cube1", "red", 0.32, 0.54)
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "image center=(0.32, 0.54)" in desc
        assert "image-left" in desc
        assert "cube1" in desc
        assert "red" in desc
        assert "confidence=0.90" in desc

    def test_guidance_note_present(self):
        """Guidance note about 3D positions unknown is included."""
        obj = _make_detector_obj("cube1", "red", 0.3)
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "3D positions are unknown" in desc
        assert "image-left/image-right" in desc
        assert "ungroundable" in desc

    def test_guidance_note_absent_for_oracle(self):
        """No guidance note for oracle-source scene."""
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose={"x": 0.1, "y": 0.0, "z": 0.8},
        )
        scene = SceneInfo(objects=[obj], source="oracle")
        desc = build_scene_description(scene)

        assert "3D positions are unknown" not in desc
        assert "ungroundable" not in desc

    def test_spatial_relations_in_description(self):
        """image-left-of / image-right-of relations appear in desc."""
        obj1 = _make_detector_obj("cube1", "red", 0.32)
        obj2 = _make_detector_obj("cube2", "green", 0.67)
        # Add relations
        obj1.attributes["spatial"]["relations"] = {"cube2": ["image_left_of"]}
        obj1.attributes["spatial"]["image_position"] = "left"
        obj2.attributes["spatial"]["relations"] = {"cube1": ["image_right_of"]}
        obj2.attributes["spatial"]["image_position"] = "right"

        scene = SceneInfo(objects=[obj1, obj2], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "image_left_of cube2" in desc
        assert "image_right_of cube1" in desc

    def test_ambiguous_relations_in_description(self):
        """x-ambiguous-with appears in desc."""
        obj1 = _make_detector_obj("cube1", "red", 0.49)
        obj2 = _make_detector_obj("cube2", "green", 0.51)
        obj1.attributes["spatial"]["image_position"] = "center"
        obj1.attributes["spatial"]["ambiguous_relations"] = ["cube2"]
        obj2.attributes["spatial"]["image_position"] = "center"
        obj2.attributes["spatial"]["ambiguous_relations"] = ["cube1"]

        scene = SceneInfo(objects=[obj1, obj2], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "x-ambiguous-with cube2" in desc
        assert "x-ambiguous-with cube1" in desc

    def test_empty_scene(self):
        """Empty scene produces concise description."""
        scene = SceneInfo(objects=[], source="rgb_detector")
        desc = build_scene_description(scene)
        assert "Current scene:" in desc
        assert "Gripper is empty" in desc
