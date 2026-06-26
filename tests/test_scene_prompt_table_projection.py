"""Tests for scene prompt with estimated 3D positions (table projection).

No GPU required — uses synthetic SceneInfo with table-projected poses.
"""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description


def _make_table_projection_obj(
    label: str,
    color: str,
    x: float,
    y: float,
    z: float = 0.025,
    confidence: float = 0.9,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        category="block",
        color=color,
        pose={"x": x, "y": y, "z": z},
        confidence=confidence,
        attributes={
            "position_source": "rgb_bbox_table_plane_projection",
            "pose_confidence": confidence,
            "spatial": {
                "world_relations": {},
                "ambiguous_x": [],
                "ambiguous_y": [],
            },
        },
    )


class TestScenePromptTableProjection:
    def test_estimated_position_in_description(self):
        """Scene description includes estimated 3D position."""
        obj = _make_table_projection_obj("cube1", "red", 0.1, 0.0, 0.025, 0.9)
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "estimated position" in desc
        assert "(0.100, 0.000, 0.025)" in desc
        assert "confidence=0.90" in desc
        assert "cube1" in desc
        assert "red" in desc

    def test_guidance_note_table_projection(self):
        """Guidance note for table projection is included."""
        obj = _make_table_projection_obj("cube1", "red", 0.1, 0.0)
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "3D positions are estimated" in desc
        assert "estimated world left/right/front/behind" in desc
        assert "ungroundable" in desc

    def test_world_relations_in_description(self):
        """World left/right/front/behind relations appear in desc."""
        obj1 = _make_table_projection_obj("cube1", "red", -0.1, 0.0)
        obj2 = _make_table_projection_obj("cube2", "green", 0.1, 0.0)
        obj1.attributes["spatial"]["world_relations"] = {"cube2": ["left_of"]}
        obj2.attributes["spatial"]["world_relations"] = {"cube1": ["right_of"]}

        scene = SceneInfo(objects=[obj1, obj2], source="rgb_detector")
        desc = build_scene_description(scene)

        assert "left_of cube2" in desc
        assert "right_of cube1" in desc

    def test_empty_scene(self):
        """Empty scene produces concise description."""
        scene = SceneInfo(objects=[], source="rgb_detector")
        desc = build_scene_description(scene)
        assert "Current scene:" in desc
        assert "Gripper is empty" in desc

    def test_oracle_no_guidance_note(self):
        """No table projection guidance for oracle-source scene."""
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose={"x": 0.1, "y": 0.0, "z": 0.8},
        )
        scene = SceneInfo(objects=[obj], source="oracle")
        desc = build_scene_description(scene)
        assert "3D positions are estimated" not in desc
