"""Test image-space spatial relation computation.

No GPU required — pure unit tests with synthetic DetectedObject data.
"""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.spatial_relations import compute_image_spatial_relations


def _make_obj(
    label: str,
    center_norm_x: float,
    center_norm_y: float = 0.5,
) -> DetectedObject:
    return DetectedObject(
        label=label,
        category="block",
        color="red",
        pose=None,
        attributes={
            "center_norm": {"x": center_norm_x, "y": center_norm_y},
            "position_source": "unknown_rgb_detector_3d",
        },
    )


class TestComputeImageSpatialRelations:
    def test_left_of_right_of(self):
        """Objects at x=0.3 and x=0.7 get image-left/image-right."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.3),
            _make_obj("cube2", 0.7),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512)
        s1 = scene.objects[0].attributes["spatial"]
        s2 = scene.objects[1].attributes["spatial"]

        assert s1["image_position"] == "left"
        assert "cube2" in s1["relations"]
        assert s1["relations"]["cube2"] == ["image_left_of"]

        assert s2["image_position"] == "right"
        assert "cube1" in s2["relations"]
        assert s2["relations"]["cube1"] == ["image_right_of"]

    def test_ambiguous_close_positions(self):
        """Objects with dx < threshold are marked ambiguous."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.49),
            _make_obj("cube2", 0.51),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512, threshold=0.05)
        s1 = scene.objects[0].attributes["spatial"]
        s2 = scene.objects[1].attributes["spatial"]

        # dx=0.02 < 0.05, so no directional relation
        assert s1["relations"] == {}
        assert "cube2" in s1["ambiguous_relations"]
        assert "cube1" in s2["ambiguous_relations"]

    def test_center_position(self):
        """Object at x=0.5 gets image_position=center."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.5),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512)
        s1 = scene.objects[0].attributes["spatial"]
        assert s1["image_position"] == "center"

    def test_three_objects(self):
        """Three objects produce correct pairwise relations."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.2),
            _make_obj("cube2", 0.5),
            _make_obj("cube3", 0.8),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512)
        s1 = scene.objects[0].attributes["spatial"]
        s2 = scene.objects[1].attributes["spatial"]
        s3 = scene.objects[2].attributes["spatial"]

        assert s1["image_position"] == "left"
        assert s1["relations"]["cube2"] == ["image_left_of"]
        assert s1["relations"]["cube3"] == ["image_left_of"]

        assert s2["image_position"] == "center"
        # cube2 is at x=0.5, cube1 at 0.2 (dx=0.3 -> cube2 right of cube1)
        assert s2["relations"]["cube1"] == ["image_right_of"]
        # cube2 at 0.5, cube3 at 0.8 (dx=-0.3 -> cube2 left of cube3)
        assert s2["relations"]["cube3"] == ["image_left_of"]

        assert s3["image_position"] == "right"
        assert s3["relations"]["cube1"] == ["image_right_of"]
        assert s3["relations"]["cube2"] == ["image_right_of"]

    def test_no_center_norm_skipped(self):
        """Objects without center_norm are skipped in spatial computation."""
        obj_no_spatial = DetectedObject(
            label="cube1", category="block", color="red",
        )
        obj_with = _make_obj("cube2", 0.7)
        scene = SceneInfo(objects=[obj_no_spatial, obj_with])
        scene = compute_image_spatial_relations(scene, 512, 512)
        # obj_no_spatial should not get a spatial key
        assert "spatial" not in scene.objects[0].attributes
        # obj_with should have spatial
        assert "spatial" in scene.objects[1].attributes

    def test_single_object(self):
        """Single object has no relations but gets image_position."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.3),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512)
        s1 = scene.objects[0].attributes["spatial"]
        assert s1["image_position"] == "left"
        assert s1["relations"] == {}
        assert s1["ambiguous_relations"] == []

    def test_center_norm_stored(self):
        """center_norm list is stored in spatial metadata."""
        scene = SceneInfo(objects=[
            _make_obj("cube1", 0.32, 0.54),
        ])
        scene = compute_image_spatial_relations(scene, 512, 512)
        cn = scene.objects[0].attributes["spatial"]["center_norm"]
        assert cn == [0.32, 0.54]
