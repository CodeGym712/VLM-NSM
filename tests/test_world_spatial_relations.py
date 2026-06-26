"""Tests for world spatial relations — no GPU required.

Uses synthetic DetectedObject data with estimated 3D poses.
"""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.spatial_relations import compute_world_spatial_relations


def _make_posed_obj(
    label: str,
    x: float,
    y: float,
    z: float = 0.025,
    color: str = "red",
) -> DetectedObject:
    return DetectedObject(
        label=label,
        category="block",
        color=color,
        pose={"x": x, "y": y, "z": z},
        attributes={
            "position_source": "rgb_bbox_table_plane_projection",
        },
    )


class TestComputeWorldSpatialRelations:
    def test_left_right(self):
        """Objects at different X get left_of / right_of."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=-0.1, y=0.0),
            _make_posed_obj("cube2", x=0.1, y=0.0),
        ])
        scene = compute_world_spatial_relations(scene, xy_threshold=0.02)
        w1 = scene.objects[0].attributes["spatial"]["world_relations"]
        w2 = scene.objects[1].attributes["spatial"]["world_relations"]

        # cube1 at x=-0.1, cube2 at x=0.1; dx=-0.2 -> left_of
        assert "left_of" in w1["cube2"]
        assert "right_of" in w2["cube1"]

    def test_front_behind(self):
        """Objects at different Y get front_of / behind."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=0.0, y=-0.1),  # front (closer to robot)
            _make_posed_obj("cube2", x=0.0, y=0.1),   # behind (away from robot)
        ])
        scene = compute_world_spatial_relations(scene, xy_threshold=0.02)
        w1 = scene.objects[0].attributes["spatial"]["world_relations"]
        w2 = scene.objects[1].attributes["spatial"]["world_relations"]

        # cube1 at y=-0.1, cube2 at y=0.1
        # dy = -0.1 - 0.1 = -0.2, dy < 0 -> front_of (closer to robot)
        assert "front_of" in w1["cube2"]
        # cube2 at y=0.1, cube1 at y=-0.1
        # dy = 0.1 - (-0.1) = 0.2, dy > 0 -> behind (further from robot)
        assert "behind" in w2["cube1"]

    def test_near_far(self):
        """Objects close by get near; far apart get far."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=0.0, y=0.0),
            _make_posed_obj("cube2", x=0.01, y=0.01),   # near (~0.014m)
            _make_posed_obj("cube3", x=0.5, y=0.0),     # far (0.5m)
        ])
        scene = compute_world_spatial_relations(
            scene, xy_threshold=0.02, near_threshold=0.15, far_threshold=0.30,
        )
        w1 = scene.objects[0].attributes["spatial"]["world_relations"]

        assert "near" in w1["cube2"]
        assert "far" in w1["cube3"]

    def test_ambiguous_x(self):
        """Objects with similar X get marked ambiguous_x."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=0.0, y=0.0),
            _make_posed_obj("cube2", x=0.01, y=0.1),  # dx=0.01 < 0.02 threshold
        ])
        scene = compute_world_spatial_relations(scene, xy_threshold=0.02)
        amb_x = scene.objects[0].attributes["spatial"]["ambiguous_x"]
        assert "cube2" in amb_x

    def test_ambiguous_y(self):
        """Objects with similar Y get marked ambiguous_y."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=0.0, y=0.0),
            _make_posed_obj("cube2", x=0.1, y=0.01),  # dy=0.01 < 0.02 threshold
        ])
        scene = compute_world_spatial_relations(scene, xy_threshold=0.02)
        amb_y = scene.objects[0].attributes["spatial"]["ambiguous_y"]
        assert "cube2" in amb_y

    def test_no_pose_skipped(self):
        """Object without pose is skipped entirely."""
        obj_no_pose = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None,
        )
        obj_with_pose = _make_posed_obj("cube2", x=0.1, y=0.0)
        scene = SceneInfo(objects=[obj_no_pose, obj_with_pose])
        scene = compute_world_spatial_relations(scene)
        # obj_no_pose should not have spatial key
        assert "spatial" not in scene.objects[0].attributes
        # obj_with_pose should have world_relations
        assert "world_relations" in scene.objects[1].attributes.get("spatial", {})

    def test_single_object(self):
        """Single object produces empty relations."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=0.0, y=0.0),
        ])
        scene = compute_world_spatial_relations(scene)
        wr = scene.objects[0].attributes["spatial"]["world_relations"]
        assert wr == {}

    def test_three_objects(self):
        """Three objects produce correct pairwise relations."""
        scene = SceneInfo(objects=[
            _make_posed_obj("cube1", x=-0.2, y=0.0),
            _make_posed_obj("cube2", x=0.0, y=0.0),
            _make_posed_obj("cube3", x=0.2, y=0.0),
        ])
        scene = compute_world_spatial_relations(scene, xy_threshold=0.02)
        w1 = scene.objects[0].attributes["spatial"]["world_relations"]
        w3 = scene.objects[2].attributes["spatial"]["world_relations"]

        # cube1 at x=-0.2 is left of both
        assert "left_of" in w1["cube2"]
        assert "left_of" in w1["cube3"]
        # cube3 at x=0.2 is right of both
        assert "right_of" in w3["cube1"]
        assert "right_of" in w3["cube2"]
