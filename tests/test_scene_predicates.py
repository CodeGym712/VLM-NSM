"""Tests for scene_predicates.py — no GPU required.

Uses synthetic DetectedObject/SceneInfo with estimated poses.
"""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_context import SceneReferenceFrame
from vlm_nsm.perception.scene_predicates import (
    compute_on_table_status,
    compute_robot_relative,
    compute_edge_relations,
    filter_pose_outliers,
    compute_scene_predicates,
)


def _make_obj(label: str, x: float, y: float, z: float, **attrs) -> DetectedObject:
    obj = DetectedObject(
        label=label,
        category="block",
        color=attrs.pop("color", "red"),
        pose={"x": x, "y": y, "z": z},
    )
    obj.attributes.update(attrs)
    return obj


def _make_obj_no_pose(label: str) -> DetectedObject:
    return DetectedObject(label=label, category="block", color="red", pose=None)


# ---------------------------------------------------------------------------
# on_table
# ---------------------------------------------------------------------------


class TestOnTableStatus:
    def test_on_table(self):
        """Object at table_z + h/2 should be on_table=True."""
        rf = SceneReferenceFrame(table_z_surface=0.8)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=0.0, z=0.825)])
        compute_on_table_status(scene, rf, object_height=0.05, z_tolerance=0.02)
        assert scene.objects[0].attributes["on_table"] is True

    def test_stacked(self):
        """Object at higher Z (stacked on another) should be on_table=False."""
        rf = SceneReferenceFrame(table_z_surface=0.8)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=0.0, z=0.90)])
        compute_on_table_status(scene, rf, object_height=0.05, z_tolerance=0.02)
        assert scene.objects[0].attributes["on_table"] is False

    def test_borderline(self):
        """Object at exactly tolerance boundary."""
        rf = SceneReferenceFrame(table_z_surface=0.8)
        expected_z = 0.8 + 0.05 / 2.0  # 0.825
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=0.0, z=expected_z + 0.019)])
        compute_on_table_status(scene, rf, object_height=0.05, z_tolerance=0.02)
        assert scene.objects[0].attributes["on_table"] is True

    def test_no_pose(self):
        rf = SceneReferenceFrame()
        scene = SceneInfo(objects=[_make_obj_no_pose("cube1")])
        compute_on_table_status(scene, rf)
        assert scene.objects[0].attributes["on_table"] is None


# ---------------------------------------------------------------------------
# robot_relative
# ---------------------------------------------------------------------------


class TestRobotRelative:
    def test_robot_left(self):
        rf = SceneReferenceFrame(robot_base_x=0.5, robot_base_y=0.0)
        scene = SceneInfo(objects=[_make_obj("cube1", x=-0.3, y=0.0, z=0.825)])
        compute_robot_relative(scene, rf)
        obj = scene.objects[0]
        assert "robot-left" in obj.attributes["robot_direction"]
        assert "robot-center-y" in obj.attributes["robot_direction"]

    def test_toward_robot(self):
        rf = SceneReferenceFrame(robot_base_x=0.5, robot_base_y=0.0)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.5, y=-0.2, z=0.825)])
        compute_robot_relative(scene, rf)
        obj = scene.objects[0]
        assert "toward-robot" in obj.attributes["robot_direction"]

    def test_distance(self):
        rf = SceneReferenceFrame(robot_base_x=0.5, robot_base_y=0.0)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.5, y=0.3, z=0.825)])
        compute_robot_relative(scene, rf)
        assert scene.objects[0].attributes["robot_distance"] == pytest.approx(0.3)

    def test_no_pose(self):
        rf = SceneReferenceFrame()
        scene = SceneInfo(objects=[_make_obj_no_pose("cube1")])
        compute_robot_relative(scene, rf)
        assert scene.objects[0].attributes["robot_distance"] is None
        assert scene.objects[0].attributes["robot_direction"] == []


# ---------------------------------------------------------------------------
# edge_relations
# ---------------------------------------------------------------------------


class TestEdgeRelations:
    def test_near_front_edge(self):
        rf = SceneReferenceFrame(
            table_y_min=-0.4, table_y_max=0.4, edge_threshold=0.10,
        )
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=-0.35, z=0.825)])
        compute_edge_relations(scene, rf)
        assert "front_edge" in scene.objects[0].attributes["near_edges"]

    def test_near_left_edge(self):
        rf = SceneReferenceFrame(
            table_x_min=-0.4, table_x_max=0.4, edge_threshold=0.10,
        )
        scene = SceneInfo(objects=[_make_obj("cube1", x=-0.35, y=0.0, z=0.825)])
        compute_edge_relations(scene, rf)
        assert "left_edge" in scene.objects[0].attributes["near_edges"]

    def test_center_no_edges(self):
        rf = SceneReferenceFrame(edge_threshold=0.10)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=0.0, z=0.825)])
        compute_edge_relations(scene, rf)
        assert scene.objects[0].attributes["near_edges"] == []

    def test_no_pose(self):
        rf = SceneReferenceFrame()
        scene = SceneInfo(objects=[_make_obj_no_pose("cube1")])
        compute_edge_relations(scene, rf)
        assert scene.objects[0].attributes["near_edges"] == []


# ---------------------------------------------------------------------------
# pose_outliers
# ---------------------------------------------------------------------------


class TestPoseOutliers:
    def test_in_workspace(self):
        rf = SceneReferenceFrame(workspace_padding=0.30)
        scene = SceneInfo(objects=[_make_obj("cube1", x=0.0, y=0.0, z=0.825)])
        filter_pose_outliers(scene, rf)
        obj = scene.objects[0]
        assert obj.attributes["pose_in_workspace"] is True
        assert obj.attributes["pose_suspicious"] is False

    def test_out_of_workspace(self):
        rf = SceneReferenceFrame(
            table_x_min=-0.4, table_x_max=0.4,
            table_y_min=-0.4, table_y_max=0.4,
            workspace_padding=0.30,
        )
        scene = SceneInfo(objects=[_make_obj("cube1", x=-1.25, y=0.0, z=0.825)])
        filter_pose_outliers(scene, rf)
        obj = scene.objects[0]
        assert obj.attributes["pose_in_workspace"] is False
        assert obj.attributes["pose_suspicious"] is True
        assert "suspicious_reason" in obj.attributes

    def test_no_pose(self):
        rf = SceneReferenceFrame()
        scene = SceneInfo(objects=[_make_obj_no_pose("cube1")])
        filter_pose_outliers(scene, rf)
        assert scene.objects[0].attributes["pose_in_workspace"] is None
        assert scene.objects[0].attributes["pose_suspicious"] is False


# ---------------------------------------------------------------------------
# master function
# ---------------------------------------------------------------------------


class TestComputeScenePredicates:
    def test_all_enriched(self):
        rf = SceneReferenceFrame(
            table_x_min=-0.4, table_x_max=0.4,
            table_y_min=-0.4, table_y_max=0.4,
            table_z_surface=0.8,
            robot_base_x=0.5, robot_base_y=0.0, robot_base_z=1.35,
            edge_threshold=0.10, workspace_padding=0.30,
        )
        scene = SceneInfo(objects=[
            _make_obj("cube1", x=0.0, y=0.0, z=0.825),  # on table, center
            _make_obj("cube2", x=-0.35, y=-0.35, z=0.90),  # stacked, near front-left edge
            _make_obj_no_pose("cube3"),  # no pose
        ])
        scene = compute_scene_predicates(scene, rf)

        # cube1: on table, center, in workspace
        o1 = scene.objects[0]
        assert o1.attributes["on_table"] is True
        assert o1.attributes["near_edges"] == []
        assert o1.attributes["pose_in_workspace"] is True
        assert o1.attributes["robot_distance"] is not None

        # cube2: stacked, near front and left edges
        o2 = scene.objects[1]
        assert o2.attributes["on_table"] is False
        assert "left_edge" in o2.attributes["near_edges"]
        assert "front_edge" in o2.attributes["near_edges"]
        assert o2.attributes["pose_in_workspace"] is True

        # cube3: no pose
        o3 = scene.objects[2]
        assert o3.attributes["on_table"] is None
        assert o3.attributes["robot_distance"] is None
        assert o3.attributes["pose_in_workspace"] is None
