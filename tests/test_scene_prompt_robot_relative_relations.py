"""Tests for pairwise robot-relative relations in scene prompt — no GPU required."""

from __future__ import annotations

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import (
    _compute_robot_pairwise_relations,
    build_scene_description,
)


class TestRobotPairwiseRelations:
    def test_two_objects_produces_pairwise(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                attributes={"robot_distance": 0.30, "on_table": True},
            ),
            DetectedObject(
                label="cube2", category="block",
                attributes={"robot_distance": 0.60, "on_table": True},
            ),
        ])
        rels = _compute_robot_pairwise_relations(scene)
        assert "cube1" in rels
        assert "cube2" in rels
        assert f"nearer_to_robot_than cube2" in rels["cube1"]
        assert f"farther_from_robot_than cube1" in rels["cube2"]

    def test_three_objects_produces_all_pairs(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                attributes={"robot_distance": 0.30},
            ),
            DetectedObject(
                label="cube2", category="block",
                attributes={"robot_distance": 0.60},
            ),
            DetectedObject(
                label="cube3", category="block",
                attributes={"robot_distance": 0.90},
            ),
        ])
        rels = _compute_robot_pairwise_relations(scene)
        assert len(rels) == 3
        # cube1 is nearest to all
        assert f"nearer_to_robot_than cube2" in rels["cube1"]
        assert f"nearer_to_robot_than cube3" in rels["cube1"]
        # cube3 is farthest from all
        assert f"farther_from_robot_than cube1" in rels["cube3"]
        assert f"farther_from_robot_than cube2" in rels["cube3"]
        # cube2: nearer than cube3, farther than cube1
        assert f"nearer_to_robot_than cube3" in rels["cube2"]
        assert f"farther_from_robot_than cube1" in rels["cube2"]

    def test_single_object_no_pairwise(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                attributes={"robot_distance": 0.30},
            ),
        ])
        rels = _compute_robot_pairwise_relations(scene)
        assert rels == {}

    def test_no_distances_no_pairwise(self):
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", attributes={}),
            DetectedObject(label="cube2", category="block", attributes={}),
        ])
        rels = _compute_robot_pairwise_relations(scene)
        assert rels == {}

    def test_equal_distances_skipped(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                attributes={"robot_distance": 0.50},
            ),
            DetectedObject(
                label="cube2", category="block",
                attributes={"robot_distance": 0.505},
            ),
        ])
        rels = _compute_robot_pairwise_relations(scene)
        assert rels == {}

    def test_pairwise_rendered_in_description(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block", color="red",
                pose={"x": 0.0, "y": -0.3, "z": 0.825},
                attributes={
                    "robot_distance": 0.30,
                    "on_table": True,
                    "near_edges": [],
                    "position_source": "rgb_dual_camera_triangulation",
                },
            ),
            DetectedObject(
                label="cube2", category="block", color="green",
                pose={"x": 0.2, "y": 0.0, "z": 0.825},
                attributes={
                    "robot_distance": 0.60,
                    "on_table": True,
                    "near_edges": [],
                    "position_source": "rgb_dual_camera_triangulation",
                },
            ),
        ])
        desc = build_scene_description(scene)
        assert f"nearer_to_robot_than cube2" in desc
        assert f"farther_from_robot_than cube1" in desc

    def test_robot_ordering_in_table_layout(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube2", category="block", color="green",
                pose={"x": 0.2, "y": 0.0, "z": 0.825},
                attributes={
                    "robot_distance": 0.60,
                    "on_table": True,
                    "near_edges": [],
                    "position_source": "rgb_dual_camera_triangulation",
                },
            ),
            DetectedObject(
                label="cube1", category="block", color="red",
                pose={"x": 0.0, "y": -0.3, "z": 0.825},
                attributes={
                    "robot_distance": 0.30,
                    "on_table": True,
                    "near_edges": [],
                    "position_source": "rgb_dual_camera_triangulation",
                },
            ),
        ])
        desc = build_scene_description(scene)
        assert "Robot-relative (near to far):" in desc
        assert "cube1 (0.30m) < cube2 (0.60m)" in desc

    def test_ambiguous_reference_guidance(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                attributes={"on_table": True, "robot_distance": 0.30},
            ),
        ])
        desc = build_scene_description(scene)
        assert "shared equally by multiple objects" in desc
        assert "MUST output status=\"ungroundable\"" in desc
        assert "first you see" in desc
        assert "lighter/darker" in desc

    def test_no_predicates_no_pairwise_in_description(self):
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ])
        desc = build_scene_description(scene)
        assert "nearer_to_robot_than" not in desc
        assert "farther_from_robot_than" not in desc
        assert "Robot-relative (near to far):" not in desc
