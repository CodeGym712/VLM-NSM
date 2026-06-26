"""Tests for predicate-rich scene prompt rendering — no GPU required."""

from __future__ import annotations

import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.scene_prompt import build_scene_description
from vlm_nsm.perception.scene_context import SceneReferenceFrame
from vlm_nsm.perception.scene_predicates import compute_scene_predicates


def _make_scene_with_predicates() -> SceneInfo:
    """Build a SceneInfo with predicate attributes already set."""
    scene = SceneInfo(objects=[
        DetectedObject(
            label="cube1", category="block", color="red",
            pose={"x": 0.0, "y": -0.3, "z": 0.825},
            attributes={
                "on_table": True,
                "near_edges": ["front_edge"],
                "robot_distance": 0.3,
                "robot_direction": ["robot-center-x", "toward-robot"],
                "pose_in_workspace": True,
                "pose_suspicious": False,
                "position_source": "rgb_dual_camera_triangulation",
                "spatial": {
                    "world_relations": {
                        "cube2": ["right_of", "front_of"],
                    },
                },
            },
        ),
        DetectedObject(
            label="cube2", category="block", color="green",
            pose={"x": 0.2, "y": 0.0, "z": 0.825},
            attributes={
                "on_table": True,
                "near_edges": [],
                "robot_distance": 0.35,
                "robot_direction": ["robot-right", "robot-center-y"],
                "pose_in_workspace": True,
                "pose_suspicious": False,
                "position_source": "rgb_dual_camera_triangulation",
                "spatial": {
                    "world_relations": {
                        "cube1": ["left_of", "behind"],
                    },
                },
            },
        ),
        DetectedObject(
            label="cube3", category="block", color="blue",
            pose={"x": -0.2, "y": 0.3, "z": 0.88},
            attributes={
                "on_table": False,
                "near_edges": ["left_edge", "back_edge"],
                "robot_distance": 0.55,
                "robot_direction": ["robot-left", "away-from-robot"],
                "pose_in_workspace": True,
                "pose_suspicious": False,
                "position_source": "rgb_dual_camera_triangulation",
                "spatial": {
                    "world_relations": {},
                },
            },
        ),
    ])
    return scene


class TestPredicateScenePrompt:
    def test_on_table_rendered(self):
        desc = build_scene_description(_make_scene_with_predicates())
        assert "on table" in desc
        assert "stacked/elevated" in desc

    def test_edge_proximity_rendered(self):
        desc = build_scene_description(_make_scene_with_predicates())
        assert "near front_edge" in desc
        assert "near left_edge, back_edge" in desc

    def test_robot_distance_rendered(self):
        desc = build_scene_description(_make_scene_with_predicates())
        assert "0.30m from robot" in desc

    def test_table_layout_section(self):
        desc = build_scene_description(_make_scene_with_predicates())
        assert "Table layout:" in desc
        assert "Robot-relative (near to far):" in desc
        assert "cube1" in desc

    def test_predicate_guidance_note(self):
        desc = build_scene_description(_make_scene_with_predicates())
        assert "on-table status" in desc
        assert "table layout summary" in desc

    def test_no_predicates_no_table_layout(self):
        """Scene without predicates should not have table layout section."""
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block"),
        ])
        desc = build_scene_description(scene)
        assert "Table layout:" not in desc
        assert "on-table status" not in desc

    def test_pose_suspicious_rendered(self):
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block",
                pose={"x": -1.25, "y": 0.0, "z": 0.825},
                attributes={
                    "on_table": True,
                    "pose_in_workspace": False,
                    "pose_suspicious": True,
                    "suspicious_reason": "outside workspace",
                    "robot_distance": 1.5,
                    "near_edges": [],
                },
            ),
        ])
        desc = build_scene_description(scene)
        assert "WARNING" in desc

    def test_compute_then_render_integration(self):
        """Integration: compute predicates then render."""
        rf = SceneReferenceFrame(
            table_z_surface=0.8,
            edge_threshold=0.10,
        )
        scene = SceneInfo(objects=[
            DetectedObject(
                label="cube1", category="block", color="red",
                pose={"x": 0.0, "y": -0.35, "z": 0.825},
            ),
            DetectedObject(
                label="cube2", category="block", color="green",
                pose={"x": 0.2, "y": 0.0, "z": 0.90},
            ),
        ])
        scene = compute_scene_predicates(scene, rf)
        desc = build_scene_description(scene)
        assert "on table" in desc
        assert "stacked/elevated" in desc
        assert "near front_edge" in desc
        assert "Table layout:" in desc
