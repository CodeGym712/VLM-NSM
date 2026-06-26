"""Tests for scene_context.py — no GPU required."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vlm_nsm.perception.scene_context import (
    SceneReferenceFrame,
    save_reference_frame,
    load_reference_frame,
)


class TestSceneReferenceFrame:
    def test_defaults(self):
        rf = SceneReferenceFrame()
        assert rf.table_x_min == -0.4
        assert rf.table_x_max == 0.4
        assert rf.table_z_surface == 0.8
        assert rf.robot_base_x == 0.5

    def test_center_properties(self):
        rf = SceneReferenceFrame(
            table_x_min=-0.5, table_x_max=0.5,
            table_y_min=-0.3, table_y_max=0.3,
        )
        assert rf.table_center_x == 0.0
        assert rf.table_center_y == 0.0

    def test_custom_values(self):
        rf = SceneReferenceFrame(
            table_x_min=-1.0, table_x_max=1.0,
            table_y_min=-0.5, table_y_max=0.5,
            table_z_surface=0.9,
            robot_base_x=0.0, robot_base_y=0.0, robot_base_z=1.0,
            edge_threshold=0.15, workspace_padding=0.5,
        )
        assert rf.table_center_x == 0.0
        assert rf.table_center_y == 0.0
        assert rf.edge_threshold == 0.15
        assert rf.workspace_padding == 0.5


class TestReferenceFrameJSON:
    def test_round_trip(self):
        rf = SceneReferenceFrame(
            table_x_min=-0.6, table_x_max=0.6,
            table_y_min=-0.3, table_y_max=0.3,
            table_z_surface=0.85,
            robot_base_x=1.2, robot_base_y=-0.1, robot_base_z=1.4,
            edge_threshold=0.08, workspace_padding=0.25,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref_frame.json"
            save_reference_frame(path, rf)
            loaded = load_reference_frame(path)
            assert loaded.table_x_min == rf.table_x_min
            assert loaded.table_x_max == rf.table_x_max
            assert loaded.table_y_min == rf.table_y_min
            assert loaded.table_y_max == rf.table_y_max
            assert loaded.table_z_surface == rf.table_z_surface
            assert loaded.robot_base_x == rf.robot_base_x
            assert loaded.robot_base_y == rf.robot_base_y
            assert loaded.robot_base_z == rf.robot_base_z
            assert loaded.edge_threshold == rf.edge_threshold
            assert loaded.workspace_padding == rf.workspace_padding

    def test_load_defaults_from_json(self):
        """Loading a minimal JSON fills missing fields with defaults."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.json"
            path.write_text(json.dumps({
                "table_x_min": -0.3, "table_x_max": 0.3,
                "table_y_min": -0.3, "table_y_max": 0.3,
                "table_z_surface": 0.82,
                "robot_base_x": 0.6, "robot_base_y": 0.0, "robot_base_z": 1.35,
            }))
            loaded = load_reference_frame(path)
            assert loaded.table_z_surface == 0.82
            assert loaded.robot_base_x == 0.6
            # edge_threshold, workspace_padding use dataclass defaults
            assert loaded.edge_threshold == 0.10
            assert loaded.workspace_padding == 0.30
