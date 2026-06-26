"""Tests for inspect_robosuite_world_geometry.py — uses mocks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class MockSim:
    """Minimal mock of robosuite sim with enough state for inspection."""

    def __init__(self):
        self.model = MockModel()
        self.data = MockData()


class MockModel:
    nbody = 5
    body_names = np.array(["world", "table", "cubeA_main", "cubeB_main", "robot0_base"], dtype=object)
    ngeom = 3
    ncam = 3

    @staticmethod
    def body_name2id(name):
        mapping = {"table": 1, "cubeA_main": 2, "cubeB_main": 3, "world": 0}
        if name in mapping:
            return mapping[name]
        raise ValueError(f"Unknown body: {name}")

    @staticmethod
    def geom_id2name(i):
        return {0: "table_geom", 1: "cubeA_geom", 2: "cubeB_geom"}.get(i)

    @staticmethod
    def camera_id2name(i):
        return {0: "agentview", 1: "frontview", 2: "birdview"}.get(i)

    @staticmethod
    def camera_name2id(name):
        mapping = {"agentview": 0, "frontview": 1, "birdview": 2}
        if name in mapping:
            return mapping[name]
        raise ValueError(f"Unknown camera: {name}")

    def __init__(self):
        self.geom_bodyid = np.array([1, 2, 3])
        self.geom_size = np.array([[0.5, 0.5, 0.025], [0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
        self.geom_type = np.array([6, 6, 6])
        self.cam_fovy = np.array([45.0, 45.0, 45.0])


class MockData:
    body_xpos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.8],
        [0.1, 0.05, 0.82],
        [-0.05, -0.03, 0.82],
        [0.0, 0.0, 0.0],
    ], dtype=np.float64)

    geom_xpos = np.array([
        [0.0, 0.0, 0.8],
        [0.1, 0.05, 0.82],
        [-0.05, -0.03, 0.82],
    ], dtype=np.float64)

    cam_xpos = np.array([
        [0.5, 0.0, 1.35],
        [1.6, 0.0, 1.45],
        [-0.2, 0.0, 3.0],
    ], dtype=np.float64)

    cam_xmat = np.array([
        [0.0, -0.70614784, 0.70806442, 1.0, 0.0, 0.0, 0.0, 0.70806442, 0.70614784],
        [0.0, -0.25817452, 0.96609829, 1.0, 0.0, 0.0, 0.0, 0.96609829, 0.25817452],
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


class TestInspectWorldGeometry:
    """Tests for inspect_world_geometry using mocked environment."""

    def test_output_is_valid_json(self):
        """Output file is valid JSON with expected top-level keys."""
        from vlm_nsm.experiments.inspect_robosuite_world_geometry import inspect_world_geometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_output.json")

            with mock.patch(
                "vlm_nsm.experiments.inspect_robosuite_world_geometry.RobosuiteEnvWrapper"
            ) as mock_env_class:
                mock_env = mock.MagicMock()
                mock_env.env.sim = MockSim()
                mock_env.cube_names = ["cube1", "cube2"]
                mock_env._cube_to_body = {"cube1": "cubeA_main", "cube2": "cubeB_main"}
                mock_env_class.return_value = mock_env

                result = inspect_world_geometry(output_path)

            assert "table" in result
            assert "cubes" in result
            assert "cameras" in result
            assert "all_body_names" in result
            assert "summary" in result

            with open(output_path) as f:
                loaded = json.load(f)
            assert loaded["table"]["position"]["z"] == 0.8
            assert loaded["summary"]["table_z_from_body"] == 0.8

    def test_camera_positions_extracted(self):
        """Camera positions from cam_xpos are present."""
        from vlm_nsm.experiments.inspect_robosuite_world_geometry import inspect_world_geometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_output.json")

            with mock.patch(
                "vlm_nsm.experiments.inspect_robosuite_world_geometry.RobosuiteEnvWrapper"
            ) as mock_env_class:
                mock_env = mock.MagicMock()
                mock_env.env.sim = MockSim()
                mock_env.cube_names = ["cube1", "cube2"]
                mock_env._cube_to_body = {"cube1": "cubeA_main", "cube2": "cubeB_main"}
                mock_env_class.return_value = mock_env

                result = inspect_world_geometry(output_path)

            assert "agentview" in result["cameras"]
            agentview = result["cameras"]["agentview"]
            assert agentview["position"]["x"] == 0.5
            assert agentview["position"]["y"] == 0.0
            assert agentview["position"]["z"] == 1.35
            assert "T_camera_world" in agentview
            assert len(agentview["T_camera_world"]) == 4

    def test_cube_positions_extracted(self):
        """Cube body positions match sim state."""
        from vlm_nsm.experiments.inspect_robosuite_world_geometry import inspect_world_geometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_output.json")

            with mock.patch(
                "vlm_nsm.experiments.inspect_robosuite_world_geometry.RobosuiteEnvWrapper"
            ) as mock_env_class:
                mock_env = mock.MagicMock()
                mock_env.env.sim = MockSim()
                mock_env.cube_names = ["cube1", "cube2"]
                mock_env._cube_to_body = {"cube1": "cubeA_main", "cube2": "cubeB_main"}
                mock_env_class.return_value = mock_env

                result = inspect_world_geometry(output_path)

            assert "cube1" in result["cubes"]
            assert result["cubes"]["cube1"]["position"]["x"] == 0.1
            assert result["cubes"]["cube2"]["position"]["x"] == -0.05

    def test_unknown_camera_has_error(self):
        """Non-existent camera gets an error entry."""
        from vlm_nsm.experiments.inspect_robosuite_world_geometry import inspect_world_geometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "test_output.json")

            with mock.patch(
                "vlm_nsm.experiments.inspect_robosuite_world_geometry.RobosuiteEnvWrapper"
            ) as mock_env_class:
                mock_env = mock.MagicMock()
                mock_env.env.sim = MockSim()
                mock_env.cube_names = ["cube1", "cube2"]
                mock_env._cube_to_body = {"cube1": "cubeA_main", "cube2": "cubeB_main"}
                mock_env_class.return_value = mock_env

                result = inspect_world_geometry(output_path)

            # robot0_eye_in_hand not in mock camera_name2id
            assert "error" in result["cameras"]["robot0_eye_in_hand"]
