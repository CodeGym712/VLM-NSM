"""Tests for robosuite_camera_utils — mocked, no GPU.

Uses pytest mocks instead of a real Robosuite environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from vlm_nsm.perception.robosuite_camera_utils import (
    get_camera_intrinsics,
    get_camera_extrinsics,
    get_camera_info,
    CameraInfo,
)


@pytest.fixture
def mock_env():
    """Build a mock RobosuiteEnvWrapper with minimal MuJoCo camera data."""
    # ── sim.model ──
    mock_model = MagicMock()
    mock_model.camera_name2id.return_value = 2
    mock_model.ncam = 6
    mock_model.nbody = 10

    # cam_fovy[cam_id]: indexable
    cam_fovy = np.zeros(6, dtype=np.float64)
    cam_fovy[2] = 45.0
    mock_model.cam_fovy = cam_fovy

    # cam_pos[cam_id]: this is in model (static), but we use data.cam_xpos
    cam_pos = np.zeros((6, 3), dtype=np.float64)
    cam_pos[2] = [0.5, 0.0, 1.35]
    mock_model.cam_pos = cam_pos

    # ── sim.data ──
    mock_data = MagicMock()

    # cam_xpos[cam_id]: dynamic camera position
    cam_xpos = np.zeros((6, 3), dtype=np.float64)
    cam_xpos[2] = [0.5, 0.0, 1.35]
    mock_data.cam_xpos = cam_xpos

    # cam_xmat[cam_id]: 9-element row-major matrix
    cam_xmat = np.zeros((6, 9), dtype=np.float64)
    # From actual MuJoCo: [0, -0.706, 0.708, 1, 0, 0, 0, 0.708, 0.706]
    cam_xmat[2] = [0.0, -0.70614784, 0.70806442,
                   1.0, 0.0, 0.0,
                   0.0, 0.70806442, 0.70614784]
    mock_data.cam_xmat = cam_xmat

    # body_xpos for table height
    body_xpos = np.zeros((10, 3), dtype=np.float64)
    body_xpos[5] = [0.0, 0.0, 0.0]  # table body
    mock_data.body_xpos = body_xpos

    # body_names
    body_names = ["floor", "robot0_base", "robot0_shoulder",
                  "robot0_upper_elbow", "robot0_lower_elbow",
                  "table", "cubeA_main", "cubeB_main", "cabinet", "shelf"]

    # ── sim ──
    mock_sim = MagicMock()
    mock_sim.model = mock_model
    mock_sim.data = mock_data

    # model.body_names needs to be indexable
    body_names_arr = np.empty(10, dtype=object)
    for i, name in enumerate(body_names):
        body_names_arr[i] = name

    # Use PropertyMock for body_names
    type(mock_model).body_names = PropertyMock(return_value=body_names_arr)

    # ── env ──
    mock_env_obj = MagicMock()
    mock_env_obj.sim = mock_sim

    mock_env = MagicMock()
    mock_env.env = mock_env_obj

    return mock_env


class TestGetCameraIntrinsics:
    def test_returns_3x3(self, mock_env):
        K = get_camera_intrinsics(mock_env, "agentview", 512, 512)
        assert K.shape == (3, 3)
        assert K[2, 2] == 1.0

    def test_focal_length_positive(self, mock_env):
        K = get_camera_intrinsics(mock_env, "agentview", 512, 512)
        assert K[0, 0] > 0
        assert K[1, 1] > 0

    def test_principal_point_centre(self, mock_env):
        K = get_camera_intrinsics(mock_env, "agentview", 640, 480)
        assert K[0, 2] == pytest.approx(319.5)
        assert K[1, 2] == pytest.approx(239.5)


class TestGetCameraExtrinsics:
    def test_returns_4x4(self, mock_env):
        T = get_camera_extrinsics(mock_env, "agentview")
        assert T.shape == (4, 4)
        assert T[3, 3] == 1.0

    def test_translation_components(self, mock_env):
        T = get_camera_extrinsics(mock_env, "agentview")
        assert T[0, 3] == pytest.approx(0.5)
        assert T[1, 3] == pytest.approx(0.0)
        assert T[2, 3] == pytest.approx(1.35)


class TestGetCameraInfo:
    def test_returns_camera_info(self, mock_env):
        info = get_camera_info(mock_env, "agentview", 512, 512)
        assert isinstance(info, CameraInfo)
        assert info.camera_name == "agentview"
        assert info.width == 512
        assert info.height == 512
        assert info.K.shape == (3, 3)
        assert info.T_camera_world.shape == (4, 4)
