"""Tests for forward-projection functions in camera_geometry.py.

world_to_camera_point, camera_point_to_pixel, project_world_point_to_pixel.
Pure geometry — no GPU/sim required.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_nsm.perception.camera_geometry import (
    camera_point_to_pixel,
    intrinsic_from_fov,
    project_world_point_to_pixel,
    world_to_camera_point,
)


# Shared fixtures
AGENTVIEW_K = np.array(
    [[618.0386719675123, 0.0, 255.5],
     [0.0, 618.0386719675123, 255.5],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

AGENTVIEW_T = np.array(
    [[0.0, -0.706147844353306, 0.7080644193257978, 0.5],
     [1.0, 0.0, 0.0, 0.0],
     [0.0, 0.7080644193257978, 0.706147844353306, 1.35],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# world_to_camera_point
# ---------------------------------------------------------------------------

class TestWorldToCameraPoint:
    def test_identity_transform(self):
        """With identity T, camera point equals world point."""
        T = np.eye(4, dtype=np.float64)
        p_world = np.array([1.0, 2.0, 3.0])
        p_cam = world_to_camera_point(p_world, T)
        assert np.allclose(p_cam, [1.0, 2.0, 3.0])

    def test_translation_only(self):
        """Pure translation subtracts camera position."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0.5, 0.0, 1.35]
        p_world = np.array([0.5, 0.0, 0.0])
        p_cam = world_to_camera_point(p_world, T)
        # Point at camera position in XY, but below
        assert np.allclose(p_cam, [0.0, 0.0, -1.35])

    def test_roundtrip_with_ray_transform(self):
        """Transform point to camera then back via transform_ray_to_world."""
        from vlm_nsm.perception.camera_geometry import transform_ray_to_world

        p_world = np.array([0.1, 0.05, 0.02])
        p_cam = world_to_camera_point(p_world, AGENTVIEW_T)

        # Convert to ray at that direction from origin
        origin_cam = np.zeros(3)
        direction_cam = p_cam / np.linalg.norm(p_cam)

        origin_w, direction_w = transform_ray_to_world(
            origin_cam, direction_cam, AGENTVIEW_T
        )
        # The ray should pass through the original world point
        t = np.linalg.norm(p_cam)  # distance from camera to point
        recovered = origin_w + t * direction_w
        assert np.allclose(recovered, p_world, atol=1e-10)

    def test_agentview_table_center(self):
        """World origin appears below and in front of agentview camera."""
        p_world = np.array([0.0, 0.0, 0.0])
        p_cam = world_to_camera_point(p_world, AGENTVIEW_T)
        # Camera at (0.5, 0, 1.35), looking at table centre area
        # Point should be in front of camera (z < 0)
        assert p_cam[2] < 0


# ---------------------------------------------------------------------------
# camera_point_to_pixel
# ---------------------------------------------------------------------------

class TestCameraPointToPixel:
    def test_optical_axis_point(self):
        """Point directly on optical axis maps to principal point."""
        K = intrinsic_from_fov(45.0, 512, 512)
        # Point at (0, 0, -1) in camera frame = directly ahead
        result = camera_point_to_pixel(np.array([0.0, 0.0, -1.0]), K)
        assert result is not None
        u, v = result
        assert u == pytest.approx(255.5)
        assert v == pytest.approx(255.5)

    def test_point_above_axis(self):
        """Point with positive camera Y (up) appears higher in image (lower v)."""
        K = intrinsic_from_fov(45.0, 512, 512)
        result = camera_point_to_pixel(np.array([0.0, 0.1, -1.0]), K)
        assert result is not None
        u, v = result
        assert v < 255.5  # above centre = smaller v

    def test_point_right_of_axis(self):
        """Point with positive camera X (right) appears to the right (higher u)."""
        K = intrinsic_from_fov(45.0, 512, 512)
        result = camera_point_to_pixel(np.array([0.1, 0.0, -1.0]), K)
        assert result is not None
        u, v = result
        assert u > 255.5  # right of centre = larger u

    def test_point_behind_camera(self):
        """Point at z >= 0 returns None."""
        K = intrinsic_from_fov(45.0, 512, 512)
        assert camera_point_to_pixel(np.array([0.0, 0.0, 0.0]), K) is None
        assert camera_point_to_pixel(np.array([0.0, 0.0, 1.0]), K) is None

    def test_result_is_tuple_of_floats(self):
        """Return type is tuple of two floats."""
        K = intrinsic_from_fov(45.0, 512, 512)
        result = camera_point_to_pixel(np.array([0.0, 0.0, -1.0]), K)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# project_world_point_to_pixel
# ---------------------------------------------------------------------------

class TestProjectWorldPointToPixel:
    def test_workspace_point_visible_agentview(self):
        """A point in the robot workspace should be visible from agentview
        and project within image bounds."""
        # Point roughly in front of agentview at table height
        p_world = np.array([-0.1, 0.0, 0.02])
        result = project_world_point_to_pixel(p_world, AGENTVIEW_K, AGENTVIEW_T)
        assert result is not None
        u, v = result
        assert 0 <= u < 512
        assert 0 <= v < 512

    def test_top_corner_visible(self):
        """A point above the table near workspace centre projects within bounds."""
        p_world = np.array([0.0, 0.0, 0.3])
        result = project_world_point_to_pixel(p_world, AGENTVIEW_K, AGENTVIEW_T)
        assert result is not None
        u, v = result
        assert 0 <= u < 512
        assert 0 <= v < 512

    def test_roundtrip_consistency(self):
        """Forward-project then unproject should recover the ray through the point."""
        from vlm_nsm.perception.camera_geometry import (
            pixel_to_camera_ray,
            transform_ray_to_world,
        )

        p_world = np.array([0.05, -0.03, 0.02])
        result = project_world_point_to_pixel(p_world, AGENTVIEW_K, AGENTVIEW_T)
        assert result is not None
        u, v = result

        # Unproject
        _, dir_cam = pixel_to_camera_ray(u, v, AGENTVIEW_K)
        origin_w, dir_w = transform_ray_to_world(
            np.zeros(3), dir_cam, AGENTVIEW_T
        )

        # Ray should pass near the original point
        # Find closest point on ray to p_world
        w = p_world - origin_w
        t = np.dot(w, dir_w)
        closest = origin_w + t * dir_w
        assert np.linalg.norm(closest - p_world) < 0.001

    def test_point_behind_camera_returns_none(self):
        """World point behind the camera returns None."""
        # Agentview looks roughly toward -X direction (from 0.5 toward 0)
        # A point at X > 0.5 is behind the camera
        p_behind = np.array([2.0, 0.0, 1.35])
        result = project_world_point_to_pixel(p_behind, AGENTVIEW_K, AGENTVIEW_T)
        assert result is None
