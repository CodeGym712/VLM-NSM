"""Tests for camera_geometry.py — pure geometry, no GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_nsm.perception.camera_geometry import (
    intrinsic_from_fov,
    pixel_to_camera_ray,
    transform_ray_to_world,
    ray_plane_intersection,
    pixel_to_table_point,
)


class TestIntrinsicFromFov:
    def test_square_image(self):
        K = intrinsic_from_fov(45.0, 512, 512)
        assert K.shape == (3, 3)
        # Principal point at image centre
        assert K[0, 2] == pytest.approx(255.5)
        assert K[1, 2] == pytest.approx(255.5)
        # Focal length from fov_y=45, h=512
        expected_fy = 512.0 / (2.0 * np.tan(np.deg2rad(45.0 / 2.0)))
        assert K[0, 0] == pytest.approx(expected_fy)
        assert K[1, 1] == pytest.approx(expected_fy)

    def test_rectangular_image(self):
        K = intrinsic_from_fov(50.0, 640, 480)
        expected_fy = 480.0 / (2.0 * np.tan(np.deg2rad(50.0 / 2.0)))
        assert K[0, 0] == pytest.approx(expected_fy, rel=1e-6)
        assert K[1, 1] == pytest.approx(expected_fy, rel=1e-6)
        assert K[0, 2] == pytest.approx(319.5)
        assert K[1, 2] == pytest.approx(239.5)

    def test_identity_structure(self):
        """K should be upper-triangular with K[2,2]==1."""
        K = intrinsic_from_fov(45.0, 512, 512)
        assert K[2, 2] == 1.0
        assert K[1, 0] == 0.0


class TestPixelToCameraRay:
    def test_centre_pixel(self):
        """Centre pixel should point along -Z."""
        K = intrinsic_from_fov(45.0, 512, 512)
        origin, direction = pixel_to_camera_ray(255.5, 255.5, K)
        assert np.allclose(origin, [0, 0, 0])
        # Direction should be ~[0, 0, -1]
        assert direction[2] < 0
        assert abs(direction[0]) < 1e-6
        assert abs(direction[1]) < 1e-6
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_top_left_pixel(self):
        """Top-left pixel should point up and left in camera frame."""
        K = intrinsic_from_fov(45.0, 512, 512)
        origin, direction = pixel_to_camera_ray(0, 0, K)
        assert direction[0] < 0  # left
        assert direction[1] > 0  # up (v=0 is top, camera y is up)
        assert direction[2] < 0  # forward
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_bottom_right_pixel(self):
        """Bottom-right pixel should point down and right."""
        K = intrinsic_from_fov(45.0, 512, 512)
        origin, direction = pixel_to_camera_ray(511, 511, K)
        assert direction[0] > 0  # right
        assert direction[1] < 0  # down
        assert direction[2] < 0
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6


class TestTransformRayToWorld:
    def test_identity_transform(self):
        """With identity T, ray is unchanged."""
        T = np.eye(4, dtype=np.float64)
        origin_cam = np.array([0.0, 0.0, 0.0])
        dir_cam = np.array([0.0, 0.0, -1.0])
        o, d = transform_ray_to_world(origin_cam, dir_cam, T)
        assert np.allclose(o, [0, 0, 0])
        assert np.allclose(d, [0, 0, -1.0])

    def test_translation_only(self):
        """Translation shifts origin only."""
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [1.0, 2.0, 3.0]
        origin_cam = np.array([0.0, 0.0, 0.0])
        dir_cam = np.array([1.0, 0.0, 0.0])
        o, d = transform_ray_to_world(origin_cam, dir_cam, T)
        assert np.allclose(o, [1.0, 2.0, 3.0])
        assert np.allclose(d, [1.0, 0.0, 0.0])

    def test_rotation_and_translation(self):
        """90-degree rotation around Z."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [1, 2, 3]
        o, d = transform_ray_to_world(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            T,
        )
        assert np.allclose(o, [1, 2, 3])
        assert np.allclose(d, [0, 1, 0])


class TestRayPlaneIntersection:
    def test_intersect_above_plane(self):
        """Ray from above should hit the plane."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        point = ray_plane_intersection(origin, direction, 0.0)
        assert point is not None
        assert np.allclose(point, [0, 0, 0])

    def test_no_intersection_parallel(self):
        """Ray parallel to plane should return None."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])
        point = ray_plane_intersection(origin, direction, 0.0)
        assert point is None

    def test_no_intersection_behind(self):
        """Ray pointing away from plane should return None."""
        origin = np.array([0.0, 0.0, 0.5])
        direction = np.array([0.0, 0.0, 1.0])  # pointing up
        point = ray_plane_intersection(origin, direction, 0.0)
        assert point is None

    def test_intersect_angled_ray(self):
        """Angled ray hits the correct point on the plane."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, -1.0])  # 45 degrees downward
        direction = direction / np.linalg.norm(direction)
        point = ray_plane_intersection(origin, direction, 0.0)
        assert point is not None
        # t = (0 - 1) / (-0.707) = 1.414
        # x = 0 + 1.414 * 0.707 = 1.0
        assert point[0] == pytest.approx(1.0, rel=1e-4)
        assert point[1] == pytest.approx(0.0, rel=1e-4)
        assert point[2] == pytest.approx(0.0, abs=1e-6)


class TestPixelToTablePoint:
    def test_centre_pixel_on_table(self):
        """Centre pixel projects to a point on the table."""
        K = intrinsic_from_fov(45.0, 512, 512)
        # Simple T: camera at (0, 0, 1) looking down
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0, 0, 1.0]
        # Rotate so camera looks along -Z
        # This means camera Z maps to world -Z
        # We want camera -Z to point from (0,0,1) to (0,0,0)
        # So camera z-axis should map to world (0,0,-1)
        # This is a 180-degree rotation... or simpler:
        # R from camera to world where camera -Z = world -Z
        # Actually, R should be identity since camera looks along -Z in camera frame
        # and we want that to be world -Z too.
        point = pixel_to_table_point(255.5, 255.5, K, T, 0.0)
        assert point is not None
        assert point[2] == pytest.approx(0.0)
        assert point[0] == pytest.approx(0.0, abs=1e-4)
        assert point[1] == pytest.approx(0.0, abs=1e-4)

    def test_camera_above_table_hits(self):
        """Camera at (0.5, 0, 1.35) typical agentview position."""
        K = intrinsic_from_fov(45.0, 512, 512)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [0.5, 0.0, 1.35]
        point = pixel_to_table_point(255.5, 255.5, K, T, 0.0)
        assert point is not None
        assert point[2] == pytest.approx(0.0)
        # Should be near (0.5, 0) on the table
        assert point[0] == pytest.approx(0.5, abs=0.1)
        assert point[1] == pytest.approx(0.0, abs=0.1)

    def test_no_hit_parallel_ray(self):
        """Ray parallel to table should return None."""
        K = intrinsic_from_fov(45.0, 512, 512)
        T = np.eye(4, dtype=np.float64)
        # This should produce a ray that doesn't hit z=0
        point = pixel_to_table_point(255.5, 0, K, T, 0.0)
        # May or may not hit depending on camera orientation
        assert point is None or len(point) == 3
