"""Tests for dual_camera_triangulation.py — pure geometry, no GPU required."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_nsm.perception.camera_geometry import (
    pixel_to_camera_ray,
    transform_ray_to_world,
    triangulate_rays,
)
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_camera_info(K, T_camera_world, width=512, height=512, name="agentview"):
    from vlm_nsm.perception.robosuite_camera_utils import CameraInfo

    return CameraInfo(
        K=np.asarray(K, dtype=np.float64),
        T_camera_world=np.asarray(T_camera_world, dtype=np.float64),
        width=width,
        height=height,
        camera_name=name,
    )


# Agentview calibration (from generated/configs/camera_calibration_agentview_512.json)
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

# Frontview calibration (rough: position (1.6, 0.0, 1.45), 45° FOV)
# Camera looks toward the table centre (~0, 0, 0) from (1.6, 0, 1.45).
# Building a plausible T: camera R rotates world to camera frame.
# For testing we build the exact T by pointing the camera at (0, 0, 0) from (1.6, 0, 1.45).
def _look_at_T(eye, target, up):
    """Build a 4x4 camera-to-world transform (robosuite convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    new_up = np.cross(forward, right)
    R = np.column_stack([right, new_up, -forward])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


FRONTVIEW_T = _look_at_T(
    eye=(1.6, 0.0, 1.45),
    target=(0.0, 0.0, 0.0),
    up=(0.0, 0.0, 1.0),
)


# ---------------------------------------------------------------------------
# TestTriangulateRays
# ---------------------------------------------------------------------------

class TestTriangulateRays:
    """Unit tests for the pure-geometry triangulate_rays function."""

    def test_perfect_intersection(self):
        """Two rays that intersect exactly at a known point."""
        # Ray 1: from (0, 0, 1) pointing toward (1, 2, 0)
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 2.0, -1.0])
        d1 /= np.linalg.norm(d1)

        # Ray 2: from (2, 0, 1) pointing toward (1, 2, 0)
        o2 = np.array([2.0, 0.0, 1.0])
        d2 = np.array([-1.0, 2.0, -1.0])
        d2 /= np.linalg.norm(d2)

        result = triangulate_rays(o1, d1, o2, d2)
        assert result is not None
        # Both rays pass through (1, 2, 0) so the result should be (1, 2, 0)
        assert np.allclose(result, [1.0, 2.0, 0.0], atol=1e-10)

    def test_skew_lines_small_offset(self):
        """Two rays that don't quite intersect — midpoint should be between them."""
        # Ray 1: from (0, 0, 1) pointing almost to (1, 2, 0) but slightly off
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 2.001, -1.0])
        d1 /= np.linalg.norm(d1)

        # Ray 2: from (2, 0, 1) pointing almost to (1, 2, 0) but slightly off
        o2 = np.array([2.0, 0.0, 1.0])
        d2 = np.array([-1.0, 1.999, -1.0])
        d2 /= np.linalg.norm(d2)

        result = triangulate_rays(o1, d1, o2, d2)
        assert result is not None
        # Should be close to (1, 2, 0)
        assert abs(result[0] - 1.0) < 0.01
        assert abs(result[1] - 2.0) < 0.01
        assert abs(result[2] - 0.0) < 0.01

        # Closest points should be distinct but close
        w = o1 - o2
        b = np.dot(d1, d2)
        d = np.dot(d1, w)
        e_val = np.dot(d2, w)
        denom = 1.0 - b * b
        t1 = (b * e_val - d) / denom
        t2 = (e_val - b * d) / denom
        p1 = o1 + t1 * d1
        p2 = o2 + t2 * d2
        residual = np.linalg.norm(p1 - p2)
        assert residual > 0  # rays don't perfectly intersect
        assert residual < 0.01  # but they're very close

    def test_parallel_rays(self):
        """Parallel rays should return None."""
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 0.0, 0.0])
        o2 = np.array([0.0, 1.0, 1.0])
        d2 = np.array([1.0, 0.0, 0.0])

        result = triangulate_rays(o1, d1, o2, d2)
        assert result is None

    def test_near_parallel_rays(self):
        """Very-nearly-parallel rays (denom < eps) should return None."""
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 0.0, 0.0])
        o2 = np.array([0.0, 1.0, 1.0])
        # Almost parallel: d2 ≈ d1 with a tiny perturbation
        d2 = np.array([1.0, 1e-13, 0.0])
        d2 /= np.linalg.norm(d2)

        # Default eps=1e-12: |1 - dot(d1,d2)| < 1e-12 should trigger
        result = triangulate_rays(o1, d1, o2, d2)
        assert result is None

    def test_behind_camera(self):
        """Rays whose closest approach occurs behind one camera origin (t < 0)."""
        # Both rays point away from each other
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 0.0, 0.0])
        o2 = np.array([2.0, 0.0, 1.0])
        d2 = np.array([1.0, 0.0, 0.0])

        result = triangulate_rays(o1, d1, o2, d2)
        assert result is None

    def test_identity_agentview_frontview(self):
        """Known camera positions: agentview at (0.5,0,1.35), frontview at (1.6,0,1.45),
        both looking at a table-point (0.1, 0.0, 0.0). Triangulated point should
        be within 1 mm of the true point."""
        # Build rays from both cameras looking at (0.1, 0.0, 0.0)
        target = np.array([0.1, 0.0, 0.0])

        # Agentview position
        agentview_pos = AGENTVIEW_T[:3, 3]  # (0.5, 0.0, 1.35)
        dir1 = target - agentview_pos
        dir1 /= np.linalg.norm(dir1)

        # Frontview position
        frontview_pos = FRONTVIEW_T[:3, 3]  # (1.6, 0.0, 1.45)
        dir2 = target - frontview_pos
        dir2 /= np.linalg.norm(dir2)

        result = triangulate_rays(agentview_pos, dir1, frontview_pos, dir2)
        assert result is not None
        assert np.allclose(result, target, atol=1e-3)  # sub-mm for perfect rays

    def test_result_is_3_vector(self):
        """Sanity check: result is a 3-element float64 array."""
        o1 = np.array([0.0, 0.0, 1.0])
        d1 = np.array([1.0, 2.0, -1.0])
        d1 /= np.linalg.norm(d1)
        o2 = np.array([2.0, 0.0, 1.0])
        d2 = np.array([-1.0, 2.0, -1.0])
        d2 /= np.linalg.norm(d2)

        result = triangulate_rays(o1, d1, o2, d2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# TestTriangulateScene
# ---------------------------------------------------------------------------

def _world_point_to_pixel(point_world, T_camera_world, K):
    """Forward-project a world point onto the camera image plane.

    Returns ``(u, v)`` pixel coordinates, or None if behind the camera.
    """
    R = T_camera_world[:3, :3]
    t = T_camera_world[:3, 3]
    p_cam = R.T @ (point_world - t)
    if p_cam[2] >= 0:
        return None  # behind or on camera plane
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * p_cam[0] / (-p_cam[2]) + cx
    v = cy - fy * p_cam[1] / (-p_cam[2])
    return (u, v)


def _make_detected_obj(label, color, u, v, bbox_half=16.0, confidence=0.9):
    """Build a DetectedObject with a synthetic bbox whose *center*
    pixel lands at (u, v), matching the triangulation pixel extraction."""
    return DetectedObject(
        label=label,
        category="block",
        color=color,
        pose=None,
        confidence=confidence,
        attributes={
            "bbox": {
                "x1": u - bbox_half,
                "y1": v - bbox_half,
                "x2": u + bbox_half,
                "y2": v + bbox_half,
            },
            "center_px": {"x": u, "y": v},
            "center_norm": {"x": u / 512.0, "y": v / 512.0},
            "bbox_source": "detector",
            "position_source": "unknown_rgb_detector_3d",
        },
    )


class TestTriangulateScene:
    """Integration tests for triangulate_scene with two cameras."""

    @staticmethod
    def _make_agentview_cam():
        return _make_camera_info(AGENTVIEW_K, AGENTVIEW_T, name="agentview")

    @staticmethod
    def _make_frontview_cam():
        return _make_camera_info(
            AGENTVIEW_K,  # same intrinsics (both 512×512, 45° FOV)
            FRONTVIEW_T,
            name="frontview",
        )

    def test_two_matched_objects(self):
        """Two objects visible in both cameras — both get triangulated poses."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        # True world positions
        true_cube1 = np.array([0.1, 0.0, 0.0])
        true_cube2 = np.array([-0.1, 0.0, 0.0])

        # Project to both cameras
        p1_c1 = _world_point_to_pixel(true_cube1, cam1.T_camera_world, cam1.K)
        p1_c2 = _world_point_to_pixel(true_cube1, cam2.T_camera_world, cam2.K)
        p2_c1 = _world_point_to_pixel(true_cube2, cam1.T_camera_world, cam1.K)
        p2_c2 = _world_point_to_pixel(true_cube2, cam2.T_camera_world, cam2.K)

        assert all(p is not None for p in [p1_c1, p1_c2, p2_c1, p2_c2]), \
            "Test points must be visible in both cameras"

        scene1 = SceneInfo(
            objects=[
                _make_detected_obj("cube1", "red", *p1_c1),
                _make_detected_obj("cube2", "green", *p2_c1),
            ],
            source="rgb_detector",
        )
        scene2 = SceneInfo(
            objects=[
                _make_detected_obj("cube1", "red", *p1_c2),
                _make_detected_obj("cube2", "green", *p2_c2),
            ],
            source="rgb_detector",
        )

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2)

        # Both objects should have valid poses
        assert len(result.objects) == 2
        for obj, true_pos in zip(result.objects, [true_cube1, true_cube2]):
            assert obj.pose is not None, f"{obj.label} should have a pose"
            assert obj.attributes["position_source"] == "rgb_dual_camera_triangulation"
            actual = np.array([obj.pose["x"], obj.pose["y"], obj.pose["z"]])
            expected_xy = true_pos[:2]
            actual_xy = actual[:2]
            assert np.allclose(actual_xy, expected_xy, atol=1e-3), \
                f"{obj.label}: expected xy {expected_xy}, got {actual_xy}"
            # Z should be table_z + object_height/2 = 0.025
            assert obj.pose["z"] == pytest.approx(0.025)

    def test_mismatched_objects(self):
        """One object in scene1 only, a different one in scene2 — no matches."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        true_c1 = np.array([0.1, 0.0, 0.0])
        true_c2 = np.array([-0.1, 0.0, 0.0])

        p1 = _world_point_to_pixel(true_c1, cam1.T_camera_world, cam1.K)
        p2 = _world_point_to_pixel(true_c2, cam2.T_camera_world, cam2.K)
        assert p1 is not None and p2 is not None

        # scene1 has only cube1, scene2 has only cube2 (different labels)
        scene1 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p1)])
        scene2 = SceneInfo(objects=[_make_detected_obj("cube2", "green", *p2)])

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2)

        assert len(result.objects) == 1  # only scene1 objects
        obj = result.objects[0]
        assert obj.pose is None
        assert obj.attributes["position_source"] == "triangulation_failed"
        assert obj.attributes["failure_reason"] == "no_match"

    def test_metadata_stored(self):
        """Triangulated objects carry all expected metadata keys."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        true_pos = np.array([0.05, -0.05, 0.0])
        p1 = _world_point_to_pixel(true_pos, cam1.T_camera_world, cam1.K)
        p2 = _world_point_to_pixel(true_pos, cam2.T_camera_world, cam2.K)
        assert p1 is not None and p2 is not None

        scene1 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p1, confidence=0.85)])
        scene2 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p2, confidence=0.72)])

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2)
        obj = result.objects[0]

        assert obj.pose is not None
        assert obj.attributes["position_source"] == "rgb_dual_camera_triangulation"
        assert obj.attributes["pose_confidence"] == pytest.approx(0.72)  # min
        assert "triangulation_residual" in obj.attributes
        assert obj.attributes["triangulation_residual"] < 1e-6  # perfect rays
        assert obj.attributes["triangulation_source"] == "agentview+frontview"
        assert obj.attributes["camera1_pixel"] == [p1[0], p1[1]]
        assert obj.attributes["camera2_pixel"] == [p2[0], p2[1]]
        assert "pose_uncertainty" in obj.attributes

    def test_color_fallback_matching(self):
        """Objects with different labels but matching colors are matched."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        true_pos = np.array([0.0, 0.0, 0.0])
        p1 = _world_point_to_pixel(true_pos, cam1.T_camera_world, cam1.K)
        p2 = _world_point_to_pixel(true_pos, cam2.T_camera_world, cam2.K)
        assert p1 is not None and p2 is not None

        # Different labels, same color (simulates label drift between camera views)
        scene1 = SceneInfo(objects=[_make_detected_obj("cube_a", "red", *p1)])
        scene2 = SceneInfo(objects=[_make_detected_obj("cube_b", "red", *p2)])

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2)
        assert len(result.objects) == 1
        obj = result.objects[0]
        assert obj.pose is not None
        assert obj.attributes["position_source"] == "rgb_dual_camera_triangulation"

    def test_pose_z_is_table_z_plus_half_height(self):
        """Pose Z = table_z + object_height/2."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        true_pos = np.array([0.0, 0.1, 0.0])
        p1 = _world_point_to_pixel(true_pos, cam1.T_camera_world, cam1.K)
        p2 = _world_point_to_pixel(true_pos, cam2.T_camera_world, cam2.K)
        assert p1 is not None and p2 is not None

        scene1 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p1)])
        scene2 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p2)])

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2,
                                    table_z=0.0, object_height=0.06)
        assert result.objects[0].pose["z"] == pytest.approx(0.03)  # 0 + 0.06/2

    def test_object_without_bbox_skipped(self):
        """Object with no bbox/center_px/center_norm gets failure reason."""
        cam1 = self._make_agentview_cam()
        cam2 = self._make_frontview_cam()

        obj_no_bbox = DetectedObject(
            label="cube1", category="block", color="red", pose=None,
            confidence=0.9,
            attributes={"position_source": "unknown_rgb_detector_3d"},
        )
        # scene2 has a valid detection so matching succeeds, but scene1 obj lacks pixel
        true_pos = np.array([0.0, 0.0, 0.0])
        p2 = _world_point_to_pixel(true_pos, cam2.T_camera_world, cam2.K)
        assert p2 is not None

        scene1 = SceneInfo(objects=[obj_no_bbox])
        scene2 = SceneInfo(objects=[_make_detected_obj("cube1", "red", *p2)])

        from vlm_nsm.perception.dual_camera_triangulation import triangulate_scene

        result = triangulate_scene(scene1, scene2, cam1, cam2)
        obj = result.objects[0]
        assert obj.pose is None
        assert obj.attributes["position_source"] == "triangulation_failed"
        assert obj.attributes["failure_reason"] == "no_pixel"
