"""Tests for rgb_table_projection.py — no GPU required.

Uses synthetic camera geometry and fake SceneInfo.
"""

from __future__ import annotations

import numpy as np
import pytest

from vlm_nsm.perception.data_types import DetectedObject, SceneInfo
from vlm_nsm.perception.camera_geometry import intrinsic_from_fov
from vlm_nsm.perception.robosuite_camera_utils import CameraInfo
from vlm_nsm.perception.rgb_table_projection import (
    add_table_plane_pose_estimates,
    get_projection_pixel_from_bbox,
)


def _make_camera_info(width=512, height=512) -> CameraInfo:
    """Build a synthetic camera looking down from above."""
    K = intrinsic_from_fov(45.0, width, height)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = [0.0, 0.0, 1.0]  # camera at (0, 0, 1) looking down
    return CameraInfo(K=K, T_camera_world=T, width=width, height=height, camera_name="test")


# ── Tests for get_projection_pixel_from_bbox ──────────────────────────────


class TestGetProjectionPixelFromBbox:
    def test_center_mode(self):
        u, v = get_projection_pixel_from_bbox({"x1": 100, "y1": 200, "x2": 300, "y2": 400}, mode="center")
        assert u == pytest.approx(200.0)
        assert v == pytest.approx(300.0)

    def test_bottom_center_mode(self):
        u, v = get_projection_pixel_from_bbox({"x1": 100, "y1": 200, "x2": 300, "y2": 400}, mode="bottom_center")
        assert u == pytest.approx(200.0)
        assert v == pytest.approx(400.0)  # y2

    def test_weighted_bottom_mode(self):
        u, v = get_projection_pixel_from_bbox({"x1": 100, "y1": 200, "x2": 300, "y2": 400}, mode="weighted_bottom")
        assert u == pytest.approx(200.0)
        assert v == pytest.approx(0.25 * 200 + 0.75 * 400)  # 350.0

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown projection_point mode"):
            get_projection_pixel_from_bbox({"x1": 0, "y1": 0, "x2": 10, "y2": 10}, mode="invalid")

    def test_accepts_tuple(self):
        u, v = get_projection_pixel_from_bbox((100, 200, 300, 400), mode="center")
        assert u == pytest.approx(200.0)
        assert v == pytest.approx(300.0)

    def test_accepts_list(self):
        u, v = get_projection_pixel_from_bbox([100, 200, 300, 400], mode="bottom_center")
        assert u == pytest.approx(200.0)
        assert v == pytest.approx(400.0)

    def test_default_mode_is_center(self):
        u, v = get_projection_pixel_from_bbox({"x1": 0, "y1": 0, "x2": 10, "y2": 10})
        assert u == pytest.approx(5.0)
        assert v == pytest.approx(5.0)

    def test_bbox_dict_with_extra_keys(self):
        """Extra keys in dict don't affect the result."""
        u, v = get_projection_pixel_from_bbox(
            {"x1": 50, "y1": 60, "x2": 150, "y2": 160, "confidence": 0.9},
            mode="weighted_bottom",
        )
        assert u == pytest.approx(100.0)
        assert v == pytest.approx(0.25 * 60 + 0.75 * 160)


# ── Tests for add_table_plane_pose_estimates ──────────────────────────────


class TestAddTablePlanePoseEstimates:
    def test_project_from_center_px(self):
        """Object with center_px gets a valid pose estimate."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1",
            category="block",
            color="red",
            pose=None,
            confidence=0.9,
            attributes={
                "center_px": {"x": 255.5, "y": 255.5},
                "position_source": "unknown_rgb_detector_3d",
            },
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0, object_height=0.05)

        assert scene.objects[0].pose is not None
        assert scene.objects[0].pose["z"] == pytest.approx(0.025)  # table_z + h/2
        assert scene.objects[0].attributes["position_source"] == "rgb_bbox_table_plane_projection"
        assert "pose_confidence" in scene.objects[0].attributes
        assert scene.objects[0].attributes["pose_confidence"] == 0.9

    def test_project_from_center_norm(self):
        """Object with only center_norm also works."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1",
            category="block",
            color="red",
            pose=None,
            confidence=0.85,
            attributes={
                "center_norm": {"x": 0.5, "y": 0.5},
            },
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0, object_height=0.05)
        assert scene.objects[0].pose is not None
        assert scene.objects[0].pose["z"] == pytest.approx(0.025)

    def test_no_center_info_fails(self):
        """Object without center_px or center_norm gets projection_failed."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1",
            category="block",
            color="red",
            pose=None,
            attributes={},
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0)
        assert scene.objects[0].pose is None
        assert scene.objects[0].attributes["position_source"] == "projection_failed"

    def test_multiple_objects(self):
        """Multiple objects each get independent pose estimates."""
        camera = _make_camera_info()
        objs = [
            DetectedObject(
                label="cube1", category="block", color="red",
                pose=None, confidence=0.9,
                attributes={"center_px": {"x": 100, "y": 200}},
            ),
            DetectedObject(
                label="cube2", category="block", color="green",
                pose=None, confidence=0.8,
                attributes={"center_px": {"x": 400, "y": 300}},
            ),
        ]
        scene = SceneInfo(objects=objs, source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0, object_height=0.05)
        assert scene.objects[0].pose is not None
        assert scene.objects[1].pose is not None
        assert scene.objects[0].pose["z"] == pytest.approx(0.025)
        assert scene.objects[1].pose["z"] == pytest.approx(0.025)

    def test_object_height_affects_z(self):
        """Object height shifts the pose Z up by half the height."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None, confidence=0.9,
            attributes={"center_px": {"x": 255.5, "y": 255.5}},
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0, object_height=0.10)
        assert scene.objects[0].pose["z"] == pytest.approx(0.05)  # 0 + 0.10/2

    def test_pose_uncertainty_stored(self):
        """Pose uncertainty metadata is attached."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None, confidence=0.9,
            attributes={"center_px": {"x": 255.5, "y": 255.5}},
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0)
        uncertainty = scene.objects[0].attributes.get("pose_uncertainty")
        assert uncertainty is not None
        assert "x" in uncertainty
        assert "y" in uncertainty
        assert "z" in uncertainty

    def test_projection_point_mode_stored(self):
        """projection_point_mode and projection_pixel are stored."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None, confidence=0.9,
            attributes={
                "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 400},
            },
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0,
                                                projection_point="bottom_center")
        assert scene.objects[0].attributes["projection_point_mode"] == "bottom_center"
        assert scene.objects[0].attributes["projection_pixel"] == [250.0, 400.0]

    def test_projection_point_default_center(self):
        """Default projection_point is center."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None, confidence=0.9,
            attributes={
                "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 400},
            },
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0)
        assert scene.objects[0].attributes["projection_point_mode"] == "center"
        assert scene.objects[0].attributes["projection_pixel"] == [250.0, 300.0]

    def test_bbox_preferred_over_center_px(self):
        """Bbox is preferred over center_px when both are present."""
        camera = _make_camera_info()
        obj = DetectedObject(
            label="cube1", category="block", color="red",
            pose=None, confidence=0.9,
            attributes={
                "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 400},
                "center_px": {"x": 999, "y": 999},  # should be ignored
            },
        )
        scene = SceneInfo(objects=[obj], source="rgb_detector")
        scene = add_table_plane_pose_estimates(scene, camera, table_z=0.0)
        # Should use bbox, not center_px
        assert scene.objects[0].attributes["projection_pixel"] == [250.0, 300.0]
