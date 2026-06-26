"""Tests for search_camera_convention.py — pure logic, no GPU required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from vlm_nsm.perception.camera_geometry import intrinsic_from_fov


class TestConventionCandidate:
    def test_build_T_row_major_cam2world_vanilla(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
        )
        cand = ConventionCandidate(
            name="test", reshape_mode="row_major",
            R_direction="cam2world", y_flip=False, z_flip=False,
        )
        xpos = np.array([1.0, 2.0, 3.0])
        xmat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)

        T = cand.build_T(xpos, xmat)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
        assert np.allclose(T[:3, :3], np.eye(3))

    def test_build_T_row_major_world2cam(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
        )
        cand = ConventionCandidate(
            name="test", reshape_mode="row_major",
            R_direction="world2cam", y_flip=False, z_flip=False,
        )
        # If cam_xmat represents world→camera, T = [R^T | pos]
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        xmat = R.flatten()  # row-major flatten
        xpos = np.array([0.5, 0.0, 1.35])

        T = cand.build_T(xpos, xmat)
        # T[:3,:3] should equal R^T
        assert np.allclose(T[:3, :3], R.T)
        assert np.allclose(T[:3, 3], xpos)

    def test_y_flip(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
        )
        cand = ConventionCandidate(
            name="test", reshape_mode="row_major",
            R_direction="cam2world", y_flip=True, z_flip=False,
        )
        xmat = np.eye(3, dtype=np.float64).flatten()
        xpos = np.zeros(3)

        T = cand.build_T(xpos, xmat)
        # Column 1 (Y axis) should be flipped
        assert T[1, 1] == -1.0  # Y column, Y row
        assert T[0, 1] == 0.0

    def test_z_flip(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
        )
        cand = ConventionCandidate(
            name="test", reshape_mode="row_major",
            R_direction="cam2world", y_flip=False, z_flip=True,
        )
        xmat = np.eye(3, dtype=np.float64).flatten()
        xpos = np.zeros(3)

        T = cand.build_T(xpos, xmat)
        # Column 2 (Z axis) should be flipped
        assert T[2, 2] == -1.0

    def test_col_major_reshape(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
        )
        # Create a known non-symmetric matrix
        R = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        row_major_flat = R.flatten()  # [1,2,3,4,5,6,7,8,9]
        col_major_flat = R.flatten(order="F")  # [1,4,7,2,5,8,3,6,9]

        cand_row = ConventionCandidate(
            name="row", reshape_mode="row_major",
            R_direction="cam2world", y_flip=False, z_flip=False,
        )
        cand_col = ConventionCandidate(
            name="col", reshape_mode="col_major",
            R_direction="cam2world", y_flip=False, z_flip=False,
        )

        T_row = cand_row.build_T(np.zeros(3), row_major_flat)
        T_col = cand_col.build_T(np.zeros(3), row_major_flat)
        # Both should recover the same R since col_major interprets the same flat
        # array differently
        assert np.allclose(T_row[:3, :3], R)
        assert np.allclose(T_col[:3, :3], R.T)

    def test_name_construction(self):
        from vlm_nsm.experiments.search_camera_convention import (
            generate_candidates,
        )
        candidates = generate_candidates()
        # Find the candidate with all flags set
        cand = next(
            c for c in candidates
            if c.reshape_mode == "row_major"
            and c.R_direction == "world2cam"
            and c.y_flip
            and c.z_flip
        )
        assert "row_major" in cand.name
        assert "world2cam" in cand.name
        assert "yflip" in cand.name
        assert "zflip" in cand.name


class TestGenerateCandidates:
    def test_generates_16_candidates(self):
        from vlm_nsm.experiments.search_camera_convention import (
            generate_candidates,
        )
        candidates = generate_candidates()
        assert len(candidates) == 16  # 2*2*2*2

    def test_all_combinations_present(self):
        from vlm_nsm.experiments.search_camera_convention import (
            generate_candidates,
        )
        candidates = generate_candidates()
        names = {c.name for c in candidates}
        # Check key variants exist
        assert any("row_major_cam2world_vanilla" in n for n in names)
        assert any("col_major" in n for n in names)
        assert any("yflip" in n for n in names)
        assert any("zflip" in n for n in names)

    def test_unique_names(self):
        from vlm_nsm.experiments.search_camera_convention import (
            generate_candidates,
        )
        candidates = generate_candidates()
        names = [c.name for c in candidates]
        assert len(names) == len(set(names))


class TestNormColor:
    def test_standard_colors(self):
        from vlm_nsm.experiments.search_camera_convention import _norm_color
        assert _norm_color("red") == "red"
        assert _norm_color("red_cube") == "red"
        assert _norm_color("GREEN_CUBE") == "green"
        assert _norm_color("Blue") == "blue"

    def test_unknown_passthrough(self):
        from vlm_nsm.experiments.search_camera_convention import _norm_color
        assert _norm_color("purple") == "purple"


class TestEvaluateCandidates:
    def test_perfect_candidate_has_zero_error(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
            evaluate_candidates,
        )

        # Create a known calibration
        K = intrinsic_from_fov(45.0, 512, 512)
        cam_pos = np.array([0.5, 0.0, 1.35])
        R_gt = np.array([
            [0.0, -0.70614784, 0.70806442],
            [1.0, 0.0, 0.0],
            [0.0, 0.70806442, 0.70614784],
        ])
        cam_xmat_flat = R_gt.flatten()  # row-major

        # Forward-project some world points to get (world, pixel) pairs
        T_gt = np.eye(4, dtype=np.float64)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = cam_pos

        from vlm_nsm.perception.camera_geometry import project_world_point_to_pixel

        world_points = [
            np.array([0.1, 0.0, 0.82]),
            np.array([-0.05, 0.03, 0.82]),
        ]

        pairs = []
        for i, p_w in enumerate(world_points):
            proj = project_world_point_to_pixel(p_w, K, T_gt)
            assert proj is not None
            pu, pv = proj
            pairs.append({
                "frame_idx": i,
                "object_id": f"cube{i+1}",
                "oracle_color": "red",
                "oracle_position": {"x": float(p_w[0]), "y": float(p_w[1]), "z": float(p_w[2])},
                "detected_center_u": pu,
                "detected_center_v": pv,
                "detected_bottom_u": pu,
                "detected_bottom_v": pv + 20,  # bottom is 20px lower
            })

        # Create candidates with the correct one
        candidates = [
            ConventionCandidate(
                name="correct", reshape_mode="row_major",
                R_direction="cam2world", y_flip=False, z_flip=False,
            ),
            ConventionCandidate(
                name="wrong_transpose", reshape_mode="row_major",
                R_direction="world2cam", y_flip=False, z_flip=False,
            ),
        ]

        results = evaluate_candidates(
            pairs, candidates, cam_pos, cam_xmat_flat,
            fov_y_deg=45.0, width=512, height=512,
        )

        # First result should be the "correct" one with near-zero error
        assert results[0]["candidate"] == "correct"
        assert results[0]["center_pixel_error_median"] < 0.01  # sub-pixel

        # Second result should have much larger error
        assert results[1]["candidate"] == "wrong_transpose"
        assert results[1]["center_pixel_error_median"] > 1.0  # significant error

    def test_empty_pairs_produces_empty_errors(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
            evaluate_candidates,
        )
        cand = ConventionCandidate(
            name="test", reshape_mode="row_major",
            R_direction="cam2world", y_flip=False, z_flip=False,
        )
        cam_pos = np.array([0.5, 0.0, 1.35])
        xmat = np.eye(3).flatten()

        results = evaluate_candidates(
            [], [cand], cam_pos, xmat, 45.0, 512, 512,
        )
        assert len(results) == 1
        assert "center_pixel_error_median" not in results[0]

    def test_results_sorted_by_center_error(self):
        from vlm_nsm.experiments.search_camera_convention import (
            ConventionCandidate,
            evaluate_candidates,
        )

        K = intrinsic_from_fov(45.0, 512, 512)
        cam_pos = np.array([0.5, 0.0, 1.35])
        R_gt = np.array([
            [0.0, -0.70614784, 0.70806442],
            [1.0, 0.0, 0.0],
            [0.0, 0.70806442, 0.70614784],
        ])
        T_gt = np.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3] = cam_pos
        cam_xmat_flat = R_gt.flatten()

        from vlm_nsm.perception.camera_geometry import project_world_point_to_pixel

        p_w = np.array([0.1, 0.0, 0.82])
        proj = project_world_point_to_pixel(p_w, K, T_gt)
        assert proj is not None
        pu, pv = proj

        pairs = [{
            "frame_idx": 0,
            "object_id": "cube1",
            "oracle_color": "red",
            "oracle_position": {"x": float(p_w[0]), "y": float(p_w[1]), "z": float(p_w[2])},
            "detected_center_u": pu,
            "detected_center_v": pv,
            "detected_bottom_u": pu,
            "detected_bottom_v": pv + 20,
        }]

        # Create several candidates with varying errors
        candidates = [
            ConventionCandidate(
                name="c_good", reshape_mode="row_major",
                R_direction="cam2world", y_flip=False, z_flip=False,
            ),
            ConventionCandidate(
                name="a_bad", reshape_mode="col_major",
                R_direction="cam2world", y_flip=True, z_flip=True,
            ),
        ]

        results = evaluate_candidates(
            pairs, candidates, cam_pos, cam_xmat_flat, 45.0, 512, 512,
        )

        # Should be sorted by median center error ascending
        assert results[0]["candidate"] == "c_good"
