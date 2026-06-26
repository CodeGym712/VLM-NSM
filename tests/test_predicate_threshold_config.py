"""Test predicate threshold configuration for robosuite environment.

Verifies that the PhysicalPredicateExtractor works with the thresholds
used by the robosuite evaluation pipeline.
Does NOT require robosuite.
"""

from __future__ import annotations

import math

import pytest


# Threshold constants matching the robosuite eval script
_PREDICATE_THRESHOLDS = {"xy_threshold": 0.05, "z_min": -0.005, "z_max": 0.05}


def is_on(obj_pose, target_pose, xy_threshold, z_min, z_max) -> bool:
    """Replica of PhysicalPredicateExtractor.is_on."""
    dx = obj_pose["x"] - target_pose["x"]
    dy = obj_pose["y"] - target_pose["y"]
    xy_distance = math.hypot(dx, dy)
    dz = obj_pose["z"] - target_pose["z"]
    return xy_distance <= xy_threshold and z_min <= dz <= z_max


class TestThresholdStructure:
    """Threshold dict should have required keys and valid ranges."""

    def test_has_required_keys(self):
        assert "xy_threshold" in _PREDICATE_THRESHOLDS
        assert "z_min" in _PREDICATE_THRESHOLDS
        assert "z_max" in _PREDICATE_THRESHOLDS

    def test_z_min_less_than_z_max(self):
        assert _PREDICATE_THRESHOLDS["z_min"] < _PREDICATE_THRESHOLDS["z_max"]

    def test_xy_threshold_positive(self):
        assert _PREDICATE_THRESHOLDS["xy_threshold"] > 0

    def test_z_min_allows_slight_negative(self):
        """z_min < 0 allows for object slightly below target surface."""
        assert _PREDICATE_THRESHOLDS["z_min"] <= 0

    def test_z_max_accommodates_block_height(self):
        """z_max should be >= typical block height (0.04) to detect stacking."""
        assert _PREDICATE_THRESHOLDS["z_max"] >= 0.04


class TestIsOn:
    """Test the is_on predicate extraction helper."""

    def test_directly_on(self):
        obj = {"x": 0.0, "y": 0.0, "z": 0.02}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_slightly_off_xy_still_on(self):
        obj = {"x": 0.03, "y": 0.03, "z": 0.02}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        # xy distance = sqrt(0.03^2 + 0.03^2) ≈ 0.042 < 0.05 → still on
        assert is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_too_far_xy_not_on(self):
        obj = {"x": 0.1, "y": 0.0, "z": 0.02}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert not is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_too_high_not_on(self):
        """Object lifted high above target should not be 'on'."""
        obj = {"x": 0.0, "y": 0.0, "z": 0.2}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert not is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_slightly_below_still_on(self):
        """z_min negative allows object to be slightly sunken into surface."""
        obj = {"x": 0.0, "y": 0.0, "z": -0.003}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_too_low_not_on(self):
        obj = {"x": 0.0, "y": 0.0, "z": -0.02}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert not is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_edge_case_at_threshold(self):
        """Exactly at xy_threshold = 0.05 should be on (<=)."""
        obj = {"x": 0.05, "y": 0.0, "z": 0.02}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert is_on(obj, target, **_PREDICATE_THRESHOLDS)

    def test_edge_case_at_z_max(self):
        obj = {"x": 0.0, "y": 0.0, "z": 0.05}
        target = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert is_on(obj, target, **_PREDICATE_THRESHOLDS)


class TestPhysicalStateToPredicates:
    """Integration with physical_state_to_predicates using robosuite thresholds."""

    def test_extract_on_predicates(self):
        from vlm_nsm.core.physical_predicate_extractor import extract_on_predicates

        # cube1 on table, cube2 stacked on cube1 (within xy threshold, above in z)
        object_poses = {
            "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
            "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        }
        targets = {
            "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
            "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
            "table": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        on_preds = extract_on_predicates(object_poses, targets, _PREDICATE_THRESHOLDS)
        assert ("on", "cube1", "table") in on_preds
        assert ("on", "cube2", "cube1") in on_preds

    def test_extract_clear_predicates(self):
        from vlm_nsm.core.physical_predicate_extractor import (
            extract_clear_predicates,
            extract_on_predicates,
        )

        object_poses = {
            "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
            "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        }
        targets = {
            "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
            "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
            "table": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        on_preds = extract_on_predicates(object_poses, targets, _PREDICATE_THRESHOLDS)
        clear_preds = extract_clear_predicates(object_poses.keys(), on_preds)

        assert ("clear", "cube2") in clear_preds  # top of stack
        assert ("clear", "cube1") not in clear_preds  # occupied by cube2

    def test_physical_state_to_predicates_full(self):
        from vlm_nsm.core.physical_predicate_extractor import physical_state_to_predicates

        object_poses = {
            "cube1": {"x": 0.0, "y": 0.0, "z": 0.02},
            "cube2": {"x": 0.0, "y": 0.0, "z": 0.06},
        }
        targets = dict(object_poses)
        targets["table"] = {"x": 0.0, "y": 0.0, "z": 0.0}

        predicates = physical_state_to_predicates(object_poses, targets, _PREDICATE_THRESHOLDS)
        assert ("on", "cube1", "table") in predicates
        assert ("on", "cube2", "cube1") in predicates
        assert ("clear", "cube2") in predicates
