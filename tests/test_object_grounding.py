"""Test object grounding evaluator.

No DashScope, Robosuite, or GPU required.
"""

import pytest
from vlm_nsm.evaluation.object_grounding import evaluate_object_grounding
from vlm_nsm.perception.data_types import DetectedObject, SceneInfo


class TestObjectGrounding:
    def test_expected_bindings_match(self):
        """Should pass when predicted matches expected."""
        goal_json = {
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
        }
        result = evaluate_object_grounding(
            instruction="Put red on green",
            goal_json=goal_json,
            expected_bindings={"top block": "cube1", "bottom block": "cube2"},
        )
        assert result["object_grounding_valid"] is True
        assert result["grounding_errors"] == []

    def test_expected_bindings_mismatch(self):
        """Should report error when predicted differs."""
        goal_json = {
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
        }
        result = evaluate_object_grounding(
            instruction="Put red on green",
            goal_json=goal_json,
            expected_bindings={"top block": "cube2", "bottom block": "cube1"},
        )
        assert result["object_grounding_valid"] is False
        assert len(result["grounding_errors"]) == 2

    def test_missing_binding(self):
        """Should report error when binding is missing."""
        goal_json = {"object_bindings": {"top block": "cube1"}}  # missing bottom
        result = evaluate_object_grounding(
            instruction="Put red on green",
            goal_json=goal_json,
            expected_bindings={"top block": "cube1", "bottom block": "cube2"},
        )
        assert result["object_grounding_valid"] is False
        errors = " ".join(result["grounding_errors"])
        assert "Missing binding" in errors

    def test_no_goal_json(self):
        """Should handle None goal_json gracefully."""
        result = evaluate_object_grounding(
            instruction="test",
            goal_json=None,
        )
        assert result["predicted_bindings"] == {}
        assert result["grounding_method_used"] == "none"

    def test_no_expected_no_scene(self):
        """Should return valid=True with method=none when nothing to check."""
        goal_json = {"object_bindings": {"top block": "cube1"}}
        result = evaluate_object_grounding(
            instruction="test",
            goal_json=goal_json,
        )
        assert result["object_grounding_valid"] is True
        assert result["grounding_method_used"] == "none"

    def test_scene_info_color_matching(self):
        """Should use scene colors when expected_bindings is not provided."""
        goal_json = {
            "object_bindings": {"top block": "cube1", "bottom block": "cube2"},
        }
        scene = SceneInfo(objects=[
            DetectedObject(label="cube1", category="block", color="red"),
            DetectedObject(label="cube2", category="block", color="green"),
        ])
        result = evaluate_object_grounding(
            instruction="Put the red cube on the green cube",
            goal_json=goal_json,
            scene_info=scene,
        )
        # With scene info, method should not be "none"
        assert result["grounding_method_used"] != "none"

    def test_empty_goal_json(self):
        """Empty object_bindings should be handled."""
        result = evaluate_object_grounding(
            instruction="test",
            goal_json={},
            expected_bindings={"top block": "cube1"},
        )
        assert result["object_grounding_valid"] is False
