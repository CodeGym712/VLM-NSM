"""Test that online_qwen_parser prompt includes ungroundable instructions.

No DashScope/API calls required — only tests prompt construction.
"""

import pytest
from vlm_nsm.language.online_qwen_parser import OnlineQwenGoalParser


class TestUngroundablePrompt:
    def setup_method(self):
        self.parser = OnlineQwenGoalParser(api_key="test-key")

    def test_prompt_without_scene_no_ungroundable(self):
        """Without scene_description, prompt should NOT mention 'ungroundable'."""
        prompt = self.parser._build_prompt(
            instruction="Put the red cube on the green cube.",
        )
        assert "ungroundable" not in prompt, (
            "ungroundable should not appear when no scene is provided"
        )

    def test_prompt_with_scene_has_ungroundable_rules(self):
        """With scene_description, prompt should include ungroundable rules."""
        scene = "  cube1 (block), color: red\n  cube2 (block), color: green"
        prompt = self.parser._build_prompt(
            instruction="Put the red cube on the green cube.",
            scene_description=scene,
        )
        assert "ungroundable" in prompt, (
            "ungroundable should appear when scene is provided"
        )
        assert "status" in prompt
        assert "NOT present in the scene inventory" in prompt

    def test_prompt_with_scene_has_ungroundable_example(self):
        """With scene_description, the yellow-cube example should appear."""
        scene = "  cube1 (block), color: red\n  cube2 (block), color: green"
        prompt = self.parser._build_prompt(
            instruction="Put the red cube on the green cube.",
            scene_description=scene,
        )
        assert "yellow cube" in prompt, (
            "The yellow-cube ungroundable example should be in the prompt"
        )
        assert "ungroundable" in prompt

    def test_prompt_with_scene_includes_reason(self):
        """The ungroundable schema should include a reason field."""
        scene = "  cube1 (block), color: red\n  cube2 (block), color: green"
        prompt = self.parser._build_prompt(
            instruction="Put the red cube on the green cube.",
            scene_description=scene,
        )
        assert '"reason"' in prompt

    def test_prompt_with_scene_rejects_missing_color(self):
        """Rules should reject mapping missing colors to objects."""
        scene = "  cube1 (block), color: red\n  cube2 (block), color: green"
        prompt = self.parser._build_prompt(
            instruction="Put the red cube on the green cube.",
            scene_description=scene,
        )
        assert "Do NOT map a missing color name" in prompt

    def test_prompt_with_scene_includes_inventory(self):
        """Scene inventory should be included in prompt."""
        scene = "  cube1 (block), color: red\n  cube2 (block), color: green"
        prompt = self.parser._build_prompt(
            instruction="Test",
            scene_description=scene,
        )
        assert "Scene inventory:" in prompt
        assert "color: red" in prompt

    def test_prompt_with_scene_allows_spatial_inference(self):
        """With scene_description, spatial attributes should be inferable from poses."""
        scene = "  cube1 (block), color: red at (0.10, 0.00, 0.80)\n  cube2 (block), color: green at (-0.10, 0.00, 0.80)"
        prompt = self.parser._build_prompt(
            instruction="Put the front cube on the back cube.",
            scene_description=scene,
        )
        assert "Spatial attributes" in prompt
        assert "CAN be inferred" in prompt
        assert "object pose coordinates" in prompt

    def test_prompt_multiple_blocks_no_scene_no_ungroundable(self):
        """3-block prompt without scene should also omit ungroundable."""
        self.parser._build_prompt(
            instruction="Stack them up.",
            available_objects=["cube1", "cube2", "cube3"],
        )
        prompt = self.parser._build_prompt(
            instruction="Stack them up.",
            available_objects=["cube1", "cube2", "cube3"],
        )
        assert "ungroundable" not in prompt
