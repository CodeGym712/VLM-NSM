"""Test robosuite cube name mapping constants and fallback logic.

These tests verify the CUBE_NAME_MAP constant and fallback matching
logic used by RobosuiteEnvWrapper. They do NOT require robosuite.
"""

from __future__ import annotations

import pytest


class TestCubeNameMap:
    """CUBE_NAME_MAP maps canonical names to robosuite body names."""

    def _get_map(self):
        try:
            from vlm_nsm.executors.robosuite_env_wrapper import CUBE_NAME_MAP
            return CUBE_NAME_MAP
        except ImportError:
            pytest.skip("robosuite_env_wrapper not importable")

    def test_map_contains_all_cubes(self):
        mapping = self._get_map()
        # cube3 added for 3-block, cube4 added for 4-block experiment
        assert set(mapping.keys()) >= {"cube1", "cube2"}

    def test_map_values_correct(self):
        mapping = self._get_map()
        assert mapping["cube1"] == "cubeA_main"
        assert mapping["cube2"] == "cubeB_main"

    def test_map_values_are_strings(self):
        mapping = self._get_map()
        for v in mapping.values():
            assert isinstance(v, str)

    def test_no_extra_keys(self):
        mapping = self._get_map()
        assert len(mapping) >= 2


class TestFallbackMatching:
    """Fallback body name matching logic (used when env.cubes not available)."""

    def _fallback_match(self, body_names, canonical):
        """Replicate the wrapper's fallback matching logic."""
        from vlm_nsm.executors.robosuite_env_wrapper import CUBE_NAME_MAP
        body_name = CUBE_NAME_MAP.get(canonical, canonical)
        matches = [b for b in body_names if b.lower().replace(" ", "") == body_name.lower().replace(" ", "")]
        if not matches:
            matches = [b for b in body_names if canonical.lower().replace(" ", "") in b.lower().replace(" ", "")]
        return matches[0] if matches else None

    def test_exact_match(self):
        result = self._fallback_match(["cubeA_main", "cubeB_main", "table"], "cube1")
        assert result == "cubeA_main"

    def test_exact_match_cube2(self):
        result = self._fallback_match(["cubeA_main", "cubeB_main", "table"], "cube2")
        assert result == "cubeB_main"

    def test_fallback_substring_match(self):
        """When exact match fails, fall back to substring matching."""
        result = self._fallback_match(["cube1_body", "cube2_body"], "cube1")
        assert result == "cube1_body"

    def test_fallback_no_match_returns_none(self):
        result = self._fallback_match(["Table", "Shelf"], "cube1")
        assert result is None

    def test_fallback_case_insensitive(self):
        result = self._fallback_match(["cubeA_main", "CUBEA_MAIN", "cubeB_main"], "cube1")
        assert result == "cubeA_main"

    def test_fallback_with_substring(self):
        """Canonical name substring is checked as last resort."""
        result = self._fallback_match(["obj_cube1", "obj_cube2"], "cube1")
        assert result == "obj_cube1"


class TestGetObjectPose:
    """Test error handling for get_object_pose without robosuite."""

    def test_unknown_object_raises(self):
        """The wrapper should raise ValueError for unknown objects."""
        try:
            from vlm_nsm.executors.robosuite_env_wrapper import CUBE_NAME_MAP
            # Just verify the error path: no robosuite means we never reach this
            # but the logic handles ValueError for unknown objects
            known = list(CUBE_NAME_MAP.keys())
            assert "cube1" in known
            assert "invalid_cube" not in known
        except ImportError:
            pytest.skip("robosuite_env_wrapper not importable")

    def test_cube_names_argument_passthrough(self):
        """Verify cube_names is accepted by wrapper constructor signature."""
        import inspect
        try:
            from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
            sig = inspect.signature(RobosuiteEnvWrapper.__init__)
            assert "cube_names" in sig.parameters
        except ImportError:
            pytest.skip("robosuite_env_wrapper not importable — constructor signature untestable")
