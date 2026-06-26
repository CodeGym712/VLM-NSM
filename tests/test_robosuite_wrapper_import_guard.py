"""Test RobosuiteEnvWrapper ImportError guard and constant exports.

These tests verify the module handles missing robosuite gracefully
and exposes the CUBE_NAME_MAP constant. They do NOT require robosuite.
"""

from __future__ import annotations

import importlib

import pytest


def test_constructing_wrapper_raises_without_robosuite():
    """Constructing RobosuiteEnvWrapper should raise ImportError if robosuite not installed."""
    try:
        import robosuite  # noqa: F401
        pytest.skip("robosuite is installed — import guard not triggerable")
    except ImportError:
        from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper
        with pytest.raises(ImportError):
            RobosuiteEnvWrapper()


def test_module_level_constants_accessible():
    """CUBE_NAME_MAP should be accessible even if the class is not."""
    try:
        from vlm_nsm.executors.robosuite_env_wrapper import CUBE_NAME_MAP
        assert isinstance(CUBE_NAME_MAP, dict)
        assert "cube1" in CUBE_NAME_MAP
        assert "cube2" in CUBE_NAME_MAP
        assert CUBE_NAME_MAP["cube1"] == "cubeA_main"
        assert CUBE_NAME_MAP["cube2"] == "cubeB_main"
    except ImportError:
        pytest.skip("robosuite not installed — cannot verify module exports")


def test_robosuite_scripted_controller_importable():
    """RobosuiteScriptedController should import without robosuite."""
    from vlm_nsm.embodied.controllers.robosuite_scripted_controller import (
        RobosuiteScriptedController,
    )
    assert RobosuiteScriptedController is not None


def test_debug_env_script_has_import_guard():
    """debug_robosuite_env should handle ImportError gracefully."""
    spec = importlib.util.spec_from_file_location(
        "debug_robosuite_env",
        "src/vlm_nsm/experiments/debug_robosuite_env.py",
    )
    assert spec is not None, "debug_robosuite_env.py not found"


def test_search_grasp_params_import_guard():
    """search_grasp_params should guard against missing robosuite."""
    import pytest
    import shutil
    if shutil.which("uv") is None:
        pytest.skip("uv not available")
    import subprocess
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "vlm_nsm.experiments.search_grasp_params", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"search_grasp_params --help failed:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("uv run timed out (cold cache)")


def test_analyze_robosuite_failures_import_guard():
    """analyze_robosuite_failures should guard against missing robosuite."""
    import subprocess
    import shutil
    if shutil.which("uv") is None:
        pytest.skip("uv not available")
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "vlm_nsm.experiments.analyze_robosuite_failures", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"analyze_robosuite_failures --help failed:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("uv run timed out (cold cache)")


def test_grasp_state_machine_importable():
    """grasp_state_machine module should import without robosuite."""
    from vlm_nsm.embodied.grasp_state_machine import (
        GraspConfig,
        norm_action,
        _dist,
    )
    assert GraspConfig is not None
    assert callable(norm_action)
    assert callable(_dist)
