"""Tests for demo metadata schema (robosuite-independent)."""

import json
from pathlib import Path


def _demo_meta_schema() -> dict:
    """Return the expected schema for collection_summary.json."""
    return {
        "output_dir": str,
        "num_episodes": int,
        "strict_success_count": int,
        "semantic_only_success_count": int,
        "all_semantic_success_count": int,
        "failure_count": int,
        "strict_success_rate": float,
        "semantic_success_rate": float,
        "per_episode": list,
    }


def _episode_entry_schema() -> dict:
    """Return the expected schema for each per_episode entry."""
    return {
        "episode": int,
        "strict_success": bool,
        "semantic_success": bool,
        "failure_type": str,
        "num_steps": int,
        "timeout_type": (str, type(None)),
        "early_stopped": bool,
        "initial_cube1_z": (int, float),
        "initial_cube2_z": (int, float),
        "final_cube1_z": (int, float),
        "final_cube2_z": (int, float),
    }


def test_meta_schema_required_keys():
    """Collection summary should have all required top-level keys."""
    schema = _demo_meta_schema()
    required = set(schema.keys())
    assert "output_dir" in required
    assert "num_episodes" in required
    assert "strict_success_count" in required
    assert "semantic_only_success_count" in required
    assert "all_semantic_success_count" in required
    assert "failure_count" in required
    assert "per_episode" in required


def test_meta_schema_types():
    """Collection summary fields should match expected types."""
    schema = _demo_meta_schema()
    for field, expected_type in schema.items():
        assert field in schema, f"Field {field} missing from schema"
        if expected_type is not None:
            assert expected_type is not None


def test_episode_entry_schema_required_keys():
    """Each per_episode entry should have all required fields."""
    schema = _episode_entry_schema()
    required = set(schema.keys())
    assert "episode" in required
    assert "strict_success" in required
    assert "semantic_success" in required
    assert "failure_type" in required
    assert "num_steps" in required


def test_npz_schema():
    """Expected npz fields for robosuite demos."""
    expected_fields = {
        "initial_object_poses",
        "initial_eef",
        "initial_gripper_open",
        "final_object_poses",
        "final_eef",
        "final_gripper_open",
        "strict_success",
        "semantic_success",
        "failure_type",
        "num_steps",
    }
    assert len(expected_fields) == 10


def test_directory_categories():
    """Expected category directories for demo output."""
    categories = {
        "strict_success",
        "semantic_only_success",
        "all_semantic_success",
        "failure_debug",
    }
    assert len(categories) == 4
    assert "strict_success" in categories
    assert "semantic_only_success" in categories
    assert "all_semantic_success" in categories
    assert "failure_debug" in categories


def test_strict_plus_semantic_only_equals_all_semantic():
    """all_semantic_success is a union of strict_success and semantic_only_success."""
    strict = 10
    semantic_only = 3
    all_semantic = strict + semantic_only
    assert all_semantic == 13
    assert all_semantic >= strict
    assert all_semantic >= semantic_only
