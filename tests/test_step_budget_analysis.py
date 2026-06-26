"""Tests for step budget analysis import and config (no robosuite dependency)."""
from __future__ import annotations

from vlm_nsm.experiments.analyze_robosuite_step_budget import (
    _try_import_robosuite,
)


def test_import_guard():
    """Import guard works without robosuite."""
    # This may return False if robosuite not installed — that's OK
    result = _try_import_robosuite()
    # We just verify it returns a bool without crashing
    assert isinstance(result, bool)
