"""Tests for execution param search import and config (no robosuite dependency)."""
from __future__ import annotations

from vlm_nsm.experiments.search_robosuite_execution_params import (
    _try_import_robosuite,
)


def test_import_guard():
    """Import guard works without robosuite."""
    result = _try_import_robosuite()
    assert isinstance(result, bool)
