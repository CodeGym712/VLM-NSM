"""Tests for SymbolConditionedObservationFilter."""

import numpy as np

from vlm_nsm.embodied.symbol_conditioned_observation_filter import (
    SymbolConditionedObservationFilter,
)


def test_filter_returns_correct_shape():
    filt = SymbolConditionedObservationFilter(obs_dim=16)
    obs = {"cube1_pos": [0.1, 0.2, 0.3], "cube2_pos": [0.4, 0.5, 0.6]}
    result = filt.filter("move", ["cube1", "table", "cube2"], obs)
    assert isinstance(result, np.ndarray)
    assert result.shape == (16,)


def test_filter_action_encoding():
    filt = SymbolConditionedObservationFilter(obs_dim=16)
    obs = {"cube1_pos": [0.0, 0.0, 0.0], "cube2_pos": [0.0, 0.0, 0.0]}
    move_result = filt.filter("move", ["cube1", "table", "cube2"], obs)
    grasp_result = filt.filter("grasp", ["cube1"], obs)
    # Different action names should produce different encodings
    assert not np.allclose(move_result, grasp_result)


def test_filter_no_privileged_labels():
    """Verify the filter does not access privileged ground-truth keys."""
    filt = SymbolConditionedObservationFilter(obs_dim=16)
    obs = {"cube1_pos": [0.1, 0.2, 0.3]}
    # Should not fail even with minimal observation
    result = filt.filter("move", ["cube1", "table", "cube2"], obs)
    assert isinstance(result, np.ndarray)
    assert result.shape == (16,)


def test_filter_padding():
    """Very small observations should be padded to obs_dim."""
    filt = SymbolConditionedObservationFilter(obs_dim=32)
    obs = {"cube1_pos": [0.0, 0.0, 0.0]}
    result = filt.filter("pick", ["cube1"], obs)
    assert result.shape == (32,)


def test_filter_truncation():
    """Very large observations should be truncated to obs_dim."""
    filt = SymbolConditionedObservationFilter(obs_dim=8)
    obs = {f"obj{i}_pos": [i * 0.1, 0.0, 0.0] for i in range(20)}
    result = filt.filter("move", ["cube1", "table", "cube2"], obs)
    assert result.shape == (8,)
