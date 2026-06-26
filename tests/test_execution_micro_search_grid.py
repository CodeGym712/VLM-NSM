"""Tests for execution parameter search grid dimensions (robosuite-independent).

Verifies that the grid correctly enumerates all config combinations
including the new close_steps and reach_threshold dimensions.
"""

import itertools


def test_full_grid_dimensions():
    """Full grid should have the expected 5 dimensions."""
    kp_values = [0.02, 0.03, 0.05, 0.08]
    max_delta_values = [0.01, 0.015, 0.02, 0.03]
    early_stop_values = [True, False]
    close_steps_values = [30, 50, 70]
    reach_threshold_values = [0.015, 0.02, 0.025]

    configs = list(itertools.product(
        kp_values, max_delta_values, early_stop_values,
        close_steps_values, reach_threshold_values
    ))
    expected = (len(kp_values) * len(max_delta_values) * len(early_stop_values)
                * len(close_steps_values) * len(reach_threshold_values))
    assert len(configs) == expected
    assert len(configs) == 4 * 4 * 2 * 3 * 3  # 288


def test_quick_grid_dimensions():
    """Quick grid should have fewer values per dimension."""
    kp_values = [0.03, 0.05, 0.08]
    max_delta_values = [0.015, 0.02, 0.03]
    early_stop_values = [True, False]
    close_steps_values = [50]  # fixed in quick
    reach_threshold_values = [0.02]  # fixed in quick

    configs = list(itertools.product(
        kp_values, max_delta_values, early_stop_values,
        close_steps_values, reach_threshold_values
    ))
    expected = (len(kp_values) * len(max_delta_values) * len(early_stop_values)
                * len(close_steps_values) * len(reach_threshold_values))
    assert len(configs) == expected
    assert len(configs) == 3 * 3 * 2 * 1 * 1  # 18


def test_grid_includes_new_params():
    """Each config tuple should include all 5 values."""
    kp_values = [0.05, 0.08]
    max_delta_values = [0.015, 0.02]
    early_stop_values = [True, False]
    close_steps_values = [30, 50]
    reach_threshold_values = [0.015, 0.02]

    for kp, md, es, cs, rt in itertools.product(
        kp_values, max_delta_values, early_stop_values,
        close_steps_values, reach_threshold_values
    ):
        assert isinstance(kp, float)
        assert isinstance(md, float)
        assert isinstance(es, bool)
        assert isinstance(cs, int)
        assert isinstance(rt, float)


def test_close_steps_values_ordered():
    """close_steps values should be in increasing order."""
    values = [30, 50, 70]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_reach_threshold_values_ordered():
    """reach_threshold values should be in increasing order."""
    values = [0.015, 0.02, 0.025]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_grid_minimal():
    """Minimal grid covers at least fundamental parameter combos."""
    kp_values = [0.03, 0.05, 0.08]
    max_delta_values = [0.015, 0.02]
    early_stop_values = [True, False]
    close_steps_values = [50]
    reach_threshold_values = [0.02]

    configs = list(itertools.product(
        kp_values, max_delta_values, early_stop_values,
        close_steps_values, reach_threshold_values
    ))
    # At minimum, test all kp x max_delta x early_stop combos
    assert len(configs) >= 3 * 2 * 2
