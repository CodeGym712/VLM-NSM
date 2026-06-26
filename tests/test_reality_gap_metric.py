"""Tests for Reality Gap metric."""

from vlm_nsm.core.metrics import reality_gap


def test_reality_gap_zero():
    """Both succeed → 0% gap."""
    gap = reality_gap(10, 10)
    assert gap == 0.0


def test_reality_gap_partial():
    """8 symbolic, 6 physical → 25% gap."""
    gap = reality_gap(8, 6)
    assert gap == 25.0


def test_reality_gap_all_physical_fail():
    """10 symbolic, 0 physical → 100% gap."""
    gap = reality_gap(10, 0)
    assert gap == 100.0


def test_reality_gap_na():
    """0 symbolic success → N/A."""
    gap = reality_gap(0, 0)
    assert gap is None


def test_reality_gap_negative():
    """Negative symbolic should also return None."""
    gap = reality_gap(-1, 5)
    assert gap is None
