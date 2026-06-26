"""Tests for strict vs semantic success metric consistency (robosuite-independent)."""

from vlm_nsm.core.metrics import reality_gap


def test_strict_rg_ge_semantic_rg_all_success():
    """When all episodes succeed, both RGs should be 0%."""
    strict_rg = reality_gap(100, 100)
    semantic_rg = reality_gap(100, 100)
    assert strict_rg == 0.0
    assert semantic_rg == 0.0
    assert strict_rg >= semantic_rg  # 0 >= 0


def test_strict_rg_ge_semantic_rg_semantic_only():
    """Semantic-only success: strict=0, semantic=100 → strict RG > semantic RG."""
    strict_rg = reality_gap(100, 0)
    semantic_rg = reality_gap(100, 100)
    assert strict_rg == 100.0
    assert semantic_rg == 0.0
    assert strict_rg >= semantic_rg


def test_strict_rg_ge_semantic_rg_mixed():
    """Mixed: strict=60, semantic=90 → strict RG=40%, semantic RG=10%."""
    strict_rg = reality_gap(100, 60)
    semantic_rg = reality_gap(100, 90)
    assert strict_rg == 40.0
    assert semantic_rg == 10.0
    assert strict_rg >= semantic_rg


def test_strict_rg_ge_semantic_rg_all_fail():
    """All fail: both RGs at 100%."""
    strict_rg = reality_gap(100, 0)
    semantic_rg = reality_gap(100, 0)
    assert strict_rg == 100.0
    assert semantic_rg == 100.0
    assert strict_rg >= semantic_rg


def test_reality_gap_none_for_zero_symbolic():
    """When symbolic success is 0, reality_gap returns None."""
    assert reality_gap(0, 0) is None
    assert reality_gap(0, 10) is None


def test_reality_gap_types():
    """Reality gap should return float or None."""
    assert isinstance(reality_gap(10, 8), float)
    assert reality_gap(0, 0) is None


def test_reality_gap_invariant_semantic_never_exceeds_strict():
    """Semantic RG should never exceed strict RG."""
    import random
    for _ in range(20):
        n = random.randint(1, 100)
        strict_s = random.randint(0, n)
        semantic_s = random.randint(strict_s, n)  # semantic >= strict
        strict_rg = reality_gap(n, strict_s)
        semantic_rg = reality_gap(n, semantic_s)
        assert strict_rg >= semantic_rg, (
            f"strict RG {strict_rg} < semantic RG {semantic_rg} "
            f"for n={n} strict={strict_s} semantic={semantic_s}"
        )
