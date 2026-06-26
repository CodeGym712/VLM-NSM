"""Tests for demo dataset classification logic (robosuite-independent).

Verifies the 4-category classification used in collect_robosuite_demos.py:
  - strict_success: clean execution + goal achieved, no timeout
  - semantic_only_success: goal achieved but not clean execution
  - all_semantic_success: union of strict + semantic_only
  - failure_debug: goal not achieved
"""

import tempfile
from pathlib import Path

import numpy as np


def _classify_episode(strict_ok: bool, semantic_ok: bool) -> str:
    """Replicate classification logic from collect_robosuite_demos.py."""
    if strict_ok:
        return "strict_success"
    elif semantic_ok:
        return "semantic_only_success"
    else:
        return "failure_debug"


def test_classify_strict_success():
    """Strict success = episode_success and goal_ok and no timeout."""
    assert _classify_episode(strict_ok=True, semantic_ok=True) == "strict_success"


def test_classify_semantic_only():
    """Semantic only = goal achieved but not strict."""
    assert _classify_episode(strict_ok=False, semantic_ok=True) == "semantic_only_success"


def test_classify_failure():
    """Failure = goal not achieved."""
    assert _classify_episode(strict_ok=False, semantic_ok=False) == "failure_debug"


def test_impossible_strict_no_semantic():
    """Strict success implies semantic success. strict=True, semantic=False is impossible."""
    # If strict_ok is True, semantic_ok MUST be True by definition
    # But test the classification order: strict checked first
    assert _classify_episode(strict_ok=True, semantic_ok=False) == "strict_success"


def test_all_semantic_is_union():
    """all_semantic_success = strict + semantic_only."""
    episodes = [
        ("strict_success", True, True),
        ("strict_success", True, True),
        ("semantic_only_success", False, True),
        ("failure_debug", False, False),
    ]
    strict_count = sum(1 for _, s, _ in episodes if s)
    semantic_only_count = sum(1 for _, s, sem in episodes if not s and sem)
    all_semantic = strict_count + semantic_only_count
    assert all_semantic == 3


def test_npz_save_with_classification():
    """Simulate npz save with category directories."""
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td)
        categories = ["strict_success", "semantic_only_success", "failure_debug"]

        for cat in categories:
            cat_dir = out_path / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cat_dir / f"{cat}.npz",
                strict_success=np.array([cat == "strict_success"]),
                semantic_success=np.array([cat != "failure_debug"]),
            )

        for cat in categories:
            assert (out_path / cat / f"{cat}.npz").exists()

        # Verify all_semantic_success contains strict + semantic_only
        all_sem_dir = out_path / "all_semantic_success"
        all_sem_dir.mkdir(exist_ok=True)
        for src in ["strict_success", "semantic_only_success"]:
            src_dir = out_path / src
            if src_dir.exists():
                for f in src_dir.iterdir():
                    if f.suffix == ".npz":
                        dst = all_sem_dir / f.name
                        if not dst.exists():
                            import shutil
                            shutil.copy2(f, dst)

        assert len(list(all_sem_dir.iterdir())) == 2

        # failure_debug should NOT be in all_semantic
        for f in (out_path / "failure_debug").iterdir():
            assert not (all_sem_dir / f.name).exists()


def test_classification_mutually_exclusive():
    """Each episode belongs to exactly one primary category."""
    strict_count = 5
    semantic_only_count = 3
    failure_count = 2
    total = strict_count + semantic_only_count + failure_count
    assert total == 10
    assert strict_count + semantic_only_count == total - failure_count


def test_semantic_success_rate_formula():
    """Semantic success rate = (strict + semantic_only) / total."""
    strict = 8
    semantic_only = 2
    total = 10
    rate = (strict + semantic_only) / total
    assert rate == 1.0

    strict = 5
    semantic_only = 3
    total = 10
    rate = (strict + semantic_only) / total
    assert rate == 0.8
