"""Tests for stage27_matched_stats.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vlm_nsm.experiments.stage27_matched_stats import (
    _bootstrap_ci,
    _load_jsonl,
    compute_matched_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_episodes_jsonl(path: Path, episodes: list[dict]) -> Path:
    """Write a list of episode dicts to a .jsonl file."""
    with open(path, "w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    return path


def _make_fake_episodes(
    n_red: int = 10,
    n_green: int = 10,
    oracle_success_rate: float = 0.9,
    live_success_rate: float = 0.7,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Create paired oracle/live episode lists with controlled success rates."""
    rng = np.random.default_rng(seed)
    oracle_eps: list[dict] = []
    live_eps: list[dict] = []

    for i in range(n_red):
        eid = f"s27_red_on_green_s{i:02d}"
        o_success = bool(rng.random() < oracle_success_rate)
        l_success = bool(rng.random() < live_success_rate)
        oracle_eps.append({
            "episode_id": eid,
            "task_direction": "red_on_green",
            "task_success": o_success,
        })
        live_eps.append({
            "episode_id": eid,
            "task_direction": "red_on_green",
            "task_success": l_success,
        })

    for i in range(n_green):
        eid = f"s27_green_on_red_s{i:02d}"
        o_success = bool(rng.random() < oracle_success_rate)
        l_success = bool(rng.random() < live_success_rate)
        oracle_eps.append({
            "episode_id": eid,
            "task_direction": "green_on_red",
            "task_success": o_success,
        })
        live_eps.append({
            "episode_id": eid,
            "task_direction": "green_on_red",
            "task_success": l_success,
        })

    return oracle_eps, live_eps


# ---------------------------------------------------------------------------
# Test 1: bootstrap CI runs without error on fake data
# ---------------------------------------------------------------------------

def test_bootstrap_ci_runs_on_fake_data() -> None:
    """Bootstrap CI computation should run without error and return valid fields."""
    rng = np.random.default_rng(42)
    values = (rng.random(30) < 0.7).astype(np.float64)

    result = _bootstrap_ci(values, n_bootstrap=500, ci=95.0, rng=rng)

    assert "mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "ci_label" in result
    assert "n" in result
    assert result["ci_label"] == "95.0%"
    assert result["n"] == 30
    assert 0.0 <= result["mean"] <= 1.0
    assert 0.0 <= result["ci_lower"] <= 1.0
    assert 0.0 <= result["ci_upper"] <= 1.0
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    # Test edge case: empty input
    empty_result = _bootstrap_ci(np.array([], dtype=np.float64), n_bootstrap=100, rng=rng)
    assert empty_result["mean"] is None
    assert empty_result["ci_lower"] is None
    assert empty_result["ci_upper"] is None
    assert empty_result["n"] == 0


# ---------------------------------------------------------------------------
# Test 2: matched-pair outcome counting is correct
# ---------------------------------------------------------------------------

def test_matched_pair_outcome_counting() -> None:
    """Matched-pair outcome counts must equal expected values with known inputs."""
    # Create controlled data where we know the exact outcomes:
    #   ep_0: both success
    #   ep_1: oracle only
    #   ep_2: live only
    #   ep_3: both fail
    #   ep_4: both success (red)
    #   ep_5: oracle only (green)
    oracle_eps = [
        {"episode_id": "ep_0", "task_direction": "red_on_green", "task_success": True},
        {"episode_id": "ep_1", "task_direction": "red_on_green", "task_success": True},
        {"episode_id": "ep_2", "task_direction": "red_on_green", "task_success": False},
        {"episode_id": "ep_3", "task_direction": "green_on_red", "task_success": False},
        {"episode_id": "ep_4", "task_direction": "green_on_red", "task_success": True},
        {"episode_id": "ep_5", "task_direction": "green_on_red", "task_success": True},
    ]
    live_eps = [
        {"episode_id": "ep_0", "task_direction": "red_on_green", "task_success": True},
        {"episode_id": "ep_1", "task_direction": "red_on_green", "task_success": False},
        {"episode_id": "ep_2", "task_direction": "red_on_green", "task_success": True},
        {"episode_id": "ep_3", "task_direction": "green_on_red", "task_success": False},
        {"episode_id": "ep_4", "task_direction": "green_on_red", "task_success": True},
        {"episode_id": "ep_5", "task_direction": "green_on_red", "task_success": False},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        oracle_path = _write_episodes_jsonl(base / "oracle_episodes.jsonl", oracle_eps)
        live_path = _write_episodes_jsonl(base / "live_episodes.jsonl", live_eps)
        output_path = base / "matched_stats.json"

        stats = compute_matched_stats(
            oracle_path=oracle_path,
            live_path=live_path,
            output_path=output_path,
            n_bootstrap=200,
            seed=42,
        )

        outcomes = stats["matched_pair_outcomes"]
        # Overall: both_success=2 (ep_0, ep_4), oracle_only=2 (ep_1, ep_5),
        #          live_only=1 (ep_2), both_fail=1 (ep_3)
        assert outcomes["both_success"] == 2, f"Expected 2 both_success, got {outcomes['both_success']}"
        assert outcomes["oracle_only"] == 2, f"Expected 2 oracle_only, got {outcomes['oracle_only']}"
        assert outcomes["live_only"] == 1, f"Expected 1 live_only, got {outcomes['live_only']}"
        assert outcomes["both_fail"] == 1, f"Expected 1 both_fail, got {outcomes['both_fail']}"
        assert outcomes["total_pairs"] == 6

        # Per-direction checks
        red_outcomes = stats["matched_pair_outcomes_red_on_green"]
        # red: ep_0 both_success, ep_1 oracle_only, ep_2 live_only
        assert red_outcomes["both_success"] == 1
        assert red_outcomes["oracle_only"] == 1
        assert red_outcomes["live_only"] == 1
        assert red_outcomes["both_fail"] == 0
        assert red_outcomes["total_pairs"] == 3

        green_outcomes = stats["matched_pair_outcomes_green_on_red"]
        # green: ep_3 both_fail, ep_4 both_success, ep_5 oracle_only
        assert green_outcomes["both_success"] == 1
        assert green_outcomes["oracle_only"] == 1
        assert green_outcomes["live_only"] == 0
        assert green_outcomes["both_fail"] == 1
        assert green_outcomes["total_pairs"] == 3

        # Verify the output file was written and is valid JSON
        assert output_path.exists()
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert written["matched_pair_outcomes"] == outcomes
