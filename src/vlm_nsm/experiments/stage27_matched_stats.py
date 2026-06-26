"""Compute matched-pair statistics and bootstrap confidence intervals for Stage 27.

Reads oracle_episodes.jsonl and live_episodes.jsonl from the matched eval
directory, then writes matched_stats.json with bootstrap CIs and matched-pair
outcome counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a .jsonl file as a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute bootstrap confidence intervals for the mean of *values*.

    Parameters
    ----------
    values : np.ndarray
        1-D array of binary (0/1) or continuous values.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level in percent (default 95).
    rng : np.random.Generator | None
        Optional seeded generator for reproducibility.

    Returns
    -------
    dict with keys:
        mean     – point estimate of the mean
        ci_lower – lower bound of the CI
        ci_upper – upper bound of the CI
        ci_label – human-readable label, e.g. "95%"
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "ci_label": f"{ci}%", "n": 0}

    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(np.mean(sample))

    lower_pct = (100.0 - ci) / 2.0
    upper_pct = 100.0 - lower_pct

    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(means, lower_pct)),
        "ci_upper": float(np.percentile(means, upper_pct)),
        "ci_label": f"{ci}%",
        "n": n,
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_matched_stats(
    oracle_path: str | Path,
    live_path: str | Path,
    output_path: str | Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute matched-pair statistics and write them to *output_path*.

    Parameters
    ----------
    oracle_path : str | Path
        Path to ``oracle_episodes.jsonl``.
    live_path : str | Path
        Path to ``live_episodes.jsonl``.
    output_path : str | Path
        Where to write ``matched_stats.json``.
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    seed : int
        Seed for the random generator.

    Returns
    -------
    The statistics dict that was written to disk.
    """
    oracle_eps = _load_jsonl(oracle_path)
    live_eps = _load_jsonl(live_path)

    rng = np.random.default_rng(seed)

    # Index by episode_id for matched pairing
    oracle_by_id: dict[str, dict[str, Any]] = {ep["episode_id"]: ep for ep in oracle_eps}
    live_by_id: dict[str, dict[str, Any]] = {ep["episode_id"]: ep for ep in live_eps}

    # Determine the task-direction key (could be "task_direction" or "goal_id")
    _sample = oracle_eps[0] if oracle_eps else {}
    direction_key = "task_direction" if "task_direction" in _sample else "goal_id"

    # Per-direction keys used in this data
    red_key = "red_on_green"
    green_key = "green_on_red"

    # ---- Assemble vectors ----
    common_ids = sorted(set(oracle_by_id) & set(live_by_id))
    oracle_success_all = np.array(
        [bool(oracle_by_id[eid].get("task_success", False)) for eid in common_ids],
        dtype=np.float64,
    )
    live_success_all = np.array(
        [bool(live_by_id[eid].get("task_success", False)) for eid in common_ids],
        dtype=np.float64,
    )

    # Per-direction masks
    is_red = np.array(
        [oracle_by_id[eid].get(direction_key) == red_key for eid in common_ids],
        dtype=bool,
    )
    is_green = np.array(
        [oracle_by_id[eid].get(direction_key) == green_key for eid in common_ids],
        dtype=bool,
    )

    oracle_red = oracle_success_all[is_red]
    oracle_green = oracle_success_all[is_green]
    live_red = live_success_all[is_red]
    live_green = live_success_all[is_green]

    # Differences (Live minus Oracle)
    diff_all = live_success_all - oracle_success_all
    diff_red = live_red - oracle_red
    diff_green = live_green - oracle_green

    # ---- Bootstrap CIs ----
    stats: dict[str, Any] = {}

    stats["oracle_overall_success"] = _bootstrap_ci(oracle_success_all, n_bootstrap, rng=rng)
    stats["live_overall_success"] = _bootstrap_ci(live_success_all, n_bootstrap, rng=rng)
    stats["oracle_red_success"] = _bootstrap_ci(oracle_red, n_bootstrap, rng=rng)
    stats["oracle_green_success"] = _bootstrap_ci(oracle_green, n_bootstrap, rng=rng)
    stats["live_red_success"] = _bootstrap_ci(live_red, n_bootstrap, rng=rng)
    stats["live_green_success"] = _bootstrap_ci(live_green, n_bootstrap, rng=rng)

    stats["diff_overall"] = _bootstrap_ci(diff_all, n_bootstrap, rng=rng)
    stats["diff_red"] = _bootstrap_ci(diff_red, n_bootstrap, rng=rng)
    stats["diff_green"] = _bootstrap_ci(diff_green, n_bootstrap, rng=rng)

    # ---- Matched-pair outcome counts ----
    both_success = int(np.sum((oracle_success_all == 1.0) & (live_success_all == 1.0)))
    oracle_only = int(np.sum((oracle_success_all == 1.0) & (live_success_all == 0.0)))
    live_only = int(np.sum((oracle_success_all == 0.0) & (live_success_all == 1.0)))
    both_fail = int(np.sum((oracle_success_all == 0.0) & (live_success_all == 0.0)))

    stats["matched_pair_outcomes"] = {
        "both_success": both_success,
        "oracle_only": oracle_only,
        "live_only": live_only,
        "both_fail": both_fail,
        "total_pairs": len(common_ids),
    }

    # Also include per-direction matched-pair breakdowns
    red_mask = is_red
    green_mask = is_green

    for label, mask in [("red_on_green", red_mask), ("green_on_red", green_mask)]:
        o = oracle_success_all[mask]
        l = live_success_all[mask]
        stats[f"matched_pair_outcomes_{label}"] = {
            "both_success": int(np.sum((o == 1.0) & (l == 1.0))),
            "oracle_only": int(np.sum((o == 1.0) & (l == 0.0))),
            "live_only": int(np.sum((o == 0.0) & (l == 1.0))),
            "both_fail": int(np.sum((o == 0.0) & (l == 0.0))),
            "total_pairs": int(mask.sum()),
        }

    # ---- Write output ----
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute Stage 27 matched-pair bootstrap statistics")
    parser.add_argument(
        "--oracle-episodes",
        required=True,
        help="Path to oracle_episodes.jsonl",
    )
    parser.add_argument(
        "--live-episodes",
        required=True,
        help="Path to live_episodes.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for matched_stats.json output",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    compute_matched_stats(
        oracle_path=args.oracle_episodes,
        live_path=args.live_episodes,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
