"""Verify multi-seed policy comparison data."""

import json
from pathlib import Path


RESULTS_DIR = Path("generated/results/stage_7B-5")


def _load(seed, prefix):
    p = RESULTS_DIR / f"{prefix}_seed{seed}_20ep.json"
    return json.loads(p.read_text())


def test_all_policies_have_three_seeds():
    for prefix in ["scripted", "bc_main", "dagger_reach"]:
        rates = []
        for seed in range(3):
            data = _load(seed, prefix)
            rates.append(data["success_rate"])
        # Sanity: success_rate is between 0 and 1
        assert all(0.0 <= r <= 1.0 for r in rates), f"{prefix}: rates out of range {rates}"


def test_dagger_outperforms_bc_main_on_avg():
    dag_rates = [_load(s, "dagger_reach")["success_rate"] for s in range(3)]
    bc_rates = [_load(s, "bc_main")["success_rate"] for s in range(3)]
    assert sum(dag_rates) >= sum(bc_rates), "DAgger avg should >= bc_main avg"


def test_dagger_meets_min_threshold():
    rates = [_load(s, "dagger_reach")["success_rate"] for s in range(3)]
    assert min(rates) >= 0.8, f"DAgger min success {min(rates)} < 0.8"


def test_50ep_meets_threshold():
    p = RESULTS_DIR / "dagger_reach_50ep.json"
    data = json.loads(p.read_text())
    assert data["success_rate"] >= 0.88, f"50ep success {data['success_rate']} < 0.88"


def test_mean_steps_below_1100():
    """p95 steps should be within reasonable horizon bound."""
    for prefix in ["scripted", "bc_main", "dagger_reach"]:
        for seed in range(3):
            data = _load(seed, prefix)
            assert data["p95_steps"] <= 1000, f"{prefix} seed={seed} p95 {data['p95_steps']} > 1000"


def test_episode_count_consistency():
    for prefix in ["scripted", "bc_main", "dagger_reach"]:
        for seed in range(3):
            data = _load(seed, prefix)
            assert len(data["episode_results"]) == data["num_episodes"]
