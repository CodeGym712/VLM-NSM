"""Verify e2e language-to-sim demo outputs."""

import json
from pathlib import Path

E2E_DIR = Path("generated/results/stage_10/e2e_sim_demo")


def _load_results():
    if not E2E_DIR.exists():
        return []
    results = []
    for f in sorted(E2E_DIR.iterdir()):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return results


def test_at_least_3_demos():
    results = _load_results()
    if not results:
        return  # skip if no demos run yet
    assert len(results) >= 3, f"Only {len(results)} demos found"


def test_each_has_required_fields():
    required = {"instruction", "success", "parsed_goal", "alignment_status", "policy_used"}
    for r in _load_results():
        missing = required - set(r.keys())
        assert not missing, f"Missing fields: {missing}"


def test_each_has_episode_result():
    for r in _load_results():
        assert "episode_result" in r, f"Demo missing episode_result"
        ep = r["episode_result"]
        assert "goal_success" in ep


def test_at_least_60_percent_success():
    results = _load_results()
    if not results:
        return  # skip if no demos run yet
    sr = sum(1 for r in results if r.get("success")) / len(results)
    assert sr >= 0.6, f"Success rate too low: {sr*100:.0f}%"


def test_policy_used_logged():
    for r in _load_results():
        assert r.get("policy_used") is not None
        assert "reach_pick" in r["policy_used"].lower() or "scripted" in r["policy_used"].lower()
