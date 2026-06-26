"""Verify phase ablation results."""

import json
from pathlib import Path


RESULTS_DIR = Path("generated/results/stage_7B-5")


def test_reach_pick_only_exists():
    p = RESULTS_DIR / "phase_reach_pick_only_20ep.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["success_rate"] > 0
    assert data["mean_bc_steps"] < data["mean_scripted_steps"]


def test_reach_place_only_exists():
    p = RESULTS_DIR / "phase_reach_place_only_20ep.json"
    assert p.exists()
    data = json.loads(p.read_text())



def test_reach_and_grasp_catastrophic():
    p = RESULTS_DIR / "phase_reach_and_grasp_20ep.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["success_rate"] == 0.0, "grasp included should be 0%"


def test_hybrid_both_reach_best():
    both = json.loads((RESULTS_DIR / "dagger_reach_seed0_20ep.json").read_text())
    pick_only = json.loads((RESULTS_DIR / "phase_reach_pick_only_20ep.json").read_text())
    place_only = json.loads((RESULTS_DIR / "phase_reach_place_only_20ep.json").read_text())
    # Both reach should be >= each individual phase
    assert both["success_rate"] >= pick_only["success_rate"]
    assert both["success_rate"] >= place_only["success_rate"]
