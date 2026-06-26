"""Generate Stage 20 teacher demo collection suite JSON.

This is a scaffold only. It prepares rollout-suite entries compatible with
Stage 19 symbolic rollout runner but does not run Robosuite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GOAL_SPECS = {
    "red_on_green": {
        "direction": "cube1_on_cube2",
        "top_object": "cube1",
        "bottom_object": "cube2",
        "top_color": "red",
        "bottom_color": "green",
        "target_stack": ["cube2", "cube1"],
    },
    "green_on_red": {
        "direction": "cube2_on_cube1",
        "top_object": "cube2",
        "bottom_object": "cube1",
        "top_color": "green",
        "bottom_color": "red",
        "target_stack": ["cube1", "cube2"],
    },
}


def parser_output_for(top_object: str, bottom_object: str) -> dict:
    return {
        "status": "ok",
        "task_type": "stacking",
        "object_bindings": {
            "top block": top_object,
            "bottom block": bottom_object,
        },
        "goal_predicates": [
            ["on", top_object, bottom_object],
            ["on", bottom_object, "table"],
        ],
    }


def generate_teacher_demo_suite(
    *,
    seeds_per_bin: int = 20,
    placement_bins: list[str] | None = None,
    goals: list[str] | None = None,
    scripted_profile: str = "cube2_robust",
) -> dict:
    if placement_bins is None:
        placement_bins = ["center", "front", "back", "left", "right"]
    if goals is None:
        goals = ["red_on_green", "green_on_red"]
    if seeds_per_bin <= 0:
        raise ValueError("seeds_per_bin must be positive")

    episodes = []
    for goal_id in goals:
        if goal_id not in GOAL_SPECS:
            raise ValueError(f"Unsupported goal: {goal_id}")
        spec = GOAL_SPECS[goal_id]
        for placement_bin in placement_bins:
            for seed in range(seeds_per_bin):
                episode_id = f"stage20_{goal_id}_{placement_bin}_s{seed}"
                episodes.append({
                    "episode_id": episode_id,
                    "goal_id": goal_id,
                    "description": f"Stage 20 teacher collection: {goal_id}, {placement_bin}, seed={seed}",
                    "direction": spec["direction"],
                    "seed": seed,
                    "placement_bin": placement_bin,
                    "expected_execution": "execute",
                    "scripted_only": True,
                    "scripted_profile": scripted_profile,
                    "teacher": "scripted_only",
                    "teacher_profile": scripted_profile,
                    "top_object": spec["top_object"],
                    "bottom_object": spec["bottom_object"],
                    "top_role": "top",
                    "bottom_role": "bottom",
                    "top_color": spec["top_color"],
                    "bottom_color": spec["bottom_color"],
                    "expected_top": spec["top_object"],
                    "expected_bottom": spec["bottom_object"],
                    "expected_target_stack": list(spec["target_stack"]),
                    "parser_output": parser_output_for(spec["top_object"], spec["bottom_object"]),
                })

    return {
        "description": "Stage 20 teacher demo collection scaffold — role-conditioned bidirectional stacking",
        "date": "2026-05-28",
        "stage": "stage_20",
        "execution_mode": "scripted_only",
        "scripted_profile": scripted_profile,
        "teacher": "scripted_only",
        "teacher_profile": scripted_profile,
        "available_objects": ["cube1", "cube2"],
        "color_map": {"cube1": "red", "cube2": "green"},
        "placement_bins": placement_bins,
        "goals": goals,
        "seeds_per_bin": seeds_per_bin,
        "notes": [
            "Scaffold only: placement_bin is nominal until reset constraints support placement control.",
            "Use with --scripted_only and --scripted_profile cube2_robust.",
            "Do not treat scripted profile as final controller; it is teacher/baseline/fallback.",
        ],
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage 20 teacher demo suite scaffold")
    parser.add_argument("--output", default="generated/eval_inputs/stage_20/stage20_teacher_demo_collection_200.json")
    parser.add_argument("--seeds_per_bin", type=int, default=20)
    parser.add_argument("--placement_bins", default="center,front,back,left,right")
    parser.add_argument("--goals", default="red_on_green,green_on_red")
    parser.add_argument("--scripted_profile", default="cube2_robust")
    args = parser.parse_args()

    suite = generate_teacher_demo_suite(
        seeds_per_bin=args.seeds_per_bin,
        placement_bins=[x.strip() for x in args.placement_bins.split(",") if x.strip()],
        goals=[x.strip() for x in args.goals.split(",") if x.strip()],
        scripted_profile=args.scripted_profile,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out} ({len(suite['episodes'])} episodes)")


if __name__ == "__main__":
    main()
