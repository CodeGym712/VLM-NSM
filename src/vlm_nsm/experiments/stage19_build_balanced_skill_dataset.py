"""Stage 19-D2-4: Build a balanced skill dataset index.

Combines forward demonstrations (cube1->cube2, from existing skill_demos manifest)
with reverse demonstrations (cube2->cube1, from D2-3 collection results) into a
balanced dataset index JSON.

CLI:
  python -m vlm_nsm.experiments.stage19_build_balanced_skill_dataset \
    --forward_manifest generated/datasets/skill_demos_v0_1_smoke_check/manifest.jsonl \
    --reverse_episodes generated/results/stage_19/stage19_d2_reverse_collection_episodes.jsonl \
    --output generated/datasets/stage19_d2_balanced_skill_dataset.json \
    --max_forward 50 \
    --max_reverse 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_reverse_successes(reverse_path: Path) -> list[dict]:
    """Load successful reverse episodes from the D2-3 rollout episodes JSONL."""
    successes: list[dict] = []
    with open(reverse_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if ep.get("task_success") is True:
                successes.append(ep)

    successes.sort(key=lambda e: e.get("seed", e.get("episode_id", "")))
    return successes


def load_forward_entries(manifest_path: Path, max_forward: int) -> list[dict]:
    """Load forward (cube1->cube2) successful entries from existing manifest."""
    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("success") is not True:
                continue
            obj = entry.get("object_id", "")
            tgt = entry.get("target_object_id", "")
            # Only include forward direction: cube1 on cube2
            if obj == "cube1" and tgt == "cube2":
                entries.append(entry)

    entries.sort(key=lambda e: e.get("episode_id", ""))
    if len(entries) > max_forward:
        entries = entries[:max_forward]
    return entries


def build_balanced_index(
    forward_entries: list[dict],
    reverse_episodes: list[dict],
    max_forward: int,
    max_reverse: int,
) -> dict:
    """Build a balanced dataset index document."""
    forward_count = min(len(forward_entries), max_forward)
    reverse_count = min(len(reverse_episodes), max_reverse)

    forward_items: list[dict] = []
    for entry in forward_entries[:max_forward]:
        forward_items.append({
            "source": "skill_demos_v0_1",
            "direction": "forward",
            "object_mapping": {"top": "cube1", "bottom": "cube2"},
            "episode_id": entry.get("episode_id", ""),
            "skill_name": entry.get("skill_name", "pick_place"),
            "object_id": entry.get("object_id", "cube1"),
            "target_object_id": entry.get("target_object_id", "cube2"),
            "trajectory_path": entry.get("trajectory_path", ""),
            "success": True,
            "num_steps": entry.get("num_steps"),
            "seed": entry.get("seed"),
            "teacher_mode": entry.get("teacher_mode"),
        })

    reverse_items: list[dict] = []
    for ep in reverse_episodes[:max_reverse]:
        reverse_items.append({
            "source": "stage19_d2_reverse_collection",
            "direction": "reverse",
            "object_mapping": {"top": "cube2", "bottom": "cube1"},
            "episode_id": ep.get("episode_id", ""),
            "skill_name": "pick_place",
            "object_id": "cube2",
            "target_object_id": "cube1",
            "trajectory_path": "",  # Will be populated after trajectory extraction
            "success": True,
            "num_steps": ep.get("steps"),
            "seed": ep.get("seed"),
            "teacher_mode": "scripted_only",
        })

    index_doc = {
        "description": "Stage 19-D2 Balanced Skill Dataset Index",
        "date": "2026-05-27",
        "total_episodes": forward_count + reverse_count,
        "forward_count": forward_count,
        "reverse_count": reverse_count,
        "is_balanced": abs(forward_count - reverse_count) <= max(forward_count, reverse_count) * 0.2,
        "forward_items": forward_items,
        "reverse_items": reverse_items,
        "notes": [
            f"Forward demos: cube1 -> cube2 (pick_place from skill_demos_v0_1 manifest)",
            f"Reverse demos: cube2 -> cube1 (D2-3 scripted-only collection)",
            "Balanced <= 20% difference between forward and reverse counts.",
            "Reverse trajectory_path fields need to be populated after trajectory extraction from D2 rollouts.",
        ],
    }

    return index_doc


def main():
    parser = argparse.ArgumentParser(description="Build balanced skill dataset index")
    parser.add_argument("--forward_manifest", required=True,
                        help="Path to existing forward demo manifest.jsonl")
    parser.add_argument("--reverse_episodes", required=True,
                        help="Path to D2-3 reverse collection episodes JSONL")
    parser.add_argument("--output", required=True,
                        help="Output path for balanced dataset index JSON")
    parser.add_argument("--max_forward", type=int, default=50,
                        help="Maximum forward episodes to include (default: 50)")
    parser.add_argument("--max_reverse", type=int, default=50,
                        help="Maximum reverse episodes to include (default: 50)")
    args = parser.parse_args()

    forward_manifest = Path(args.forward_manifest)
    reverse_path = Path(args.reverse_episodes)
    output_path = Path(args.output)

    if not forward_manifest.exists():
        print(f"ERROR: Forward manifest not found: {forward_manifest}")
        sys.exit(1)
    if not reverse_path.exists():
        print(f"ERROR: Reverse episodes not found: {reverse_path}")
        print("Run D2-3 first:")
        print(f"  MUJOCO_GL=egl python -m vlm_nsm.experiments.stage19_run_symbolic_rollouts \\")
        print(f"    --rollout_suite generated/eval_inputs/stage_19/stage19_d2_reverse_demo_collection_suite.json \\")
        print(f"    --execution_config generated/configs/stage_7A-12/robosuite_best_execution_config_stage7a12_final.json \\")
        print(f"    --scripted_only --save_diagnostics \\")
        print(f"    --output_episodes generated/results/stage_19/stage19_d2_reverse_collection_episodes.jsonl \\")
        print(f"    --output_summary generated/results/stage_19/stage19_d2_reverse_collection_summary.json")
        sys.exit(1)

    print(f"Loading forward entries from: {forward_manifest}")
    forward_entries = load_forward_entries(forward_manifest, args.max_forward)
    print(f"  Forward successes: {len(forward_entries)} (max {args.max_forward})")

    print(f"Loading reverse episodes from: {reverse_path}")
    reverse_successes = load_reverse_successes(reverse_path)
    print(f"  Reverse successes: {len(reverse_successes)} (max {args.max_reverse})")

    index_doc = build_balanced_index(
        forward_entries, reverse_successes,
        args.max_forward, args.max_reverse,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nBalanced dataset index written to: {output_path}")
    print(f"  Total: {index_doc['total_episodes']} episodes")
    print(f"  Forward: {index_doc['forward_count']}")
    print(f"  Reverse: {index_doc['reverse_count']}")
    print(f"  Balanced: {'YES' if index_doc['is_balanced'] else 'NO'}")

    if not index_doc["is_balanced"]:
        print(f"\n  WARNING: Dataset is not balanced. Forward/reverse differ by >20%.")
        print(f"  Consider collecting more reverse demonstrations or reducing max_forward.")

    return index_doc


if __name__ == "__main__":
    main()
