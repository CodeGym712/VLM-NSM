"""Extract symbolic traces from demonstration data and run GraphLearner.

Reads manifest.jsonl from collected demonstrations, extracts
(state_before, action_label, state_after) triples based on the
operator_check field, and feeds them to GraphLearner.

Usage:
    nsm/bin/python -m vlm_nsm.experiments.run_symbolic_learning \
        --dataset_dir generated/datasets/vla_demos/d1_small_200ep \
        --output_prefix generated/results/symbolic/stacking
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from vlm_nsm.symbolic.graph_learner import GraphLearner


def extract_state_from_effects(effects: list[dict]) -> frozenset:
    """Convert operator_check add_effects to a predicate set."""
    items = []
    for eff in effects:
        name = eff.get("predicate", "")
        # Parse "on(cube1,cube2)" → ("on","cube1","cube2")
        pred_name = name.split("(")[0] if "(" in name else name
        args_str = name.split("(")[1].rstrip(")") if "(" in name else ""
        args = tuple(args_str.split(",")) if args_str else ()
        if eff.get("satisfied", False):
            items.append((pred_name,) + args)
    return frozenset(items)


def extract_triples_from_manifest(
    manifest_path: str | Path,
) -> tuple[list, dict]:
    """Extract (state, action, next_state) triples from manifest.

    Uses the operator_check field which records per-operator effects.
    Since each episode has exactly one skill (pick_place), we use:
      - state_before: empty (assumed starting from initial state)
      - action: pick_place
      - state_after: operator_check.add_effects that are satisfied

    For multi-step tasks, we would chain episodes together.
    """
    triples = []
    stats = defaultdict(int)

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            ep = json.loads(line.strip())
            oc = ep.get("operator_check", {})
            if not oc:
                continue

            # Build state_after from satisfied effects
            effects = oc.get("add_effects", [])
            state_after = extract_state_from_effects(effects)

            # state_before: assume empty (initial state for single-step task)
            # For 2-block stacking, initial state is: ontable(cube1), ontable(cube2),
            # clear(cube1), clear(cube2), handempty
            state_before = frozenset({
                ("ontable", "cube1"), ("ontable", "cube2"),
                ("clear", "cube1"), ("clear", "cube2"),
                ("handempty",),
            })

            action_label = "pick_place"
            task_key = f"{ep.get('pick_object','?')}_to_{ep.get('target_object','?')}"

            triples.append((state_before, action_label, state_after))
            stats[f"task_{task_key}"] += 1
            if ep.get("success", False):
                stats["success_triples"] += 1
            else:
                stats["failure_triples"] += 1

    return triples, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Run GraphLearner on demo data")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_prefix", default="generated/results/symbolic/stacking")
    parser.add_argument("--success_only", action="store_true", default=True)
    args = parser.parse_args()

    manifest_path = Path(args.dataset_dir) / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Extract triples
    triples, stats = extract_triples_from_manifest(manifest_path)
    print(f"Extracted {len(triples)} triples")
    print(f"Stats: {json.dumps(stats, indent=2)}")

    # Show a few examples
    if triples:
        print("\nExample triples:")
        for i, (sb, action, sa) in enumerate(triples[:3]):
            print(f"  {i}: {action}")
            print(f"    before: {sorted(str(p) for p in sb)}")
            print(f"    after:  {sorted(str(p) for p in sa)}")

    # Learn graph
    learner = GraphLearner(args.output_prefix)
    learner.learn_from_triples(triples)
    learner.save()

    # Verify: check if learned structure matches expected
    graph = learner.graph
    print(f"\nGraph: {graph.number_of_nodes()} states, {graph.number_of_edges()} edges")
    print("State mapping:")
    for name, predicates in sorted(learner.state_mapping.items(), key=lambda x: int(x[0])):
        pred_strs = sorted(str(p) for p in predicates)
        print(f"  s{name}: {pred_strs}")


if __name__ == "__main__":
    main()
