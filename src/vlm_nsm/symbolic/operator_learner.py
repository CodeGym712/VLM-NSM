# OperatorLearner — extracts operator preconditions/effects from a DFA graph.
# Track A: DFA → operator definitions without ASP solver.

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


def extract_operators_from_graph(graph_json_path: str) -> dict[str, Any]:
    """Extract operator definitions from a GraphLearner JSON output.

    For each action label (e.g., "pick_place"), finds all edges with that label,
    compares state predicates before and after, and extracts:
      - preconditions: predicates present in ALL source states (intersection)
      - effects_add: predicates added (in target, not in source)
      - effects_del: predicates deleted (in source, not in target)

    Args:
        graph_json_path: Path to *_graph.json from GraphLearner.save()

    Returns:
        Dict with operator name → {preconditions, effects_add, effects_del}
    """
    with open(graph_json_path) as f:
        data = json.load(f)

    states = data["states"]
    edges = data["edges"]

    # Group edges by action label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        label = edge["label"]
        by_label[label].append(edge)

    operators = {}
    for label, label_edges in by_label.items():
        preconditions_sets = []
        add_effects_sets = []
        del_effects_sets = []

        for edge in label_edges:
            src_preds = set(states.get(str(edge["from"]), []))
            tgt_preds = set(states.get(str(edge["to"]), []))

            preconditions_sets.append(src_preds)
            add_effects_sets.append(tgt_preds - src_preds)
            del_effects_sets.append(src_preds - tgt_preds)

        # Preconditions = intersection across all edges (must hold in ALL cases)
        preconditions = sorted(set.intersection(*preconditions_sets)) if preconditions_sets else []

        # Effects = union across all edges (what CAN change)
        add_effects = sorted(set.union(*add_effects_sets)) if add_effects_sets else []
        del_effects = sorted(set.union(*del_effects_sets)) if del_effects_sets else []

        operators[label] = {
            "num_examples": len(label_edges),
            "preconditions": preconditions,
            "effects_add": add_effects,
            "effects_del": del_effects,
        }

    return operators


def verify_against_hand_coded(learned_ops: dict, verbose: bool = True) -> dict:
    """Verify learned operators match expected hand-coded definitions.

    Args:
        learned_ops: Output from extract_operators_from_graph().
        verbose: Print comparison details.

    Returns:
        Dict with match status per operator.
    """
    # Expected hand-coded MoveOn (pick_place with target)
    expected_moveon = {
        "pre": sorted([
            "('clear', 'cube1')", "('clear', 'cube2')", "('handempty',)",
            "('ontable', 'cube1')", "('ontable', 'cube2')",
        ]),
        "add": sorted([
            "('clear', 'cube1')", "('on', 'cube1', 'cube2')",
        ]),
        "del": sorted([
            "('clear', 'cube2')", "('ontable', 'cube1')",
        ]),
    }

    results = {}
    for op_name, op in learned_ops.items():
        pre_ok = set(op["preconditions"]) == set(expected_moveon["pre"])
        add_ok = set(op["effects_add"]) == set(expected_moveon["add"])
        del_ok = set(op["effects_del"]) == set(expected_moveon["del"])
        all_ok = pre_ok and add_ok and del_ok

        results[op_name] = {
            "preconditions_match": pre_ok,
            "add_effects_match": add_ok,
            "del_effects_match": del_ok,
            "all_match": all_ok,
        }

        if verbose:
            print(f"\nOperator: {op_name} ({op['num_examples']} examples)")
            print(f"  Preconditions: {'✅' if pre_ok else '❌'}")
            print(f"    Learned: {op['preconditions']}")
            print(f"    Expected: {expected_moveon['preconditions']}")
            print(f"  Add effects: {'✅' if add_ok else '❌'}")
            print(f"    Learned: {op['effects_add']}")
            print(f"    Expected: {expected_moveon['add_effects']}")
            print(f"  Del effects: {'✅' if del_ok else '❌'}")
            print(f"    Learned: {op['effects_del']}")
            print(f"    Expected: {expected_moveon['del_effects']}")
            print(f"  → {'ALL MATCH ✅' if all_ok else 'MISMATCH ❌'}")

    return results
